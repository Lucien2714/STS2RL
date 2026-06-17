"""Trainable candidate-action DQN battle agent and battle-state encoder."""

from __future__ import annotations

from collections import deque
import logging
import random
import re

from sts2rl.agents.base import BattleAgent as BattleAgentBase
from sts2rl.agents.selection import (
  can_confirm_selection,
  can_select_more,
  selection_selected_count,
)
from sts2rl.data.card import Card, CardIdentity, CardManager, normalize_enchantment_id
from sts2rl.data.loader import (
  get_card_index,
  get_card_map_size,
  get_data_index_or_default,
  get_data_map_size,
  get_intent_index,
  get_monster_index,
  get_potion_index,
  get_power_index,
  get_relic_index,
)

import torch
from torch import nn


logger = logging.getLogger(__name__)


BATTLE_MAX_HAND = 10
BATTLE_MAX_POTIONS = 10
BATTLE_MAX_ENEMIES = 5
BATTLE_MAX_POWERS = 259
BATTLE_CARD_FEATURES = 10
BATTLE_ENEMY_FEATURES = 11
BATTLE_PLAYER_FEATURES = 10
BATTLE_ACTION_TYPE_FEATURES = 5
BATTLE_ACTION_IDENTITY_FEATURES = 3
BATTLE_ACTION_TARGET_FEATURES = 3 + BATTLE_ENEMY_FEATURES
BATTLE_ACTION_SCHEMA = "candidate_action_v1"


class BattleQNetwork(nn.Module):
  """Small fully connected Q-network for state/action candidate scoring."""

  def __init__(self, input_size: int, hidden_size: int = 256):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1),
    )

  def forward(self, state_action):
    """Return one Q-value for each encoded state/action pair."""
    return self.net(state_action).squeeze(-1)


class BattleDQNAgent(BattleAgentBase):
  """Battle agent that scores legal action candidates with a DQN."""

  MAX_HAND = BATTLE_MAX_HAND
  MAX_POTIONS = BATTLE_MAX_POTIONS
  MAX_ENEMIES = BATTLE_MAX_ENEMIES
  MAX_POWERS = BATTLE_MAX_POWERS
  CARD_FEATURES = BATTLE_CARD_FEATURES
  ENEMY_FEATURES = BATTLE_ENEMY_FEATURES
  PLAYER_FEATURES = BATTLE_PLAYER_FEATURES
  ACTION_SCHEMA = BATTLE_ACTION_SCHEMA

  ACTION_TYPES = (
    "end_turn",
    "play_card",
    "use_potion",
    "combat_select_card",
    "combat_confirm_selection",
  )

  def __init__(
    self,
    gamma=0.8,
    epsilon=1.0,
    learning_rate=0.00025,
    epsilon_decay=0.9999,
    epsilon_min=0.10,
    batch_size=64,
    memory_size=100000,
    update_freq=4,
    update_freq_target=2000,
    hidden_size=256,
    device=None,
  ):
    self.card_vector_size = get_card_map_size() * 2
    self.enchantment_vector_size = get_data_map_size("enchantments")
    self.intent_vector_size = get_data_map_size("intents")
    self.potion_vector_size = get_data_map_size("potions")
    self.relic_vector_size = get_data_map_size("relics")
    self.potion_slot_size = 1 + self.potion_vector_size + 2
    self.action_feature_size = (
      BATTLE_ACTION_TYPE_FEATURES
      + self.CARD_FEATURES
      + BATTLE_ACTION_IDENTITY_FEATURES
      + self.potion_slot_size
      + BATTLE_ACTION_TARGET_FEATURES
    )
    self.state_size = (
      1
      + self.PLAYER_FEATURES
      + self.MAX_HAND * self.CARD_FEATURES
      + self.card_vector_size * 3
      + self.MAX_ENEMIES * self.ENEMY_FEATURES
      + self.MAX_POTIONS * self.potion_slot_size
      + self.MAX_POWERS
      + self.relic_vector_size
    )
    self.model_input_size = self.state_size + self.action_feature_size

    self.update_freq = update_freq
    self.update_freq_target = update_freq_target
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.replay_buffer = deque(maxlen=memory_size)
    self.trained_steps = 0
    self.learn_steps = 0
    self.last_action_selection = {}

    self.device = self._resolve_device(device)
    self.model = BattleQNetwork(self.model_input_size, hidden_size).to(self.device)
    self.target_model = BattleQNetwork(self.model_input_size, hidden_size).to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    self.loss_fn = nn.SmoothL1Loss()

  def choose_action(self, raw_state: dict, training: bool = True) -> dict:
    """Choose a legal battle action using exploration or candidate Q-values."""
    candidates = self.valid_action_candidates(raw_state)
    if not candidates:
      action = self._fallback_action(raw_state)
      self.last_action_selection = {
        "method": "fallback",
        "reason": "no_valid_candidates",
        "training": training,
        "epsilon": self.epsilon,
        "action_key": None,
        "q": None,
        "valid_action_count": 0,
      }
      return action

    if training and random.random() < self.epsilon:
      candidate = random.choice(candidates)
      self.last_action_selection = {
        "method": "epsilon_random",
        "reason": "epsilon_exploration",
        "training": training,
        "epsilon": self.epsilon,
        "action_key": candidate["action_key"],
        "q": None,
        "valid_action_count": len(candidates),
      }
      return self._public_action(candidate)

    q_values = self._score_candidates(raw_state, candidates, self.model)
    best_index = max(range(len(candidates)), key=lambda index: q_values[index])
    candidate = candidates[best_index]
    self.last_action_selection = {
      "method": "greedy_q",
      "reason": "candidate_argmax",
      "training": training,
      "epsilon": self.epsilon,
      "action_key": candidate["action_key"],
      "q": q_values[best_index],
      "valid_action_count": len(candidates),
    }
    return self._public_action(candidate)

  def remember(
    self,
    state,
    action_vector,
    reward: float,
    next_state,
    done: bool,
    next_action_vectors,
  ) -> None:
    """Store one transition in replay memory."""
    self.replay_buffer.append((
      list(state),
      list(action_vector),
      float(reward),
      list(next_state),
      bool(done),
      [list(vector) for vector in next_action_vectors],
    ))
    self.trained_steps += 1

  def train_step(self) -> float | None:
    """Run one replay update when enough samples are available."""
    if len(self.replay_buffer) < self.batch_size:
      return None

    batch = random.sample(self.replay_buffer, self.batch_size)
    states, actions, rewards, next_states, dones, next_actions = zip(*batch)

    state_action_inputs = [
      list(state) + list(action)
      for state, action in zip(states, actions)
    ]
    current_inputs = torch.tensor(
      state_action_inputs,
      dtype=torch.float32,
      device=self.device,
    )
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

    current_q = self.model(current_inputs)

    max_next_q_values = []
    with torch.no_grad():
      for next_state, done, candidate_vectors in zip(next_states, dones, next_actions):
        if done or not candidate_vectors:
          max_next_q_values.append(0.0)
          continue
        next_inputs = [
          list(next_state) + list(candidate_vector)
          for candidate_vector in candidate_vectors
        ]
        next_tensor = torch.tensor(next_inputs, dtype=torch.float32, device=self.device)
        next_q = self.target_model(next_tensor)
        max_next_q_values.append(float(next_q.max().item()))

      max_next_q = torch.tensor(max_next_q_values, dtype=torch.float32, device=self.device)
      target_q = rewards_tensor + self.gamma * max_next_q * (~dones_tensor).float()

    loss = self.loss_fn(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.learn_steps += 1
    if self.learn_steps % self.update_freq_target == 0:
      self.target_model.load_state_dict(self.model.state_dict())

    if self.learn_steps % self.update_freq == 0:
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    return float(loss.item())

  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    """Encode a raw battle state into a fixed-width numeric feature vector."""
    battle = raw_state.get("battle", {})
    player = raw_state.get("player", {})

    features = []
    features.append(self._scale(raw_state.get("round", battle.get("round", 0)), 100))
    features.extend(self._encode_player(player))

    hand = player.get("hand", raw_state.get("hand", []))
    for card_index in range(self.MAX_HAND):
      card = hand[card_index] if card_index < len(hand) else None
      features.extend(self._encode_hand_card(card, card_index, player))

    features.extend(self._encode_card_pile(player.get("draw_pile", [])))
    features.extend(self._encode_card_pile(player.get("discard_pile", [])))
    features.extend(self._encode_card_pile(player.get("exhaust_pile", [])))

    enemies = battle.get("enemies", raw_state.get("enemies", []))
    for enemy_index in range(self.MAX_ENEMIES):
      enemy = enemies[enemy_index] if enemy_index < len(enemies) else None
      features.extend(self._encode_enemy(enemy))

    potion_slots = self._potion_slots(player.get("potions", []))
    for potion_index in range(self.MAX_POTIONS):
      potion = potion_slots.get(potion_index)
      features.extend(self._encode_potion(potion, potion_index))

    features.extend(self._encode_power_bucket(player.get("status", player.get("powers", [])), self.MAX_POWERS))
    features.extend(self._encode_relic_bucket(player.get("relics", []), self.relic_vector_size))

    return [float(value) for value in features]

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    """Encode a single action candidate or dispatched game action."""
    player = raw_state.get("player", {})
    action_type = action.get("type")
    features = [1.0 if action_type == name else 0.0 for name in self.ACTION_TYPES]

    card = self._action_item(action, raw_state) if action_type in {
      "play_card",
      "combat_select_card",
    } else None
    if card is None:
      features.extend([0.0] * self.CARD_FEATURES)
      features.extend([0.0] * BATTLE_ACTION_IDENTITY_FEATURES)
    else:
      card_index = self._parse_int(action.get("card_index", card.get("index", 0)))
      features.extend(self._encode_hand_card(card, card_index, player))
      features.extend(self._encode_card_identity(Card.from_raw(card).identity))

    if action_type == "use_potion":
      potion = self._action_item(action, raw_state)
      slot = self._parse_int(action.get("slot", 0))
      features.extend(self._encode_potion(potion, slot))
    else:
      features.extend([0.0] * self.potion_slot_size)

    features.extend(self._encode_action_target(raw_state, action))
    return [float(value) for value in features]

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    """Return legal action candidates for the current battle state."""
    state_type = raw_state.get("state_type")
    if state_type == "hand_select":
      return self._hand_select_candidates(raw_state)

    if state_type not in {"monster", "elite", "boss"}:
      return [self._candidate({"type": "end_turn"}, "end_turn")]

    battle = raw_state.get("battle", {})
    player = raw_state.get("player", {})
    enemies = battle.get("enemies", raw_state.get("enemies", []))
    potions = player.get("potions", [])
    energy = self._parse_int(player.get("energy", raw_state.get("energy", 0)))

    candidates = [self._candidate({"type": "end_turn"}, "end_turn")]
    candidates.extend(self._play_card_candidates(raw_state, energy, enemies))
    candidates.extend(self._potion_candidates(potions, enemies))
    return candidates

  def valid_action_mask(self, raw_state: dict) -> list[bool]:
    """Return a dynamic all-true mask for compatibility with older callers."""
    return [True for _ in self.valid_action_candidates(raw_state)]

  def current_q_values(self, raw_state: dict, selected_action: dict | None = None) -> dict:
    """Return candidate Q-values for dashboards and evaluation telemetry."""
    candidates = self.valid_action_candidates(raw_state)
    selected_key = None
    if selected_action is not None:
      try:
        selected_key = self.action_key(selected_action, raw_state)
      except (KeyError, ValueError):
        selected_key = None

    q_values = self._score_candidates(raw_state, candidates, self.model)
    actions = []
    best_valid = None
    for candidate, q_value in zip(candidates, q_values):
      action = {
        "id": candidate["action_key"],
        "key": candidate["action_key"],
        "q": self._safe_float(q_value),
        "masked_q": self._safe_float(q_value),
        "valid": True,
        "selected": candidate["action_key"] == selected_key,
      }
      actions.append(action)
      if action["q"] is not None and (best_valid is None or action["q"] > best_valid["q"]):
        best_valid = action

    return {
      "available": True,
      "screen_type": raw_state.get("state_type"),
      "selected_action_id": selected_key,
      "selected_q": next((action["q"] for action in actions if action["selected"]), None),
      "best_valid_action": best_valid,
      "actions": actions,
    }

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    """Return a stable key for an action within the current state."""
    if action.get("action_key"):
      return str(action["action_key"])

    action_type = action.get("type")
    if action_type == "end_turn":
      return "end_turn"

    if action_type == "play_card":
      identity_key = action.get("card_identity")
      if identity_key is None and raw_state is not None:
        item = self._action_item(action, raw_state)
        if item is not None:
          identity_key = Card.from_raw(item).identity_key
      identity_key = identity_key or "UNKNOWN_CARD"
      target_index = self._action_target_index(action, raw_state)
      if target_index is not None:
        return f"play_card:{identity_key}:target:{target_index}"
      return f"play_card:{identity_key}:self"

    if action_type == "use_potion":
      target_index = self._action_target_index(action, raw_state)
      if target_index is not None:
        return f"use_potion:{action['slot']}:target:{target_index}"
      return f"use_potion:{action['slot']}:self"

    if action_type == "combat_select_card":
      return f"combat_select_card:{action['card_index']}"

    if action_type == "combat_confirm_selection":
      return "combat_confirm_selection"

    raise ValueError(f"Unknown game action: {action}")

  def candidate_action_vectors(self, raw_state: dict) -> list[list[float]]:
    """Encode every currently legal action candidate."""
    return [
      self.encode_action(raw_state, candidate["action"])
      for candidate in self.valid_action_candidates(raw_state)
    ]

  def _score_candidates(self, raw_state: dict, candidates: list[dict], model) -> list[float]:
    if not candidates:
      return []
    state_vector = self.encode_state(raw_state)
    inputs = [
      state_vector + self.encode_action(raw_state, candidate["action"])
      for candidate in candidates
    ]
    with torch.no_grad():
      input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
      q_tensor = model(input_tensor).detach().cpu()
    return [float(value) for value in q_tensor.tolist()]

  def _play_card_candidates(self, raw_state: dict, energy: int, enemies: list[dict]) -> list[dict]:
    manager = CardManager.from_state_hand(raw_state)
    candidates = []
    for identity_key in manager.identity_keys():
      cards = [
        card for card in manager.matching_cards(identity_key)
        if card.is_playable_with_energy(energy)
      ]
      if not cards:
        continue
      card = self._best_card(cards, energy)
      if card.index is None:
        continue
      action = {
        "type": "play_card",
        "card_index": card.index,
        "card_identity": card.identity_key,
      }
      if self._requires_enemy_target(card.raw):
        for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
          target = self._enemy_id_by_index(enemies, enemy_index)
          if target is None:
            continue
          targeted_action = dict(action)
          targeted_action["target_index"] = enemy_index
          targeted_action["target"] = target
          candidates.append(self._candidate(
            targeted_action,
            f"play_card:{card.identity_key}:target:{enemy_index}",
          ))
      else:
        candidates.append(self._candidate(action, f"play_card:{card.identity_key}:self"))
    return candidates

  def _potion_candidates(self, potions: list[dict], enemies: list[dict]) -> list[dict]:
    candidates = []
    for slot, potion in enumerate(potions[: self.MAX_POTIONS]):
      if not potion or not potion.get("can_use_in_combat", True):
        continue
      potion_slot = self._potion_slot(potion, slot)
      if not 0 <= potion_slot < self.MAX_POTIONS:
        continue
      action = {"type": "use_potion", "slot": potion_slot}
      if self._requires_enemy_target(potion):
        for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
          target = self._enemy_id_by_index(enemies, enemy_index)
          if target is None:
            continue
          targeted_action = dict(action)
          targeted_action["target_index"] = enemy_index
          targeted_action["target"] = target
          candidates.append(self._candidate(
            targeted_action,
            f"use_potion:{potion_slot}:target:{enemy_index}",
          ))
      else:
        candidates.append(self._candidate(action, f"use_potion:{potion_slot}:self"))
    return candidates

  def _hand_select_candidates(self, raw_state: dict) -> list[dict]:
    hand_select = raw_state.get("hand_select", {})
    cards = hand_select.get("cards", [])
    selected_indices = {
      self._parse_int(card.get("index", -1), -1)
      for card in hand_select.get("selected_cards", [])
    }
    selected_count = selection_selected_count(
      raw_state,
      "hand_select",
      len(selected_indices),
    )
    candidates = []
    if can_select_more(raw_state, "hand_select", selected_count):
      for card in cards[: self.MAX_HAND]:
        card_index = self._parse_int(card.get("index", len(cards)))
        if card_index in selected_indices:
          continue
        if 0 <= card_index < self.MAX_HAND:
          action = {"type": "combat_select_card", "card_index": card_index}
          candidates.append(self._candidate(action, f"combat_select_card:{card_index}"))
    if can_confirm_selection(raw_state, "hand_select", selected_count):
      candidates.append(self._candidate(
        {"type": "combat_confirm_selection"},
        "combat_confirm_selection",
      ))
    return candidates

  def _candidate(self, action: dict, key: str) -> dict:
    action = dict(action)
    action["action_key"] = key
    return {"action": action, "action_key": key}

  def _public_action(self, candidate: dict) -> dict:
    return dict(candidate["action"])

  def _best_card(self, cards: list[Card], energy: int) -> Card:
    return sorted(
      cards,
      key=lambda card: (
        card.cost_for_energy(energy),
        -card.current_upgrade_level,
        card.index if card.index is not None else 999,
      ),
    )[0]

  def _encode_card_identity(self, identity: CardIdentity) -> list[float]:
    enchantment_index = get_data_index_or_default(
      "enchantments",
      normalize_enchantment_id(identity.enchantment_id),
      default=-1,
    )
    return [
      self._scale(identity.upgrade_level, 10),
      self._scale(enchantment_index + 1, max(1, self.enchantment_vector_size)),
      1.0 if identity.enchantment_id else 0.0,
    ]

  def _encode_action_target(self, raw_state: dict, action: dict) -> list[float]:
    target_index = self._action_target_index(action, raw_state)
    features = [
      1.0 if target_index is None and action.get("type") in {"play_card", "use_potion"} else 0.0,
      1.0 if target_index is not None else 0.0,
      self._scale(target_index if target_index is not None else 0, self.MAX_ENEMIES),
    ]
    if target_index is None:
      features.extend([0.0] * self.ENEMY_FEATURES)
      return features
    enemies = raw_state.get("battle", {}).get("enemies", raw_state.get("enemies", []))
    enemy = enemies[target_index] if 0 <= target_index < len(enemies) else None
    features.extend(self._encode_enemy(enemy))
    return features

  def _encode_player(self, player: dict) -> list[float]:
    max_hp = max(1, self._parse_int(player.get("max_hp", 1)))
    max_energy = max(1, self._parse_int(player.get("max_energy", 3)))
    return [
      self._scale(player.get("max_hp", 0), 200),
      self._scale(player.get("hp", 0), max_hp),
      self._scale(player.get("block", 0), 100),
      self._scale(player.get("energy", 0), max_energy),
      self._scale(player.get("orb_slots", player.get("orb_slot", 0)), 10),
      self._scale(len(player.get("status", player.get("powers", []))), 50),
      self._scale(len(player.get("hand", [])), self.MAX_HAND),
      self._scale(player.get("draw_pile_count", len(player.get("draw_pile", []))), 60),
      self._scale(player.get("discard_pile_count", len(player.get("discard_pile", []))), 60),
      self._scale(player.get("gold", 0), 999),
    ]

  def _encode_hand_card(self, card: dict | None, card_index: int, player: dict) -> list[float]:
    if card is None:
      return [0.0] * self.CARD_FEATURES

    energy = self._parse_int(player.get("energy", 0))
    card_id = get_card_index(card.get("id") or card.get("name"), default=-1)
    card_type = str(card.get("type", "")).lower()
    target_type = str(card.get("target_type", card.get("target", ""))).lower()

    return [
      1.0,
      self._scale(card_id + 1, max(1, get_card_map_size())),
      self._scale(card_index, self.MAX_HAND),
      self._scale(self._parse_cost(card.get("cost", 0), energy), 5),
      self._scale(self._parse_cost(card.get("star_cost", 0), 0), 5),
      self._card_type_value(card_type),
      1.0 if card.get("is_upgraded", card.get("upgraded", False)) else 0.0,
      1.0 if self._is_playable_card(card, energy) else 0.0,
      self._target_type_value(target_type),
      self._scale(self._estimated_damage(card), 100),
    ]

  def _encode_card_pile(self, pile: list) -> list[float]:
    vector = [0.0 for _ in range(self.card_vector_size)]
    for card in pile:
      if isinstance(card, dict):
        card_id = card.get("id") or card.get("name")
        upgraded = bool(card.get("is_upgraded", card.get("upgraded", False)))
      else:
        card_id = str(card)
        upgraded = False

      card_index = get_card_index(card_id, default=-1)
      if card_index < 0:
        continue

      vector[card_index * 2 + int(upgraded)] += 1.0

    return [min(value / 10.0, 1.0) for value in vector]

  def _encode_enemy(self, enemy: dict | None) -> list[float]:
    if enemy is None:
      return [0.0] * self.ENEMY_FEATURES

    max_hp = max(1, self._parse_int(enemy.get("max_hp", 1)))
    enemy_id = get_monster_index(self._monster_lookup_id(enemy), default=-1)
    attack_intent, support_intent = self._enemy_intents(enemy)
    intent_damage, intent_hit_count = self._intent_attack_parts(enemy)

    return [
      1.0,
      self._scale(enemy_id + 1, 200),
      self._scale(enemy.get("max_hp", 0), 500),
      self._scale(enemy.get("hp", 0), max_hp),
      self._scale_intent(attack_intent),
      self._scale_intent(support_intent),
      self._scale(len(enemy.get("status", enemy.get("powers", []))), 50),
      self._scale(intent_damage, 150),
      self._scale(intent_hit_count, 10),
      self._scale(enemy.get("block", 0), 150),
      1.0 if self._parse_int(enemy.get("hp", 0)) > 0 else 0.0,
    ]

  def _encode_potion(self, potion: dict | None, slot: int) -> list[float]:
    features = [0.0 for _ in range(self.potion_slot_size)]
    if potion is None:
      return features

    potion_id = get_potion_index(potion.get("id") or potion.get("name"), default=-1)
    features[0] = 1.0
    if 0 <= potion_id < self.potion_vector_size:
      features[1 + potion_id] = 1.0
    features[1 + self.potion_vector_size] = self._scale(slot, self.MAX_POTIONS)
    features[1 + self.potion_vector_size + 1] = (
      1.0 if potion.get("can_use_in_combat", True) else 0.0
    )
    return features

  def _encode_power_bucket(self, powers: list, bucket_size: int) -> list[float]:
    vector = [0.0 for _ in range(bucket_size)]
    for power in powers:
      if isinstance(power, dict):
        power_id = power.get("id") or power.get("name")
        amount = max(1, self._parse_int(power.get("amount", 1)))
      else:
        power_id = str(power)
        amount = 1

      index = get_power_index(power_id, default=-1)
      if 0 <= index < bucket_size:
        vector[index] += amount

    return [min(value / 10.0, 1.0) for value in vector]

  def _encode_relic_bucket(self, relics: list, bucket_size: int) -> list[float]:
    vector = [0.0 for _ in range(bucket_size)]
    for relic in relics:
      relic_id = relic.get("id") or relic.get("name") if isinstance(relic, dict) else str(relic)
      index = get_relic_index(relic_id, default=-1)
      if 0 <= index < bucket_size:
        vector[index] = 1.0

    return vector

  def _action_item(self, action: dict, raw_state: dict) -> dict | None:
    player = raw_state.get("player", {})
    if action.get("type") == "play_card":
      hand = player.get("hand", raw_state.get("hand", []))
      card_index = self._parse_int(action.get("card_index"), -1)
      return hand[card_index] if 0 <= card_index < len(hand) else None

    if action.get("type") == "combat_select_card":
      hand_select = raw_state.get("hand_select", {})
      card_index = self._parse_int(action.get("card_index"), -1)
      for card in hand_select.get("cards", []):
        if self._parse_int(card.get("index", -1), -1) == card_index:
          return card
      return None

    if action.get("type") == "use_potion":
      potions = player.get("potions", [])
      slot = self._parse_int(action.get("slot"), -1)
      for potion_index, potion in enumerate(potions):
        if self._potion_slot(potion, potion_index) == slot:
          return potion
      return None

    return None

  def _potion_slot(self, potion: dict, fallback_slot: int) -> int:
    return self._parse_int(potion.get("slot", fallback_slot), fallback_slot)

  def _potion_slots(self, potions: list[dict]) -> dict[int, dict]:
    slots = {}
    for potion_index, potion in enumerate(potions):
      if not potion:
        continue
      slot = self._potion_slot(potion, potion_index)
      if 0 <= slot < self.MAX_POTIONS:
        slots[slot] = potion
    return slots

  def _fallback_action(self, raw_state: dict) -> dict:
    if raw_state.get("state_type") == "hand_select":
      hand_select = raw_state.get("hand_select", {})
      selected_indices = {
        self._parse_int(card.get("index", -1), -1)
        for card in hand_select.get("selected_cards", [])
      }
      selected_count = selection_selected_count(
        raw_state,
        "hand_select",
        len(selected_indices),
      )
      if can_confirm_selection(raw_state, "hand_select", selected_count):
        return {"type": "combat_confirm_selection"}

      cards = hand_select.get("cards", [])
      if cards and can_select_more(raw_state, "hand_select", selected_count):
        return {
          "type": "combat_select_card",
          "card_index": self._parse_int(cards[0].get("index", 0)),
        }

    return {"type": "end_turn"}

  def _is_playable_card(self, card: dict, energy: int) -> bool:
    return Card.from_raw(card).is_playable_with_energy(energy)

  def _requires_enemy_target(self, item: dict) -> bool:
    if item.get("requires_target", False):
      return True
    target_type = str(item.get("target_type", item.get("target", ""))).lower()
    return target_type in {"anyenemy", "enemy"}

  def _enemy_id_by_index(self, enemies: list[dict], enemy_index: object) -> str | None:
    enemy_index = self._parse_int(enemy_index, -1)
    if not 0 <= enemy_index < min(len(enemies), self.MAX_ENEMIES):
      return None
    enemy = enemies[enemy_index]
    if self._parse_int(enemy.get("hp", 0)) <= 0:
      return None
    return enemy.get("entity_id") or enemy.get("id")

  def _enemy_index_by_id(self, enemies: list[dict], enemy_id: object) -> int | None:
    if enemy_id is None:
      return None
    enemy_id = str(enemy_id)
    for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
      if enemy_id in {str(enemy.get("entity_id")), str(enemy.get("id"))}:
        return enemy_index
    return None

  def _action_target_index(self, action: dict, raw_state: dict | None = None) -> int | None:
    if action.get("target_index") is not None:
      target_index = self._parse_int(action.get("target_index"), -1)
      if 0 <= target_index < self.MAX_ENEMIES:
        return target_index

    if raw_state is None or action.get("target") is None:
      return None

    enemies = raw_state.get("battle", {}).get("enemies", raw_state.get("enemies", []))
    return self._enemy_index_by_id(enemies, action.get("target"))

  def _monster_lookup_id(self, enemy: dict) -> str | None:
    monster_id = enemy.get("id") or enemy.get("entity_id")
    if monster_id is not None:
      return re.sub(r"_\d+$", "", str(monster_id))
    return enemy.get("name")

  def _enemy_intents(self, enemy: dict) -> tuple[str | None, str | None]:
    attack_intent = None
    support_intent = None

    for intent in enemy.get("intents", []):
      intent_id = self._intent_id(intent)
      if intent_id is None:
        continue

      if self._is_attack_intent(intent):
        attack_intent = intent_id
      else:
        support_intent = intent_id

    if attack_intent is None and support_intent is None:
      fallback_intent = enemy.get("intent") or enemy.get("intent_id")
      if fallback_intent is not None:
        if self._is_attack_intent(fallback_intent):
          attack_intent = str(fallback_intent)
        else:
          support_intent = str(fallback_intent)

    return attack_intent, support_intent

  def _intent_id(self, intent: object) -> str | None:
    if isinstance(intent, dict):
      return intent.get("id") or intent.get("type") or intent.get("label")
    if intent is None:
      return None
    return str(intent)

  def _intent_attack_parts(self, enemy: dict) -> tuple[int, int]:
    for intent in enemy.get("intents", []):
      if not isinstance(intent, dict) or not self._is_attack_intent(intent):
        continue

      label = str(intent.get("label", ""))
      repeated_damage = re.fullmatch(r"(\d+)\s*x\s*(\d+)", label, flags=re.IGNORECASE)
      if repeated_damage:
        return int(repeated_damage.group(1)), int(repeated_damage.group(2))

      if label.isdigit():
        return int(label), 1

      return self._parse_int(intent.get("damage", 0)), self._parse_int(intent.get("hit_count", 1), 1)

    return 0, 0

  def _is_attack_intent(self, intent: object) -> bool:
    if isinstance(intent, dict):
      intent_type = intent.get("type", intent.get("id", ""))
    else:
      intent_type = intent
    return str(intent_type).lower() == "attack"

  def _scale_intent(self, intent: str | None) -> float:
    intent_index = get_intent_index(intent, default=-1)
    return self._scale(intent_index + 1, self.intent_vector_size)

  def _estimated_damage(self, card: dict) -> int:
    if card.get("damage") is not None:
      return self._parse_int(card.get("damage"))
    description = str(card.get("description", ""))
    match = re.search(r"Deal\s+(\d+)", description, flags=re.IGNORECASE)
    return self._parse_int(match.group(1)) if match else 0

  def _card_type_value(self, card_type: str) -> float:
    values = {
      "attack": 0.2,
      "skill": 0.4,
      "power": 0.6,
      "status": 0.8,
      "curse": 1.0,
    }
    return values.get(card_type, 0.0)

  def _target_type_value(self, target_type: str) -> float:
    if "enemy" in target_type:
      return 0.33
    if "self" in target_type:
      return 0.66
    if target_type in {"none", ""}:
      return 0.0
    return 1.0

  def _parse_cost(self, cost: object, energy: int) -> int:
    if cost is None:
      return 0
    if isinstance(cost, str) and cost.strip().upper() == "X":
      return energy
    return self._parse_int(cost)

  def _parse_int(self, value: object, default: int = 0) -> int:
    try:
      return int(value)
    except (TypeError, ValueError):
      return default

  def _scale(self, value: object, denominator: int | float) -> float:
    denominator = max(float(denominator), 1.0)
    return max(0.0, min(float(self._parse_int(value)) / denominator, 1.0))

  def _safe_float(self, value: float) -> float | None:
    try:
      value = float(value)
    except (TypeError, ValueError):
      return None
    return value if value == value and value not in {float("inf"), float("-inf")} else None

  def _resolve_device(self, device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

  def save(self, path: str) -> None:
    """Save model, optimizer, and exploration state to a checkpoint."""
    torch.save(
      {
        "action_schema": self.ACTION_SCHEMA,
        "model_state_dict": self.model.state_dict(),
        "target_model_state_dict": self.target_model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epsilon": self.epsilon,
        "trained_steps": self.trained_steps,
        "learn_steps": self.learn_steps,
        "state_size": self.state_size,
        "action_feature_size": self.action_feature_size,
        "model_input_size": self.model_input_size,
      },
      path,
    )

  def load(self, path: str) -> None:
    """Load model, optimizer, and exploration state from a checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)
    action_schema = checkpoint.get("action_schema")
    if action_schema != self.ACTION_SCHEMA:
      raise ValueError(
        f"Incompatible battle checkpoint action_schema={action_schema!r}; "
        f"expected {self.ACTION_SCHEMA!r}. Retrain or use a matching checkpoint."
      )
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.epsilon = checkpoint.get("epsilon", self.epsilon)
    self.trained_steps = checkpoint.get(
      "trained_steps",
      checkpoint.get("trained_step", checkpoint.get("learn_steps", self.trained_steps)),
    )
    self.learn_steps = checkpoint.get("learn_steps", self.learn_steps)
    logger.info(
      "Loaded DQN checkpoint from %s: trained_steps=%d learn_steps=%d epsilon=%.4f schema=%s",
      path,
      self.trained_steps,
      self.learn_steps,
      self.epsilon,
      self.ACTION_SCHEMA,
    )


DQNBattleAgent = BattleDQNAgent
