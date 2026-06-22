"""Model-free battle-state encoder and legal-action candidate generator.

This holds everything needed to turn a raw STS2MCP battle state into the fixed
numeric feature vectors the battle agents score, plus the enumeration of legal
action candidates. It is intentionally free of torch / model / training concerns
so DQN and PPO agents can both *compose* it (see ``agents/battle/base.py``).
"""

from __future__ import annotations

import re

from sts2rl.agents.selection import (
    can_confirm_selection,
    can_select_more,
    selected_card_indices,
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

BATTLE_MAX_HAND = 10
BATTLE_MAX_POTIONS = 10
BATTLE_MAX_ENEMIES = 5
BATTLE_MAX_POWERS = 259
BATTLE_CARD_FEATURES = 10
BATTLE_ENEMY_FEATURES = 11
BATTLE_PLAYER_FEATURES = 10
BATTLE_ACTION_IDENTITY_FEATURES = 3
BATTLE_ACTION_TARGET_FEATURES = 3 + BATTLE_ENEMY_FEATURES
BATTLE_ACTION_SCHEMA = "candidate_action_v3"

BATTLE_ACTION_TYPES = (
    "end_turn",
    "play_card",
    "use_potion",
    "discard_potion",
    "combat_select_card",
    "combat_confirm_selection",
    "select_card",
    "confirm_selection",
    "cancel_selection",
)

# One-hot width over the action types; derived so it stays in lockstep with
# BATTLE_ACTION_TYPES (changing the action set changes the schema/dimensions).
BATTLE_ACTION_TYPE_FEATURES = len(BATTLE_ACTION_TYPES)


class BattleStateEncoder:
    """Encode battle states/actions and enumerate legal action candidates."""

    MAX_HAND = BATTLE_MAX_HAND
    MAX_POTIONS = BATTLE_MAX_POTIONS
    MAX_ENEMIES = BATTLE_MAX_ENEMIES
    MAX_POWERS = BATTLE_MAX_POWERS
    CARD_FEATURES = BATTLE_CARD_FEATURES
    ENEMY_FEATURES = BATTLE_ENEMY_FEATURES
    PLAYER_FEATURES = BATTLE_PLAYER_FEATURES
    ACTION_SCHEMA = BATTLE_ACTION_SCHEMA
    ACTION_TYPES = BATTLE_ACTION_TYPES
    # Per-algorithm checkpoint schema strings (kept as the existing literals so
    # battle checkpoints stay compatible). Each screen encoder declares its own.
    DQN_SCHEMA = "candidate_action_v3"
    PPO_SCHEMA = "candidate_action_ppo_v3"

    def __init__(self):
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

        features.extend(
            self._encode_power_bucket(
                player.get("status", player.get("powers", [])), self.MAX_POWERS
            )
        )
        features.extend(self._encode_relic_bucket(player.get("relics", []), self.relic_vector_size))

        return [float(value) for value in features]

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        """Encode a single action candidate or dispatched game action."""
        player = raw_state.get("player", {})
        action_type = action.get("type")
        features = [1.0 if action_type == name else 0.0 for name in self.ACTION_TYPES]

        card = (
            self._action_item(action, raw_state)
            if action_type
            in {
                "play_card",
                "combat_select_card",
                "select_card",
            }
            else None
        )
        if card is None:
            features.extend([0.0] * self.CARD_FEATURES)
            features.extend([0.0] * BATTLE_ACTION_IDENTITY_FEATURES)
        else:
            card_index = self._parse_int(
                action.get("card_index", action.get("index", card.get("index", 0)))
            )
            features.extend(self._encode_hand_card(card, card_index, player))
            features.extend(self._encode_card_identity(Card.from_raw(card).identity))

        if action_type in {"use_potion", "discard_potion"}:
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
        if state_type == "card_select" and raw_state.get("in_battle") is True:
            return self._card_select_candidates(raw_state)

        if state_type not in {"monster", "elite", "boss"}:
            return []

        if not self._is_player_play_phase(raw_state):
            return []

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

        if action_type == "discard_potion":
            return f"discard_potion:{action['slot']}"

        if action_type == "combat_select_card":
            return f"combat_select_card:{action['card_index']}"

        if action_type == "combat_confirm_selection":
            return "combat_confirm_selection"

        if action_type == "select_card":
            return f"select_card:{action['index']}"

        if action_type == "confirm_selection":
            return "confirm_selection"

        if action_type == "cancel_selection":
            return "cancel_selection"

        raise ValueError(f"Unknown game action: {action}")

    def candidate_action_vectors(self, raw_state: dict) -> list[list[float]]:
        """Encode every currently legal action candidate."""
        return [
            self.encode_action(raw_state, candidate["action"])
            for candidate in self.valid_action_candidates(raw_state)
        ]

    def _play_card_candidates(
        self, raw_state: dict, energy: int, enemies: list[dict]
    ) -> list[dict]:
        manager = CardManager.from_state_hand(raw_state)
        candidates = []
        for identity_key in manager.identity_keys():
            cards = [
                card
                for card in manager.matching_cards(identity_key)
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
                    candidates.append(
                        self._candidate(
                            targeted_action,
                            f"play_card:{card.identity_key}:target:{enemy_index}",
                        )
                    )
            else:
                candidates.append(self._candidate(action, f"play_card:{card.identity_key}:self"))
        return candidates

    def _potion_candidates(self, potions: list[dict], enemies: list[dict]) -> list[dict]:
        candidates = []
        for slot, potion in enumerate(potions[: self.MAX_POTIONS]):
            if not potion:
                continue
            potion_slot = self._potion_slot(potion, slot)
            if not 0 <= potion_slot < self.MAX_POTIONS:
                continue
            if potion.get("can_use_in_combat", True):
                action = {"type": "use_potion", "slot": potion_slot}
                if self._requires_enemy_target(potion):
                    for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
                        target = self._enemy_id_by_index(enemies, enemy_index)
                        if target is None:
                            continue
                        targeted_action = dict(action)
                        targeted_action["target_index"] = enemy_index
                        targeted_action["target"] = target
                        candidates.append(
                            self._candidate(
                                targeted_action,
                                f"use_potion:{potion_slot}:target:{enemy_index}",
                            )
                        )
                else:
                    candidates.append(self._candidate(action, f"use_potion:{potion_slot}:self"))
            # A held potion can always be discarded, even if it cannot be used in combat.
            candidates.append(
                self._candidate(
                    {"type": "discard_potion", "slot": potion_slot},
                    f"discard_potion:{potion_slot}",
                )
            )
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
            candidates.append(
                self._candidate(
                    {"type": "combat_confirm_selection"},
                    "combat_confirm_selection",
                )
            )
        return candidates

    def _card_select_candidates(self, raw_state: dict) -> list[dict]:
        card_select = raw_state.get("card_select", {})
        cards = card_select.get("cards", [])
        selected_indices = selected_card_indices(card_select)
        selected_count = selection_selected_count(
            raw_state,
            "card_select",
            len(selected_indices),
        )
        candidates = []
        if can_select_more(raw_state, "card_select", selected_count):
            for fallback_index, card in enumerate(cards):
                card_index = self._parse_int(card.get("index", fallback_index), fallback_index)
                if card_index in selected_indices:
                    continue
                action = {"type": "select_card", "index": card_index}
                candidates.append(self._candidate(action, f"select_card:{card_index}"))
        if can_confirm_selection(raw_state, "card_select", selected_count):
            candidates.append(
                self._candidate(
                    {"type": "confirm_selection"},
                    "confirm_selection",
                )
            )
        if card_select.get("can_cancel", False):
            candidates.append(
                self._candidate(
                    {"type": "cancel_selection"},
                    "cancel_selection",
                )
            )
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
            1.0
            if target_index is None and action.get("type") in {"play_card", "use_potion"}
            else 0.0,
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
            relic_id = (
                relic.get("id") or relic.get("name") if isinstance(relic, dict) else str(relic)
            )
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

        if action.get("type") == "select_card":
            card_select = raw_state.get("card_select", {})
            card_index = self._parse_int(action.get("index"), -1)
            for fallback_index, card in enumerate(card_select.get("cards", [])):
                if self._parse_int(card.get("index", fallback_index), fallback_index) == card_index:
                    return card
            return None

        if action.get("type") in {"use_potion", "discard_potion"}:
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
        if raw_state.get("state_type") in {"monster", "elite", "boss"}:
            if not self._is_player_play_phase(raw_state):
                return {"type": "proceed"}
            return {"type": "end_turn"}

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

        if raw_state.get("state_type") == "card_select":
            card_select = raw_state.get("card_select", {})
            selected_indices = selected_card_indices(card_select)
            selected_count = selection_selected_count(
                raw_state,
                "card_select",
                len(selected_indices),
            )
            if can_confirm_selection(raw_state, "card_select", selected_count):
                return {"type": "confirm_selection"}

            cards = card_select.get("cards", [])
            if cards and can_select_more(raw_state, "card_select", selected_count):
                return {
                    "type": "select_card",
                    "index": self._parse_int(cards[0].get("index", 0)),
                }

            if card_select.get("can_cancel", False):
                return {"type": "cancel_selection"}

        return {"type": "end_turn"}

    def _is_player_play_phase(self, raw_state: dict) -> bool:
        battle = raw_state.get("battle", {})
        return battle.get("turn") == "player" and battle.get("is_play_phase") is True

    def _is_playable_card(self, card: dict, energy: int) -> bool:
        return Card.from_raw(card).is_playable_with_energy(energy)

    def _requires_enemy_target(self, item: dict) -> bool:
        if item.get("requires_target", False):
            return True
        target_type = self._normalize_target_type(item.get("target_type", item.get("target", "")))
        return (
            self._target_type_has_enemy(target_type)
            and "random" not in target_type
            and "all" not in target_type
        )

    def _normalize_target_type(self, value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _target_type_has_enemy(self, target_type: str) -> bool:
        return "enemy" in target_type or "enemies" in target_type

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

            return self._parse_int(intent.get("damage", 0)), self._parse_int(
                intent.get("hit_count", 1), 1
            )

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
        target_type = self._normalize_target_type(target_type)
        if self._target_type_has_enemy(target_type):
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
