"""Model-free battle-state featurizer.

This holds everything needed to turn a raw STS2MCP battle state or action into
the fixed numeric feature vectors the battle agents score. It is intentionally
free of torch / model / training concerns so DQN and PPO agents can both
*compose* it. Legal-action enumeration lives in
:mod:`sts2rl.action_spaces.battle`; this module imports that package's shared
rule helpers rather than duplicating them.
"""

from __future__ import annotations

import re

from sts2rl.action_spaces.battle import (
    BATTLE_MAX_ENEMIES,
    BATTLE_MAX_HAND,
    BATTLE_MAX_POTIONS,
    action_item,
    action_target_index,
    is_playable_card,
    normalize_target_type,
    potion_slots,
    target_type_has_enemy,
)
from sts2rl.data.card import Card, CardIdentity, normalize_enchantment_id
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

BATTLE_MAX_POWERS = 259
BATTLE_CARD_FEATURES = 10
BATTLE_ENEMY_FEATURES = 11
BATTLE_PLAYER_FEATURES = 10
BATTLE_ACTION_IDENTITY_FEATURES = 3
BATTLE_ACTION_TARGET_FEATURES = 3 + BATTLE_ENEMY_FEATURES
BATTLE_ACTION_SCHEMA = "candidate_action_v3"

# Schema-critical: the tuple order defines the encode_action one-hot layout.
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
    """Encode battle states and actions as fixed-width numeric feature vectors."""

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

        slots = potion_slots(player.get("potions", []))
        for potion_index in range(self.MAX_POTIONS):
            potion = slots.get(potion_index)
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
            action_item(action, raw_state)
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
            potion = action_item(action, raw_state)
            slot = self._parse_int(action.get("slot", 0))
            features.extend(self._encode_potion(potion, slot))
        else:
            features.extend([0.0] * self.potion_slot_size)

        features.extend(self._encode_action_target(raw_state, action))
        return [float(value) for value in features]

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
        target_index = action_target_index(action, raw_state)
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
            1.0 if is_playable_card(card, energy) else 0.0,
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
        target_type = normalize_target_type(target_type)
        if target_type_has_enemy(target_type):
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
