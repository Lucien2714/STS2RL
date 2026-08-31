"""Battle action space: legal-action enumeration and rule helpers for combat.

Holds the game-rule half of what used to live in ``BattleStateEncoder``:
candidate enumeration, action keying, and the raw-state helpers shared with the
battle featurizer (which imports them from here — ``encoders → action_spaces``
is the sanctioned dependency direction). Torch-free by package rule.
"""

from __future__ import annotations

import re

from sts2rl.action_spaces.base import REFRESH_STATE_ACTION, ActionSpace, parse_int
from sts2rl.action_spaces.selection import (
    can_confirm_selection,
    can_select_more,
    selected_card_indices,
    selection_selected_count,
)
from sts2rl.data.card import Card, CardManager

# Rule-side caps (also bound the featurizer's fixed-width layouts, which import
# them from here).
BATTLE_MAX_HAND = 10
BATTLE_MAX_POTIONS = 10
BATTLE_MAX_ENEMIES = 5


# --- raw-state helpers shared by enumeration and encoding ----------------------


def potion_slot(potion: dict, fallback_slot: int) -> int:
    """Return a potion's slot, falling back to its list position."""
    return parse_int(potion.get("slot", fallback_slot), fallback_slot)


def potion_slots(potions: list[dict]) -> dict[int, dict]:
    """Map slot index -> potion for every held potion within the slot cap."""
    slots = {}
    for potion_index, potion in enumerate(potions):
        if not potion:
            continue
        slot = potion_slot(potion, potion_index)
        if 0 <= slot < BATTLE_MAX_POTIONS:
            slots[slot] = potion
    return slots


def action_item(action: dict, raw_state: dict) -> dict | None:
    """Resolve the hand card / selection card / potion an action refers to."""
    player = raw_state.get("player", {})
    if action.get("type") == "play_card":
        hand = player.get("hand", raw_state.get("hand", []))
        card_index = parse_int(action.get("card_index"), -1)
        return hand[card_index] if 0 <= card_index < len(hand) else None

    if action.get("type") == "combat_select_card":
        hand_select = raw_state.get("hand_select", {})
        card_index = parse_int(action.get("card_index"), -1)
        for card in hand_select.get("cards", []):
            if parse_int(card.get("index", -1), -1) == card_index:
                return card
        return None

    if action.get("type") == "select_card":
        card_select = raw_state.get("card_select", {})
        card_index = parse_int(action.get("index"), -1)
        for fallback_index, card in enumerate(card_select.get("cards", [])):
            if parse_int(card.get("index", fallback_index), fallback_index) == card_index:
                return card
        return None

    if action.get("type") in {"use_potion", "discard_potion"}:
        potions = player.get("potions", [])
        slot = parse_int(action.get("slot"), -1)
        for potion_index, potion in enumerate(potions):
            if potion_slot(potion, potion_index) == slot:
                return potion
        return None

    return None


def normalize_target_type(value: object) -> str:
    """Normalize a target-type string for substring matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def target_type_has_enemy(target_type: str) -> bool:
    """Return whether a normalized target type refers to enemies."""
    return "enemy" in target_type or "enemies" in target_type


def requires_enemy_target(item: dict) -> bool:
    """Return whether a card/potion needs an explicit enemy target."""
    if item.get("requires_target", False):
        return True
    target_type = normalize_target_type(item.get("target_type", item.get("target", "")))
    return (
        target_type_has_enemy(target_type)
        and "random" not in target_type
        and "all" not in target_type
    )


def is_playable_card(card: dict, energy: int) -> bool:
    """Return whether a hand card is playable with the given energy."""
    return Card.from_raw(card).is_playable_with_energy(energy)


def enemy_id_by_index(enemies: list[dict], enemy_index: object) -> str | None:
    """Return the entity id of a living enemy at an index, if any."""
    enemy_index = parse_int(enemy_index, -1)
    if not 0 <= enemy_index < min(len(enemies), BATTLE_MAX_ENEMIES):
        return None
    enemy = enemies[enemy_index]
    if parse_int(enemy.get("hp", 0)) <= 0:
        return None
    return enemy.get("entity_id") or enemy.get("id")


def enemy_index_by_id(enemies: list[dict], enemy_id: object) -> int | None:
    """Return the index of the enemy with a given entity id, if present."""
    if enemy_id is None:
        return None
    enemy_id = str(enemy_id)
    for enemy_index, enemy in enumerate(enemies[:BATTLE_MAX_ENEMIES]):
        if enemy_id in {str(enemy.get("entity_id")), str(enemy.get("id"))}:
            return enemy_index
    return None


def action_target_index(action: dict, raw_state: dict | None = None) -> int | None:
    """Resolve an action's enemy-target index from target_index or entity id."""
    if action.get("target_index") is not None:
        target_index = parse_int(action.get("target_index"), -1)
        if 0 <= target_index < BATTLE_MAX_ENEMIES:
            return target_index

    if raw_state is None or action.get("target") is None:
        return None

    enemies = raw_state.get("battle", {}).get("enemies", raw_state.get("enemies", []))
    return enemy_index_by_id(enemies, action.get("target"))


class BattleActionSpace(ActionSpace):
    """Enumerate legal battle actions (combat, hand_select, in-battle card_select)."""

    MAX_HAND = BATTLE_MAX_HAND
    MAX_POTIONS = BATTLE_MAX_POTIONS
    MAX_ENEMIES = BATTLE_MAX_ENEMIES

    def candidates(self, raw_state: dict) -> list[dict]:
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
        energy = parse_int(player.get("energy", raw_state.get("energy", 0)))

        candidates = [self._candidate({"type": "end_turn"}, "end_turn")]
        candidates.extend(self._play_card_candidates(raw_state, energy, enemies))
        candidates.extend(self._potion_candidates(potions, enemies))
        return candidates

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
                item = action_item(action, raw_state)
                if item is not None:
                    identity_key = Card.from_raw(item).identity_key
            identity_key = identity_key or "UNKNOWN_CARD"
            target_index = action_target_index(action, raw_state)
            if target_index is not None:
                return f"play_card:{identity_key}:target:{target_index}"
            return f"play_card:{identity_key}:self"

        if action_type == "use_potion":
            target_index = action_target_index(action, raw_state)
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
            if requires_enemy_target(card.raw):
                for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
                    target = enemy_id_by_index(enemies, enemy_index)
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
            slot_index = potion_slot(potion, slot)
            if not 0 <= slot_index < self.MAX_POTIONS:
                continue
            if potion.get("can_use_in_combat", True):
                action = {"type": "use_potion", "slot": slot_index}
                if requires_enemy_target(potion):
                    for enemy_index, enemy in enumerate(enemies[: self.MAX_ENEMIES]):
                        target = enemy_id_by_index(enemies, enemy_index)
                        if target is None:
                            continue
                        targeted_action = dict(action)
                        targeted_action["target_index"] = enemy_index
                        targeted_action["target"] = target
                        candidates.append(
                            self._candidate(
                                targeted_action,
                                f"use_potion:{slot_index}:target:{enemy_index}",
                            )
                        )
                else:
                    candidates.append(self._candidate(action, f"use_potion:{slot_index}:self"))
            # A held potion can always be discarded, even if it cannot be used in combat.
            candidates.append(
                self._candidate(
                    {"type": "discard_potion", "slot": slot_index},
                    f"discard_potion:{slot_index}",
                )
            )
        return candidates

    def _hand_select_candidates(self, raw_state: dict) -> list[dict]:
        hand_select = raw_state.get("hand_select", {})
        cards = hand_select.get("cards", [])
        selected_indices = {
            parse_int(card.get("index", -1), -1)
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
                card_index = parse_int(card.get("index", len(cards)))
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
                card_index = parse_int(card.get("index", fallback_index), fallback_index)
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

    def _best_card(self, cards: list[Card], energy: int) -> Card:
        return sorted(
            cards,
            key=lambda card: (
                card.cost_for_energy(energy),
                -card.current_upgrade_level,
                card.index if card.index is not None else 999,
            ),
        )[0]

    def fallback(self, raw_state: dict) -> dict:
        if raw_state.get("state_type") in {"monster", "elite", "boss"}:
            if not self._is_player_play_phase(raw_state):
                # Combat does not accept `proceed`, and `end_turn` is not ours to
                # send outside the play phase: wait for the server to settle.
                return dict(REFRESH_STATE_ACTION)
            return {"type": "end_turn"}

        if raw_state.get("state_type") == "hand_select":
            hand_select = raw_state.get("hand_select", {})
            selected_indices = {
                parse_int(card.get("index", -1), -1)
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
                    "card_index": parse_int(cards[0].get("index", 0)),
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
                    "index": parse_int(cards[0].get("index", 0)),
                }

            if card_select.get("can_cancel", False):
                return {"type": "cancel_selection"}

        # A selection prompt with nothing selectable and nothing confirmable has no
        # legal action; `end_turn` is not accepted there, so re-read state instead.
        if raw_state.get("state_type") in {"hand_select", "card_select"}:
            return dict(REFRESH_STATE_ACTION)

        return {"type": "end_turn"}

    def _is_player_play_phase(self, raw_state: dict) -> bool:
        battle = raw_state.get("battle", {})
        return battle.get("turn") == "player" and battle.get("is_play_phase") is True
