"""Dynamic structured legal-action candidates for a complete run."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.env.types import RawState


class NoLegalActionsError(RuntimeError):
    """Raised when the current state exposes no safe automatable action."""


class LegalActionProvider:
    """Enumerate complete, parameterized actions that are legal in a raw state."""

    COMBAT_TYPES = {"monster", "elite", "boss"}

    def candidates(self, state: RawState) -> tuple[GameAction, ...]:
        """Return all currently legal actions without assigning global IDs."""
        state_type = state.get("state_type")

        if state_type in self.COMBAT_TYPES:
            actions = self._combat_actions(state)
        elif state_type == "hand_select":
            actions = self._hand_select_actions(state)
        elif state_type == "rewards":
            actions = self._reward_actions(state)
        elif state_type == "card_reward":
            actions = self._card_reward_actions(state)
        elif state_type == "map":
            actions = self._indexed_actions(
                state, "map", "next_options", "choose_map_node"
            )
        elif state_type == "event":
            actions = self._event_actions(state)
        elif state_type == "rest_site":
            actions = self._rest_actions(state)
        elif state_type in {"shop", "fake_merchant"}:
            actions = self._shop_actions(state)
        elif state_type == "treasure":
            actions = self._treasure_actions(state)
        elif state_type == "card_select":
            actions = self._card_select_actions(state)
        elif state_type == "bundle_select":
            actions = self._bundle_select_actions(state)
        elif state_type == "relic_select":
            actions = self._relic_select_actions(state)
        elif state_type == "crystal_sphere":
            actions = self._crystal_sphere_actions(state)
        elif state_type in {"menu", "game_over", "unknown", "overlay"}:
            actions = []
        else:
            actions = []

        return tuple(self._deduplicate(actions))

    def require_candidates(self, state: RawState) -> tuple[GameAction, ...]:
        """Return candidates or explain why the state cannot be automated."""
        actions = self.candidates(state)
        if actions:
            return actions
        raise NoLegalActionsError(
            f"No safe legal actions for state_type={state.get('state_type')!r}: {state}"
        )

    def _combat_actions(self, state: RawState) -> list[GameAction]:
        battle = self._mapping(state.get("battle"))
        if battle.get("turn") != "player" or not battle.get("is_play_phase"):
            return []

        player = self._mapping(state.get("player"))
        enemies = [
            enemy
            for enemy in self._records(battle.get("enemies"))
            if self._number(enemy.get("hp")) > 0
        ]
        actions: list[GameAction] = []
        for card in self._records(player.get("hand")):
            if card.get("can_play") is not True:
                continue
            index = self._index(card)
            if index is None:
                continue
            actions.extend(
                self._targeted_actions("play_card", "card_index", index, card, enemies)
            )

        for potion in self._records(player.get("potions")):
            slot = self._integer(potion.get("slot"))
            if slot is None:
                continue
            if potion.get("can_use_in_combat") is True:
                actions.extend(
                    self._targeted_actions("use_potion", "slot", slot, potion, enemies)
                )
            actions.append(GameAction("discard_potion", slot=slot))

        actions.append(GameAction("end_turn"))
        return actions

    def _hand_select_actions(self, state: RawState) -> list[GameAction]:
        prompt = self._mapping(state.get("hand_select"))
        actions = [
            GameAction("combat_select_card", card_index=index)
            for index in self._indices(prompt.get("cards"))
        ]
        if prompt.get("can_confirm") is True:
            actions.append(GameAction("combat_confirm_selection"))
        return actions

    def _reward_actions(self, state: RawState) -> list[GameAction]:
        rewards = self._mapping(state.get("rewards"))
        actions = [
            GameAction("claim_reward", index=index)
            for item in self._records(rewards.get("items"))
            if item.get("type") != "potion" or not self._potion_belt_is_full(state)
            for index in [self._index(item)]
            if index is not None
        ]
        if rewards.get("can_proceed") is True:
            actions.append(GameAction("proceed"))
        actions.extend(self._discard_potion_actions(state))
        return actions

    def _card_reward_actions(self, state: RawState) -> list[GameAction]:
        reward = self._mapping(state.get("card_reward"))
        actions = [
            GameAction("select_card_reward", card_index=index)
            for index in self._indices(reward.get("cards"))
        ]
        if reward.get("can_skip") is True:
            actions.append(GameAction("skip_card_reward"))
        return actions

    def _event_actions(self, state: RawState) -> list[GameAction]:
        event = self._mapping(state.get("event"))
        if event.get("in_dialogue") is True:
            return [GameAction("advance_dialogue")]
        return [
            GameAction("choose_event_option", index=index)
            for option in self._records(event.get("options"))
            if option.get("is_locked") is not True
            for index in [self._index(option)]
            if index is not None
        ]

    def _rest_actions(self, state: RawState) -> list[GameAction]:
        rest_site = self._mapping(state.get("rest_site"))
        actions = [
            GameAction("choose_rest_option", index=index)
            for option in self._records(rest_site.get("options"))
            if option.get("is_enabled") is not False
            for index in [self._index(option)]
            if index is not None
        ]
        if rest_site.get("can_proceed") is True:
            actions.append(GameAction("proceed"))
        return actions

    def _shop_actions(self, state: RawState) -> list[GameAction]:
        if state.get("state_type") == "fake_merchant":
            merchant = self._mapping(state.get("fake_merchant"))
            shop = self._mapping(merchant.get("shop"))
        else:
            shop = self._mapping(state.get("shop"))
        actions = [
            GameAction("shop_purchase", index=index)
            for item in self._records(shop.get("items"))
            if item.get("is_stocked") is not False and item.get("can_afford") is True
            if item.get("category") != "potion" or not self._potion_belt_is_full(state)
            for index in [self._index(item)]
            if index is not None
        ]
        if shop.get("can_proceed") is True:
            actions.append(GameAction("proceed"))
        actions.extend(self._discard_potion_actions(state))
        return actions

    def _treasure_actions(self, state: RawState) -> list[GameAction]:
        treasure = self._mapping(state.get("treasure"))
        actions = [
            GameAction("claim_treasure_relic", index=index)
            for index in self._indices(treasure.get("relics"))
        ]
        if treasure.get("can_proceed") is True:
            actions.append(GameAction("proceed"))
        return actions

    def _card_select_actions(self, state: RawState) -> list[GameAction]:
        selection = self._mapping(state.get("card_select"))
        actions = [
            GameAction("select_card", index=index)
            for index in self._indices(selection.get("cards"))
        ]
        if selection.get("can_confirm") is True:
            actions.append(GameAction("confirm_selection"))
        if selection.get("can_cancel") is True or selection.get("can_skip") is True:
            actions.append(GameAction("cancel_selection"))
        return actions

    def _bundle_select_actions(self, state: RawState) -> list[GameAction]:
        selection = self._mapping(state.get("bundle_select"))
        actions = [
            GameAction("select_bundle", index=index)
            for index in self._indices(selection.get("bundles"))
        ]
        if selection.get("can_confirm") is True:
            actions.append(GameAction("confirm_bundle_selection"))
        if selection.get("can_cancel") is True:
            actions.append(GameAction("cancel_bundle_selection"))
        return actions

    def _relic_select_actions(self, state: RawState) -> list[GameAction]:
        selection = self._mapping(state.get("relic_select"))
        actions = [
            GameAction("select_relic", index=index)
            for index in self._indices(selection.get("relics"))
        ]
        if selection.get("can_skip") is True:
            actions.append(GameAction("skip_relic_selection"))
        return actions

    def _crystal_sphere_actions(self, state: RawState) -> list[GameAction]:
        sphere = self._mapping(state.get("crystal_sphere"))
        actions: list[GameAction] = []
        if sphere.get("can_use_big_tool") is True:
            actions.append(GameAction("crystal_sphere_set_tool", tool="big"))
        if sphere.get("can_use_small_tool") is True:
            actions.append(GameAction("crystal_sphere_set_tool", tool="small"))
        if sphere.get("tool") in {"big", "small"}:
            for cell in self._records(sphere.get("clickable_cells")):
                x = self._integer(cell.get("x"))
                y = self._integer(cell.get("y"))
                if x is not None and y is not None:
                    actions.append(GameAction("crystal_sphere_click_cell", x=x, y=y))
        if sphere.get("can_proceed") is True:
            actions.append(GameAction("crystal_sphere_proceed"))
        return actions

    def _indexed_actions(
        self,
        state: RawState,
        section_name: str,
        collection_name: str,
        action_type: str,
    ) -> list[GameAction]:
        section = self._mapping(state.get(section_name))
        return [
            GameAction(action_type, index=index)
            for index in self._indices(section.get(collection_name))
        ]

    def _targeted_actions(
        self,
        action_type: str,
        parameter_name: str,
        parameter_value: int,
        source: dict[str, Any],
        enemies: list[dict[str, Any]],
    ) -> list[GameAction]:
        base = {parameter_name: parameter_value}
        if self._requires_enemy_target(source.get("target_type")):
            return [
                GameAction(action_type, **base, target=str(target))
                for enemy in enemies
                for target in [enemy.get("entity_id")]
                if target is not None
            ]
        return [GameAction(action_type, **base)]

    def _discard_potion_actions(self, state: RawState) -> list[GameAction]:
        player = self._mapping(state.get("player"))
        return [
            GameAction("discard_potion", slot=slot)
            for potion in self._records(player.get("potions"))
            for slot in [self._integer(potion.get("slot"))]
            if slot is not None
        ]

    def _potion_belt_is_full(self, state: RawState) -> bool:
        player = self._mapping(state.get("player"))
        capacity = self._integer(player.get("max_potion_slots"))
        if capacity is None:
            return False
        return len(self._records(player.get("potions"))) >= capacity

    @staticmethod
    def _requires_enemy_target(target_type: object) -> bool:
        return isinstance(target_type, str) and target_type.casefold() == "anyenemy"

    @staticmethod
    def _mapping(value: object) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _records(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @classmethod
    def _indices(cls, value: object) -> list[int]:
        return [
            index
            for item in cls._records(value)
            for index in [cls._index(item)]
            if index is not None
        ]

    @classmethod
    def _index(cls, value: dict[str, Any]) -> int | None:
        return cls._integer(value.get("index"))

    @staticmethod
    def _integer(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _number(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _deduplicate(actions: Iterable[GameAction]) -> Iterable[GameAction]:
        seen: set[str] = set()
        for action in actions:
            key = repr(sorted(action.to_dict().items()))
            if key not in seen:
                seen.add(key)
                yield action
