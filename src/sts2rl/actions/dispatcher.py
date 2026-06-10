"""Map normalized game-action dictionaries onto STS2MCP client calls."""

from __future__ import annotations

from sts2rl.actions.game_action import GameAction


class ActionDispatcher:
    """Dispatches game-action dictionaries to an STS2MCP client."""

    def __init__(self, client) -> None:
        self.client = client

    def dispatch(self, action: GameAction):
        """Execute an action dictionary against the wrapped game client."""
        action_type = action.get("type")
        if action_type == "end_turn":
            return self.client.end_turn()
        if action_type == "proceed":
            return self.client.proceed()
        if action_type == "play_card":
            return self.client.play_card(action["card_index"], target=action.get("target"))
        if action_type == "use_potion":
            return self.client.use_potion(action["slot"], target=action.get("target"))
        if action_type == "combat_select_card":
            return self.client.combat_select_card(action["card_index"])
        if action_type == "combat_confirm_selection":
            return self.client.combat_confirm_selection()
        if action_type == "choose_map_node":
            return self.client.choose_map_node(action["index"])
        if action_type == "claim_reward":
            return self.client.claim_reward(action["index"])
        if action_type == "select_card_reward":
            return self.client.select_card_reward(action["card_index"])
        if action_type == "skip_card_reward":
            return self.client.skip_card_reward()
        if action_type == "claim_treasure_relic":
            return self.client.claim_treasure_relic(action["index"])
        if action_type == "choose_rest_option":
            return self.client.choose_rest_option(action["index"])
        if action_type == "choose_event_option":
            return self.client.choose_event_option(action["index"])
        if action_type == "advance_dialogue":
            return self.client.advance_dialogue()
        if action_type == "select_card":
            return self.client.select_card(action["index"])
        if action_type == "confirm_selection":
            return self.client.confirm_selection()
        if action_type == "cancel_selection":
            return self.client.cancel_selection()
        if action_type == "shop_purchase":
            return self.client.shop_purchase(action["index"])
        raise ValueError(f"Unknown action type: {action_type}")
