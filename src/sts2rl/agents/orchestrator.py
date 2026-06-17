"""Top-level policy router that delegates each screen to a specialized agent."""

import logging

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.agents.map.rule_based import MapPolicy
from sts2rl.agents.reward.rule_based import RewardPolicy
from sts2rl.agents.shop.rule_based import ShopPolicy
from sts2rl.agents.rest.rule_based import RestPolicy
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.default.rule_based import DefaultPolicy
from sts2rl.agents.selection import (
    can_confirm_selection,
    can_select_more,
    first_unselected_card_index,
    selected_card_indices,
    selection_selected_count,
)


logger = logging.getLogger(__name__)


BATTLE_SCREEN_TYPES = {"monster", "elite", "boss"}
BATTLE_ACTION_TYPES = {
    "end_turn",
    "play_card",
    "use_potion",
    "combat_select_card",
    "combat_confirm_selection",
}


class Agent:
    """Coordinate battle, map, reward, shop, rest, event, and fallback policies."""

    def __init__(self):
        self.battle_agent = BattleDQNAgent()
        self.map_policy = MapPolicy()
        self.reward_policy = RewardPolicy()
        self.shop_policy = ShopPolicy()
        self.rest_policy = RestPolicy()
        self.event_policy = EventPolicy()
        self.default_policy = DefaultPolicy()

    def choose_action(self, state: dict) -> dict:
        """Choose the next action for a normalized policy-state wrapper."""
        screen_type = state.get("screen_type")
        raw_state = state["raw_state"]

        logger.debug("Agent: choosing action screen_type=%s", screen_type)

        forced_action = self._forced_transition_action(raw_state)
        if forced_action is not None:
            logger.debug("Agent: selected forced transition action=%s", forced_action)
            return forced_action

        if screen_type in BATTLE_SCREEN_TYPES:
            action = self.battle_agent.choose_action(raw_state, training=True)
            logger.debug("Agent: selected BattleDQNAgent action=%s", action)
            return action

        if screen_type == "map":
            action = self.map_policy.choose_action(raw_state)
            logger.debug("Agent: selected MapPolicy action=%s", action)
            return action

        if screen_type in ["rewards", "card_reward", "treasure"]:
            action = self.reward_policy.choose_action(raw_state)
            logger.debug("Agent: selected RewardPolicy action=%s", action)
            return action

        if screen_type == "shop":
            action = self.shop_policy.choose_action(raw_state)
            logger.debug("Agent: selected ShopPolicy action=%s", action)
            return action

        if screen_type in ["rest", "rest_site"]:
            action = self.rest_policy.choose_action(raw_state)
            logger.debug("Agent: selected RestPolicy action=%s", action)
            return action

        if screen_type in ["event", "card_select"]:
            action = self.event_policy.choose_action(raw_state)
            logger.debug("Agent: selected EventPolicy action=%s", action)
            return action

        action = self.default_policy.choose_action(raw_state)
        logger.debug("Agent: selected DefaultPolicy action=%s", action)
        return action

    def _forced_transition_action(self, raw_state: dict) -> dict | None:
        """Return required confirmation actions before normal policy selection."""
        state_type = raw_state.get("state_type")

        if state_type == "hand_select":
            hand_select = raw_state.get("hand_select", {})
            selected_indices = selected_card_indices(hand_select)
            selected_count = selection_selected_count(
                raw_state,
                "hand_select",
                len(selected_indices),
            )
            if can_confirm_selection(raw_state, "hand_select", selected_count):
                return {"type": "combat_confirm_selection"}
            if can_select_more(raw_state, "hand_select", selected_count):
                card_index = first_unselected_card_index(hand_select)
                if card_index is not None:
                    return {
                        "type": "combat_select_card",
                        "card_index": card_index,
                    }
            if hand_select.get("can_confirm", False):
                return {"type": "combat_confirm_selection"}

        if state_type == "card_select":
            card_select = raw_state.get("card_select", {})
            selected_count = selection_selected_count(
                raw_state,
                "card_select",
                len(card_select.get("selected_cards", [])),
            )
            if can_confirm_selection(raw_state, "card_select", selected_count):
                return {"type": "confirm_selection"}

        return None

    def train_from_step(
        self,
        prev_raw_state: dict,
        action: dict,
        reward: float,
        next_raw_state: dict,
        done: bool,
        reward_details: dict | None = None,
    ) -> dict | None:
        """Train the battle agent when a step belongs to the battle action space."""
        if prev_raw_state.get("state_type") not in BATTLE_SCREEN_TYPES:
            return None

        if action.get("type") not in BATTLE_ACTION_TYPES:
            return None

        state = self.battle_agent.encode_state(prev_raw_state)
        action_vector = self.battle_agent.encode_action(prev_raw_state, action)
        next_state = self.battle_agent.encode_state(next_raw_state)
        next_action_vectors = self.battle_agent.candidate_action_vectors(next_raw_state)

        reward_details = reward_details or {}
        battle_result = reward_details.get("result")
        battle_done = done or battle_result in {"won", "lost"}

        self.battle_agent.remember(
            state,
            action_vector,
            reward,
            next_state,
            battle_done,
            next_action_vectors,
        )
        loss = self.battle_agent.train_step()
        won_battle = battle_result == "won"
        lost_battle = battle_result == "lost"

        logger.debug(
            "Agent: battle training step reward=%.2f loss=%s replay_size=%d epsilon=%.3f",
            reward,
            loss,
            len(self.battle_agent.replay_buffer),
            self.battle_agent.epsilon,
        )
        return {
            "loss": loss,
            "updated": loss is not None,
            "reward": float(reward),
            "action_type": action.get("type"),
            "epsilon": self.battle_agent.epsilon,
            "replay_size": len(self.battle_agent.replay_buffer),
            "learn_steps": self.battle_agent.learn_steps,
            "won_battle": won_battle,
            "lost_battle": lost_battle,
            "reward_details": reward_details,
        }
