"""Top-level policy router that delegates each screen to a specialized agent."""

import logging

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.agents.battle.ppo_agent import BattlePPOAgent
from sts2rl.agents.map.rule_based import MapPolicy
from sts2rl.agents.reward.rule_based import RewardPolicy
from sts2rl.agents.shop.rule_based import ShopPolicy
from sts2rl.agents.rest.rule_based import RestPolicy
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.default.rule_based import DefaultPolicy


logger = logging.getLogger(__name__)


BATTLE_SCREEN_TYPES = {"monster", "elite", "boss"}
BATTLE_PROMPT_TYPES = {"hand_select"}
BATTLE_ACTION_TYPES = {
    "end_turn",
    "play_card",
    "use_potion",
    "combat_select_card",
    "combat_confirm_selection",
    "select_card",
    "confirm_selection",
    "cancel_selection",
}
BATTLE_AGENT_TYPES = {
    "DQN": BattleDQNAgent,
    "PPO": BattlePPOAgent,
}


def normalize_battle_agent_type(agent_type: str) -> str:
    """Return a canonical battle agent type name."""
    agent_type = str(agent_type).strip().upper()
    if agent_type not in BATTLE_AGENT_TYPES:
        raise ValueError(f"Unsupported battle agent type: {agent_type!r}")
    return agent_type


def create_battle_agent(agent_type: str):
    """Create a battle agent implementation by type name."""
    return BATTLE_AGENT_TYPES[normalize_battle_agent_type(agent_type)]()


def is_battle_policy_state(raw_state: dict) -> bool:
    """Return whether a raw state should be controlled by the battle policy."""
    state_type = raw_state.get("state_type")
    if state_type in BATTLE_SCREEN_TYPES or state_type in BATTLE_PROMPT_TYPES:
        return True
    return state_type == "card_select" and raw_state.get("in_battle") is True


class Agent:
    """Coordinate battle, map, reward, shop, rest, event, and fallback policies."""

    def __init__(self, battle_agent=None, battle_agent_type: str = "DQN"):
        self.battle_agent = battle_agent or create_battle_agent(battle_agent_type)
        # Each orchestrator drives one trajectory/client, so it gets its own
        # rollout collector over the shared battle model. This keeps concurrent
        # clients from clobbering each other's on-policy rollout state.
        self.rollout = self.battle_agent.new_rollout()
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

        if is_battle_policy_state(raw_state):
            action = self.rollout.choose_action(raw_state, training=True)
            logger.debug(
                "Agent: selected %s action=%s",
                type(self.battle_agent).__name__,
                action,
            )
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
        if not is_battle_policy_state(prev_raw_state):
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

        self.rollout.remember(
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
            self.battle_agent.buffered_steps(),
            self.battle_agent.epsilon,
        )
        return {
            "loss": loss,
            "updated": loss is not None,
            "reward": float(reward),
            "action_type": action.get("type"),
            "epsilon": self.battle_agent.epsilon,
            "replay_size": self.battle_agent.buffered_steps(),
            "learn_steps": self.battle_agent.learn_steps,
            "won_battle": won_battle,
            "lost_battle": lost_battle,
            "reward_details": reward_details,
        }
