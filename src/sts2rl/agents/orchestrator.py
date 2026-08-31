"""Top-level policy router that delegates each screen to a specialized agent."""

import logging
from dataclasses import dataclass

from sts2rl.action_spaces.battle import BattleActionSpace
from sts2rl.action_spaces.event import EventActionSpace
from sts2rl.action_spaces.map import MapActionSpace
from sts2rl.action_spaces.rest import RestActionSpace
from sts2rl.action_spaces.reward import RewardActionSpace
from sts2rl.action_spaces.shop import ShopActionSpace
from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.agents.default.rule_based import DefaultPolicy
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.map.rule_based import MapPolicy
from sts2rl.agents.rest.rule_based import RestPolicy
from sts2rl.agents.reward.rule_based import RewardPolicy
from sts2rl.agents.shop.rule_based import ShopPolicy
from sts2rl.encoders.event_encoder import EventEncoder
from sts2rl.encoders.map_encoder import MapEncoder
from sts2rl.encoders.rest_encoder import RestEncoder
from sts2rl.encoders.reward_encoder import RewardEncoder
from sts2rl.encoders.shop_encoder import ShopEncoder

logger = logging.getLogger(__name__)


BATTLE_SCREEN_TYPES = {"monster", "elite", "boss"}
BATTLE_PROMPT_TYPES = {"hand_select"}
BATTLE_ACTION_TYPES = {
    "end_turn",
    "play_card",
    "use_potion",
    "discard_potion",
    "combat_select_card",
    "combat_confirm_selection",
    "select_card",
    "confirm_selection",
    "cancel_selection",
}
BATTLE_AGENT_TYPES = {
    "DQN": DQNCandidateAgent,
    "PPO": PPOCandidateAgent,
}

# Which policy module the battle agent scores candidates with. "flat" is the
# handcrafted featurizer + MLP (what every existing checkpoint holds); "learned"
# embeds CardModelEncoder in the policy so it trains with the network.
POLICY_VARIANTS = ("flat", "learned")
DEFAULT_POLICY_VARIANT = "flat"

# --- non-battle screen agents -------------------------------------------------
# Trainable candidate-action agents for the non-battle screens. A screen agent is
# a *configuration* of the shared DQN/PPO algorithms — its action space (legal
# moves) plus its encoder (features) — not a dedicated subclass. Agents are
# created via create_screen_agent(s)() and shared across clients (one
# model/optimizer each), mirroring the battle agent.


@dataclass(frozen=True)
class ScreenSpec:
    """The per-screen components bound into a candidate-action agent."""

    action_space: type
    encoder: type


SCREEN_SPECS = {
    "map": ScreenSpec(MapActionSpace, MapEncoder),
    "reward": ScreenSpec(RewardActionSpace, RewardEncoder),
    "shop": ScreenSpec(ShopActionSpace, ShopEncoder),
    "rest": ScreenSpec(RestActionSpace, RestEncoder),
    "event": ScreenSpec(EventActionSpace, EventEncoder),
}
SCREEN_NAMES = tuple(SCREEN_SPECS)
SCREEN_AGENT_TYPES = {"DQN", "PPO"}

# Map raw state_type -> screen name used for routing/training.
SCREEN_BY_STATE_TYPE = {
    "map": "map",
    "rewards": "reward",
    "card_reward": "reward",
    "treasure": "reward",
    "shop": "shop",
    "fake_merchant": "shop",
    "rest": "rest",
    "rest_site": "rest",
    "event": "event",
}


def normalize_battle_agent_type(agent_type: str) -> str:
    """Return a canonical battle agent type name."""
    agent_type = str(agent_type).strip().upper()
    if agent_type not in BATTLE_AGENT_TYPES:
        raise ValueError(f"Unsupported battle agent type: {agent_type!r}")
    return agent_type


def normalize_screen_agent_type(agent_type: str) -> str:
    """Return a canonical screen agent type name."""
    agent_type = str(agent_type).strip().upper()
    if agent_type not in SCREEN_AGENT_TYPES:
        raise ValueError(f"Unsupported screen agent type: {agent_type!r}")
    return agent_type


def normalize_policy_variant(policy: str) -> str:
    """Return a canonical battle-policy variant name."""
    policy = str(policy).strip().lower()
    if policy not in POLICY_VARIANTS:
        raise ValueError(f"Unsupported policy variant: {policy!r}")
    return policy


def create_battle_agent(agent_type: str, policy: str = DEFAULT_POLICY_VARIANT, **kwargs):
    """Create a battle agent implementation by type name and policy variant.

    ``policy="flat"`` is the handcrafted featurizer + MLP that every existing
    checkpoint was trained with. ``policy="learned"`` swaps in a policy module
    that owns a :class:`~sts2rl.models.card_encoder.CardModelEncoder`, so the
    per-card embedding trains end-to-end with the rest of the network (ADR-0007
    Phase 3). The two use different action schemas, so their checkpoints cannot
    be loaded into one another.
    """
    agent_class = BATTLE_AGENT_TYPES[normalize_battle_agent_type(agent_type)]
    if normalize_policy_variant(policy) == "flat":
        return agent_class(**kwargs)

    # Imported lazily: torch policy modules should not be pulled in for callers
    # that only touch the flat path.
    from sts2rl.encoders.learned_battle_encoder import LearnedBattleStateEncoder
    from sts2rl.models.learned_policies import (
        CardIndexLayout,
        LearnedCandidatePPOPolicy,
        LearnedCandidateQNetwork,
    )

    encoder = LearnedBattleStateEncoder()
    layout = CardIndexLayout.from_encoder(encoder)
    policy_class = (
        LearnedCandidateQNetwork
        if agent_class is DQNCandidateAgent
        else LearnedCandidatePPOPolicy
    )
    hidden_size = kwargs.pop("hidden_size", 256)
    return agent_class(
        encoder=encoder,
        action_space=BattleActionSpace(),
        policy=policy_class(layout, hidden_size=hidden_size),
        **kwargs,
    )


def create_screen_agent(screen: str, agent_type: str = "PPO", **kwargs):
    """Create the trainable agent for one non-battle screen by name and type."""
    agent_type = normalize_screen_agent_type(agent_type)
    if screen not in SCREEN_SPECS:
        raise ValueError(f"Unknown screen agent: {screen!r}")
    spec = SCREEN_SPECS[screen]
    return BATTLE_AGENT_TYPES[agent_type](
        encoder=spec.encoder(),
        action_space=spec.action_space(),
        **kwargs,
    )


def create_screen_agents(agent_type: str = "PPO") -> dict:
    """Create one trainable agent per registered non-battle screen."""
    return {screen: create_screen_agent(screen, agent_type) for screen in SCREEN_SPECS}


def is_battle_policy_state(raw_state: dict) -> bool:
    """Return whether a raw state should be controlled by the battle policy."""
    state_type = raw_state.get("state_type")
    if state_type in BATTLE_SCREEN_TYPES or state_type in BATTLE_PROMPT_TYPES:
        return True
    return state_type == "card_select" and raw_state.get("in_battle") is True


def screen_name_for_state(raw_state: dict) -> str | None:
    """Return the non-battle screen name controlling this state, if any."""
    state_type = raw_state.get("state_type")
    if state_type == "card_select" and not raw_state.get("in_battle"):
        return "event"
    return SCREEN_BY_STATE_TYPE.get(state_type)


class Agent:
    """Coordinate battle, map, reward, shop, rest, event, and fallback policies."""

    def __init__(
        self,
        battle_agent=None,
        battle_agent_type: str = "DQN",
        screen_agents: dict | None = None,
    ):
        self.battle_agent = battle_agent or create_battle_agent(battle_agent_type)
        # Each orchestrator drives one trajectory/client, so it gets its own
        # rollout collector over the shared battle model. This keeps concurrent
        # clients from clobbering each other's on-policy rollout state.
        self.rollout = self.battle_agent.new_rollout()
        # Optional trainable screen agents (shared models); each gets its own
        # per-client rollout collector here. Absent screens stay rule-based.
        self.screen_agents = screen_agents or {}
        self.screen_rollouts = {
            screen: agent.new_rollout() for screen, agent in self.screen_agents.items()
        }
        self.map_policy = MapPolicy()
        self.reward_policy = RewardPolicy()
        self.shop_policy = ShopPolicy()
        self.rest_policy = RestPolicy()
        self.event_policy = EventPolicy()
        self.default_policy = DefaultPolicy()

    def choose_action(self, state: dict, training: bool = True) -> dict:
        """Choose the next action for a normalized policy-state wrapper.

        ``training`` enables exploration (epsilon-greedy / sampling). Evaluation
        passes ``training=False`` so screen agents act greedily.
        """
        screen_type = state.get("screen_type")
        raw_state = state["raw_state"]

        logger.debug("Agent: choosing action screen_type=%s", screen_type)

        if is_battle_policy_state(raw_state):
            action = self.rollout.choose_action(raw_state, training=training)
            logger.debug(
                "Agent: selected %s action=%s",
                type(self.battle_agent).__name__,
                action,
            )
            return action

        # Trainable screen agent, when one is registered and has legal candidates.
        screen = screen_name_for_state(raw_state)
        if screen in self.screen_agents and self.screen_agents[screen].valid_action_candidates(
            raw_state
        ):
            action = self.screen_rollouts[screen].choose_action(raw_state, training=training)
            logger.debug("Agent: selected %s screen agent action=%s", screen, action)
            return action

        return self._rule_based_action(screen_type, raw_state)

    def _rule_based_action(self, screen_type: str | None, raw_state: dict) -> dict:
        """Fall back to the hand-written policy for a screen type."""
        if screen_type == "map":
            return self.map_policy.choose_action(raw_state)
        if screen_type in ["rewards", "card_reward", "treasure"]:
            return self.reward_policy.choose_action(raw_state)
        if screen_type == "shop":
            return self.shop_policy.choose_action(raw_state)
        if screen_type in ["rest", "rest_site"]:
            return self.rest_policy.choose_action(raw_state)
        if screen_type in ["event", "card_select"]:
            return self.event_policy.choose_action(raw_state)
        return self.default_policy.choose_action(raw_state)

    def train_from_step(
        self,
        prev_raw_state: dict,
        action: dict,
        reward: float,
        next_raw_state: dict,
        done: bool,
        reward_details: dict | None = None,
    ) -> dict | None:
        """Train the agent that controls the previous state (battle or screen)."""
        if is_battle_policy_state(prev_raw_state):
            return self._train_battle(
                prev_raw_state, action, reward, next_raw_state, done, reward_details
            )

        screen = screen_name_for_state(prev_raw_state)
        if screen in self.screen_agents:
            return self._train_screen(
                screen, prev_raw_state, action, reward, next_raw_state, done, reward_details
            )
        return None

    def _train_battle(
        self,
        prev_raw_state: dict,
        action: dict,
        reward: float,
        next_raw_state: dict,
        done: bool,
        reward_details: dict | None,
    ) -> dict | None:
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

    def _train_screen(
        self,
        screen: str,
        prev_raw_state: dict,
        action: dict,
        reward: float,
        next_raw_state: dict,
        done: bool,
        reward_details: dict | None,
    ) -> dict | None:
        """Train a screen agent, but only on transitions it actually chose.

        Fallback (rule-based) steps are skipped: they are exactly the steps where
        the screen had no legal candidates, so the agent never acted on-policy.
        """
        agent = self.screen_agents[screen]
        rollout = self.screen_rollouts[screen]

        candidates = agent.valid_action_candidates(prev_raw_state)
        if not candidates:
            return None
        try:
            key = agent.action_key(action, prev_raw_state)
        except (KeyError, ValueError):
            return None
        if key not in {candidate["action_key"] for candidate in candidates}:
            return None

        state = agent.encode_state(prev_raw_state)
        action_vector = agent.encode_action(prev_raw_state, action)
        next_state = agent.encode_state(next_raw_state)
        next_action_vectors = agent.candidate_action_vectors(next_raw_state)

        rollout.remember(
            state,
            action_vector,
            reward,
            next_state,
            bool(done),
            next_action_vectors,
        )
        loss = agent.train_step()
        logger.debug(
            "Agent: %s screen training step reward=%.2f loss=%s replay_size=%d",
            screen,
            reward,
            loss,
            agent.buffered_steps(),
        )
        return {
            "loss": loss,
            "updated": loss is not None,
            "reward": float(reward),
            "action_type": action.get("type"),
            "screen": screen,
            "epsilon": agent.epsilon,
            "replay_size": agent.buffered_steps(),
            "learn_steps": agent.learn_steps,
            "reward_details": reward_details or {},
        }
