"""Helpers for battle-control flow shared by training and evaluation."""

import logging
from typing import Any

from sts2rl.env.rewards import RewardModel

COMBAT_SCREEN_TYPES = {"monster", "elite", "boss"}
MAX_FORCED_END_TURN_ADVANCES = 5
MAX_FORCED_HAND_SELECT_ADVANCES = 20


def should_skip_agent(raw_state: dict) -> bool:
    """Return whether combat is outside the player's playable phase."""
    if raw_state.get("state_type") not in COMBAT_SCREEN_TYPES:
        return False

    battle = raw_state.get("battle", {})
    return not (battle.get("turn") == "player" and battle.get("is_play_phase") is True)


def is_forced_end_turn_state(agent: Any, raw_state: dict) -> bool:
    """Return whether end_turn is the only legal battle action."""
    if raw_state.get("state_type") not in COMBAT_SCREEN_TYPES:
        return False
    if should_skip_agent(raw_state):
        return False

    valid_action_keys = [
        candidate["action_key"]
        for candidate in agent.battle_agent.valid_action_candidates(raw_state)
    ]
    return valid_action_keys == ["end_turn"]


def forced_end_turn_q_values(raw_state: dict) -> dict:
    """Build dashboard Q-value metadata for forced end-turn states."""
    state_type = raw_state.get("state_type")
    return {
        "available": False,
        "reason": "Only end_turn is legal; state is auto-advanced",
        "screen_type": state_type,
        "actions": [],
    }


def forced_end_turn_action_selection() -> dict:
    """Build dashboard action-selection metadata for forced end turns."""
    return {
        "method": "forced_end_turn",
        "reason": "Only end_turn is legal; state is auto-advanced",
        "q": None,
    }


def is_forced_hand_select_state(agent: Any, raw_state: dict) -> bool:
    """Return whether a hand-selection prompt can be auto-advanced."""
    if raw_state.get("state_type") != "hand_select":
        return False
    action = agent._forced_transition_action(raw_state)
    return action is not None and action.get("type") in {
        "combat_select_card",
        "combat_confirm_selection",
    }


def advance_forced_hand_select_states(
    game: Any,
    agent: Any,
    reward_model: RewardModel,
    raw_state: dict,
) -> tuple[dict, float, bool, list[dict]]:
    """Auto-advance hand-selection prompts that have deterministic count rules."""
    auto_steps = []
    total_reward = 0.0
    done = raw_state.get("state_type") == "game_over"

    while not done and is_forced_hand_select_state(agent, raw_state):
        if len(auto_steps) >= MAX_FORCED_HAND_SELECT_ADVANCES:
            logging.warning(
                "Stopped auto-advancing hand_select states after %d steps",
                len(auto_steps),
            )
            break

        prev_raw_state = raw_state
        action = agent._forced_transition_action(raw_state)
        raw_state, done, info = game.step(action)
        raw_state = info.get("raw_state", raw_state)
        if info.get("action_error"):
            reward, reward_details = reward_model.action_error_reward(
                info.get("error", "action dispatch failed")
            )
            auto_steps.append(
                {
                    "prev_raw_state": prev_raw_state,
                    "raw_state": raw_state,
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                    "reward_details": reward_details,
                }
            )
            total_reward += float(reward)
            break

        reward, reward_details = reward_model.compute(
            prev_raw_state,
            raw_state,
            action,
        )
        auto_steps.append(
            {
                "prev_raw_state": prev_raw_state,
                "raw_state": raw_state,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "reward_details": reward_details,
            }
        )
        total_reward += float(reward)

    return raw_state, total_reward, done, auto_steps


def advance_forced_end_turn_states(
    game: Any,
    agent: Any,
    reward_model: RewardModel,
    raw_state: dict,
) -> tuple[dict, float, bool, list[dict]]:
    """Auto-advance consecutive states where only end_turn is legal."""
    auto_steps = []
    total_reward = 0.0
    done = raw_state.get("state_type") == "game_over"

    while not done and is_forced_end_turn_state(agent, raw_state):
        if len(auto_steps) >= MAX_FORCED_END_TURN_ADVANCES:
            logging.warning(
                "Stopped auto-advancing forced end_turn states after %d steps",
                len(auto_steps),
            )
            break

        prev_raw_state = raw_state
        action = {"type": "end_turn"}
        raw_state, done, info = game.step(action)
        raw_state = info.get("raw_state", raw_state)
        if info.get("action_error"):
            reward, reward_details = reward_model.action_error_reward(
                info.get("error", "action dispatch failed")
            )
        else:
            reward, reward_details = reward_model.compute(
                prev_raw_state,
                raw_state,
                action,
            )
        auto_steps.append(
            {
                "prev_raw_state": prev_raw_state,
                "raw_state": raw_state,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "reward_details": reward_details,
            }
        )
        total_reward += float(reward)

    return raw_state, total_reward, done, auto_steps


def fold_reward_details(
    reward_details: dict,
    total_reward: float,
    auto_steps: list[dict],
) -> dict:
    """Fold automatically advanced reward details into the visible step reward."""
    if not auto_steps:
        return reward_details

    folded = dict(reward_details or {})
    folded["total"] = float(total_reward)
    folded["battle_reward"] = float(folded.get("battle_reward", 0.0)) + sum(
        float(step.get("reward_details", {}).get("battle_reward", 0.0)) for step in auto_steps
    )
    folded["run_reward"] = float(folded.get("run_reward", 0.0)) + sum(
        float(step.get("reward_details", {}).get("run_reward", 0.0)) for step in auto_steps
    )
    folded["auto_steps"] = len(auto_steps)
    folded["auto_end_turns"] = sum(
        1 for step in auto_steps if step.get("action", {}).get("type") == "end_turn"
    )
    folded["auto_hand_selects"] = sum(
        1
        for step in auto_steps
        if step.get("action", {}).get("type")
        in {
            "combat_select_card",
            "combat_confirm_selection",
        }
    )
    folded["auto_step_reward"] = sum(float(step.get("reward", 0.0)) for step in auto_steps)
    folded["auto_end_turn_reward"] = folded["auto_step_reward"]

    reward_component_keys = {
        "hp_penalty",
        "gold_penalty",
        "max_hp_penalty",
        "enemy_damage_reward",
        "enemy_kill_reward",
        "end_turn_energy_penalty",
        "potion_penalty",
        "win_reward",
        "loss_penalty",
    }
    latest_battle_details = None
    for step in auto_steps:
        step_details = step.get("reward_details", {})
        if step_details.get("type") != "battle":
            continue
        latest_battle_details = step_details.get("battle_details", step_details)
        for key in reward_component_keys:
            folded[key] = float(folded.get(key, 0.0)) + float(step_details.get(key, 0.0))

    if latest_battle_details is not None:
        folded["type"] = "battle"
        for key in (
            "prev_state_type",
            "next_state_type",
            "result",
            "next_hp",
            "next_gold",
            "next_max_hp",
            "hp_lost",
            "gold_lost",
            "max_hp_lost",
            "step_hp_lost",
            "enemies_killed",
            "enemy_hp_lost",
            "battle_start_hp",
            "battle_start_gold",
            "battle_start_max_hp",
        ):
            if key in latest_battle_details:
                folded[key] = latest_battle_details[key]
        if "battle_details" in folded:
            folded["battle_details"] = latest_battle_details

    return folded
