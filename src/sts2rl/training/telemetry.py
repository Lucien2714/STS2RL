"""Build dashboard telemetry for training-time action selection and Q values."""

from __future__ import annotations

import math

from sts2rl.agents.orchestrator import BATTLE_ACTION_TYPES, is_battle_policy_state


def action_selection_details(
    agent,
    raw_state: dict,
    action: dict,
    skipped_agent: bool,
    q_values: dict,
) -> dict:
    """Describe why the training loop selected a particular action."""
    state_type = raw_state.get("state_type")
    if skipped_agent:
        return {
            "method": "skip_agent",
            "reason": "combat screen is not in player play phase",
            "q": None,
        }

    forced_action = agent._forced_transition_action(raw_state)
    if forced_action == action:
        return {
            "method": "forced_transition",
            "reason": "screen requires confirmation/transition",
            "q": selected_action_q(q_values),
        }

    if is_battle_policy_state(raw_state) and action.get("type") in BATTLE_ACTION_TYPES:
        selection = dict(getattr(agent.battle_agent, "last_action_selection", {}) or {})
        selection.setdefault("method", "battle_unknown")
        selection.setdefault("reason", "battle action selection metadata unavailable")
        if selection.get("q") is None:
            selection["q"] = selected_action_q(q_values)
        return selection

    return {
        "method": f"policy:{state_type}",
        "reason": "non-battle policy action",
        "q": None,
    }


def current_q_values(agent, raw_state: dict, selected_action: dict | None = None) -> dict:
    """Return masked Q-values for the current battle state when available."""
    state_type = raw_state.get("state_type")
    if not is_battle_policy_state(raw_state):
        return {
            "available": False,
            "reason": f"No Q model is used for screen_type={state_type}",
            "screen_type": state_type,
            "actions": [],
        }

    return agent.battle_agent.current_q_values(raw_state, selected_action)


def selected_action_q(q_values: dict) -> float | None:
    """Extract the selected action's Q-value from dashboard action metadata."""
    for action in q_values.get("actions", []):
        if action.get("selected"):
            return action.get("q")
    return None


def safe_float(value: float) -> float | None:
    """Convert non-finite floats to None for JSON-safe telemetry."""
    value = float(value)
    if not math.isfinite(value):
        return None
    return value
