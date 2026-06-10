"""Build dashboard telemetry for training-time action selection and Q values."""

from __future__ import annotations

import math

from sts2rl.agents.orchestrator import BATTLE_ACTION_TYPES, BATTLE_SCREEN_TYPES


try:
    import torch
except ImportError:
    torch = None


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

    if state_type in BATTLE_SCREEN_TYPES and action.get("type") in BATTLE_ACTION_TYPES:
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
    if state_type not in BATTLE_SCREEN_TYPES:
        return {
            "available": False,
            "reason": f"No Q model is used for screen_type={state_type}",
            "screen_type": state_type,
            "actions": [],
        }

    battle_agent = agent.battle_agent
    if torch is None or battle_agent.model is None:
        return {
            "available": False,
            "reason": "Torch/model is not available",
            "screen_type": state_type,
            "actions": [],
        }

    action_mask = battle_agent.valid_action_mask(raw_state)
    state_vector = battle_agent.encode_state(raw_state, action_mask)
    with torch.no_grad():
        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32,
            device=battle_agent.device,
        ).unsqueeze(0)
        q_tensor = battle_agent.model(state_tensor).squeeze(0).detach().cpu()

    selected_action_id = None
    if selected_action is not None:
        try:
            selected_action_id = battle_agent.get_game_action_id(selected_action, raw_state)
        except (KeyError, ValueError):
            selected_action_id = None

    actions = []
    best_valid = None
    for action_id, q_value in enumerate(q_tensor.tolist()):
        valid = bool(action_mask[action_id])
        q_value = safe_float(q_value)
        masked_q = q_value if valid else None
        action = {
            "id": action_id,
            "key": battle_agent.get_action_key(action_id),
            "q": q_value,
            "masked_q": masked_q,
            "valid": valid,
            "selected": action_id == selected_action_id,
        }
        actions.append(action)
        if valid and q_value is not None and (best_valid is None or q_value > best_valid["q"]):
            best_valid = action

    return {
        "available": True,
        "screen_type": state_type,
        "selected_action_id": selected_action_id,
        "selected_q": selected_action_q({"actions": actions}),
        "best_valid_action": best_valid,
        "actions": actions,
    }


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
