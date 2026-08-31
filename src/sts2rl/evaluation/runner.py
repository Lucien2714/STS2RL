"""Checkpoint evaluation loop and result aggregation utilities."""

import csv
import logging
import time
from pathlib import Path
from typing import Callable

from sts2rl.agents.orchestrator import (
    Agent,
    create_screen_agents,
    is_battle_policy_state,
    normalize_battle_agent_type,
    normalize_screen_agent_type,
)
from sts2rl.checkpoints.manager import agent_latest_path
from sts2rl.env.game_env import Game
from sts2rl.env.player import Player
from sts2rl.env.rewards import ScopedRewardModel
from sts2rl.evaluation.dashboard import LiveEvaluationDashboard
from sts2rl.flow.battle_flow import forced_end_turn_q_values
from sts2rl.flow.player_detail import refresh_player_detail_for_map
from sts2rl.flow.step_loop import apply_action, decide_action


def load_eval_screen_agents(screen_agent_type: str | None) -> dict:
    """Build screen agents and load each one's latest checkpoint for evaluation.

    Screens with no checkpoint (or an incompatible one) are dropped so the
    orchestrator falls back to their rule-based policy. Returns ``{}`` when no
    screen-agent type is requested.
    """
    if screen_agent_type is None:
        return {}

    screen_agent_type = normalize_screen_agent_type(screen_agent_type)
    loaded = {}
    for screen, agent in create_screen_agents(screen_agent_type).items():
        path = agent_latest_path(screen, screen_agent_type)
        if not path.exists():
            logging.info(
                "No %s %s screen checkpoint at %s; using rule-based fallback",
                screen_agent_type,
                screen,
                path,
            )
            continue
        try:
            agent.load(str(path))
            agent.model.eval()
        except Exception as exc:
            logging.warning(
                "Could not load %s %s screen agent from %s; using rule-based fallback. error=%s",
                screen_agent_type,
                screen,
                path,
                exc,
            )
            continue
        loaded[screen] = agent
        logging.info("Loaded %s %s screen agent from %s", screen_agent_type, screen, path)
    return loaded


def current_q_values(agent: Agent, raw_state: dict, selected_action: dict | None = None) -> dict:
    """Return masked Q-values for a battle state when a model is available."""
    state_type = raw_state.get("state_type")
    if not is_battle_policy_state(raw_state):
        return {
            "available": False,
            "reason": f"No Q model is used for screen_type={state_type}",
            "screen_type": state_type,
            "actions": [],
        }

    return agent.battle_agent.current_q_values(raw_state, selected_action)


def evaluate_episode(
    game: Game,
    agent: Agent,
    max_steps: int,
    sleep_seconds: float,
    checkpoint_name: str,
    client_id: str,
    base_url: str,
    episode_index: int,
    run_seed: str | None,
    reset_environment: bool,
    dashboard: LiveEvaluationDashboard | None = None,
) -> dict:
    """Run one evaluation episode and return aggregate episode metrics."""
    reward_model = ScopedRewardModel()
    player = Player(character=game.character)
    if dashboard is not None:
        dashboard.wait_if_paused()

    if reset_environment:
        raw_state = game.reset(run_seed=run_seed)
    else:
        raw_state = game.get_state()
    reward_model.reset(raw_state)

    episode_reward = 0.0
    battle_reward = 0.0
    run_reward = 0.0
    steps = 0
    battle_steps = 0
    battle_wins = 0
    battle_losses = 0
    has_step_limit = max_steps > 0

    while raw_state.get("state_type") != "game_over" and (not has_step_limit or steps < max_steps):
        if dashboard is not None:
            dashboard.wait_if_paused()

        refresh_player_detail_for_map(game, player, raw_state)

        decision = decide_action(agent, raw_state, training=False)
        action = decision.action
        if decision.forced_end_turn:
            q_values = forced_end_turn_q_values(raw_state)
        else:
            q_values = current_q_values(agent, raw_state, action)

        outcome = apply_action(game, agent, reward_model, raw_state, action)
        next_raw_state = outcome.next_raw_state
        reward = outcome.reward
        done = outcome.done
        reward_details = outcome.reward_details

        episode_reward += reward
        run_reward += reward_details.get("run_reward", 0.0)
        folded_step_count = outcome.step_count
        steps += folded_step_count

        if reward_details.get("type") == "battle":
            battle_reward += reward_details.get("battle_reward", reward)
            battle_steps += folded_step_count
            if reward_details.get("result") == "won":
                battle_wins += 1
            if reward_details.get("result") == "lost":
                battle_losses += 1

        if dashboard is not None:
            dashboard.update_step(
                checkpoint_name,
                episode_index,
                client_id,
                base_url,
                run_seed,
                steps,
                raw_state,
                next_raw_state,
                action,
                reward,
                episode_reward,
                battle_reward,
                battle_wins,
                battle_losses,
                done,
                q_values,
            )
        print(
            f"{client_id} {checkpoint_name} episode={episode_index} step={steps} "
            f"seed={run_seed or 'normal'} "
            f"{raw_state.get('state_type')} -> {next_raw_state.get('state_type')} "
            f"action={action} reward={reward:.2f} done={done}"
        )
        raw_state = next_raw_state
        if done:
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    run = raw_state.get("run", {})
    return {
        "reward": episode_reward,
        "battle_reward": battle_reward,
        "run_reward": run_reward,
        "steps": steps,
        "battle_steps": battle_steps,
        "battle_wins": battle_wins,
        "battle_losses": battle_losses,
        "floor": run.get("floor", 0),
        "act": run.get("act", 0),
        "seed": run_seed or "normal",
        "timed_out": has_step_limit and raw_state.get("state_type") != "game_over",
    }


def summarize_episode_results(results: list[dict]) -> dict:
    """Aggregate per-episode metrics into one checkpoint summary."""
    episode_count = max(1, len(results))
    wins = sum(result["battle_wins"] for result in results)
    losses = sum(result["battle_losses"] for result in results)
    return {
        "episodes": len(results),
        "avg_reward": sum(result["reward"] for result in results) / episode_count,
        "avg_battle_reward": sum(result["battle_reward"] for result in results) / episode_count,
        "avg_run_reward": sum(result["run_reward"] for result in results) / episode_count,
        "avg_steps": sum(result["steps"] for result in results) / episode_count,
        "avg_battle_steps": sum(result["battle_steps"] for result in results) / episode_count,
        "battle_wins": wins,
        "battle_losses": losses,
        "battle_win_rate": wins / max(1, wins + losses),
        "avg_floor": sum(result["floor"] for result in results) / episode_count,
        "max_floor": max((result["floor"] for result in results), default=0),
        "timeouts": sum(1 for result in results if result["timed_out"]),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    character: int,
    base_url: str,
    timeout: float,
    game_mode: str,
    episode_seeds: list[str],
    max_steps: int,
    sleep_seconds: float,
    reset_environment: bool,
    dashboard: LiveEvaluationDashboard | None = None,
    client_id: str = "client-1",
    pause_between_episodes: bool = True,
    after_episode: Callable[[str, str, str, int], None] | None = None,
    battle_agent_type: str = "DQN",
    screen_agent_type: str | None = None,
) -> dict:
    """Load a checkpoint, run evaluation episodes, and summarize results."""
    if dashboard is not None:
        dashboard.wait_if_paused()

    battle_agent_type = normalize_battle_agent_type(battle_agent_type)
    screen_agents = load_eval_screen_agents(screen_agent_type)
    agent = Agent(battle_agent_type=battle_agent_type, screen_agents=screen_agents)
    agent.battle_agent.load(str(checkpoint_path))
    agent.battle_agent.model.eval()

    game = Game(
        character=character,
        base_url=base_url,
        timeout=timeout,
        game_mode=game_mode,
        start_run_option="embark",
    )
    episode_results = []
    for episode_index in range(1, episodes + 1):
        run_seed = (
            episode_seeds[episode_index - 1]
            if reset_environment and game_mode in {"custom", "daily"}
            else None
        )
        episode_result = evaluate_episode(
            game,
            agent,
            max_steps,
            sleep_seconds,
            checkpoint_path.name,
            client_id,
            base_url,
            episode_index,
            run_seed,
            reset_environment,
            dashboard,
        )
        episode_results.append(episode_result)
        if after_episode is not None:
            after_episode(client_id, base_url, checkpoint_path.name, episode_index)
        if dashboard is not None and pause_between_episodes:
            dashboard.pause_between_episodes(checkpoint_path.name, episode_index)
    summary = summarize_episode_results(episode_results)
    summary.update(
        {
            "checkpoint": checkpoint_path.name,
            "client_id": client_id,
            "base_url": base_url,
            "path": str(checkpoint_path),
            "trained_steps": agent.battle_agent.trained_steps,
            "learn_steps": agent.battle_agent.learn_steps,
            "epsilon": agent.battle_agent.epsilon,
            "battle_agent": battle_agent_type,
            "game_mode": game_mode,
            "start_mode": "reset" if reset_environment else "current_state",
            "seeds": ",".join(result["seed"] for result in episode_results),
            "ending_steps": ",".join(str(result["steps"]) for result in episode_results),
        }
    )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write evaluation summaries to a CSV file."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
