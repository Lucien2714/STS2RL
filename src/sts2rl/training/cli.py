"""Command-line training loop for one or more STS2MCP clients."""

import argparse
import json
import logging
from pathlib import Path
import threading
import time
from sts2rl.checkpoints.manager import (
    battle_agent_checkpoint_dir,
    battle_backup_path,
    battle_latest_path,
    normalize_battle_agent_type,
    TRAINING_BACKUP_INTERVAL,
)
from sts2rl.env.game_env import Game
from sts2rl.env.player import Player
from sts2rl.env.rewards import BattleProgressReward
from sts2rl.flow.player_detail import refresh_player_detail_for_map
from sts2rl.agents.orchestrator import Agent
from sts2rl.flow.battle_flow import (
    advance_forced_end_turn_states,
    fold_reward_details,
    forced_end_turn_action_selection,
    forced_end_turn_q_values,
    is_forced_end_turn_state,
    should_skip_agent,
)
from sts2rl.training.config import (
    DEFAULT_TRAINING_LIVE_HTTP_HOST,
    DEFAULT_TRAINING_LIVE_HTTP_PORT,
    DEFAULT_TRAINING_LIVE_WS_HOST,
    DEFAULT_TRAINING_LIVE_WS_PORT,
)
from sts2rl.training.dashboard import LiveTrainingDashboard
from sts2rl.training.episode_log import EpisodeLogWriter
from sts2rl.training.telemetry import action_selection_details, current_q_values
from sts2rl.training.tensorboard import TensorBoardLogger

SAVE_INTERVAL = 10
DEFAULT_EPISODE_LOG_PATH = Path("logs") / "training_episodes.jsonl"
DEFAULT_CLIENT_PORT = 15526
RECONNECT_POLL_SECONDS = 2.0


def save_battle_checkpoint(agent: Agent, path: Path, reason: str) -> None:
    """Save the battle agent and log the checkpoint reason."""
    agent.battle_agent.save(str(path))
    logging.info(
        "Saved battle agent checkpoint (%s) to %s trained_steps=%d learn_steps=%d",
        reason,
        path,
        agent.battle_agent.trained_steps,
        agent.battle_agent.learn_steps,
    )


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train an STS2RL battle agent.")
    parser.add_argument(
        "--battle-agent",
        choices=["DQN", "PPO"],
        default="DQN",
        help="Battle agent implementation to train.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Serve a live training dashboard over HTTP.",
    )
    parser.add_argument(
        "--live-html",
        type=Path,
        default=None,
        help="Optionally write a static copy of the live training dashboard HTML to this file.",
    )
    parser.add_argument(
        "--live-http-host",
        default=DEFAULT_TRAINING_LIVE_HTTP_HOST,
        help="Host interface for the live training dashboard HTTP server.",
    )
    parser.add_argument(
        "--live-http-port",
        type=int,
        default=DEFAULT_TRAINING_LIVE_HTTP_PORT,
        help="Port for the live training dashboard HTTP server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--live-ws-host",
        default=DEFAULT_TRAINING_LIVE_WS_HOST,
        help="Host interface for the live training dashboard WebSocket server.",
    )
    parser.add_argument(
        "--live-ws-port",
        type=int,
        default=DEFAULT_TRAINING_LIVE_WS_PORT,
        help="Port for the live training dashboard WebSocket server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--client-host",
        default="localhost",
        help="Host used when building STS2 MCP API URLs from --client-port.",
    )
    parser.add_argument(
        "--client-port",
        "--mcp-port",
        action="append",
        dest="client_ports",
        type=int,
        default=None,
        help=(
            "STS2 MCP API port for a game client. Specify this once per client, "
            "for example --client-port 15526 --client-port 15527."
        ),
    )
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        default=None,
        help=(
            "Full STS2 MCP API base URL for one client. Can be specified multiple "
            "times; defaults to http://localhost:15526/api/v1 when no clients are set."
        ),
    )
    parser.add_argument(
        "--episode-log",
        type=Path,
        default=DEFAULT_EPISODE_LOG_PATH,
        help="JSONL file that receives one summary record per completed episode.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory for training metrics.",
    )
    return parser.parse_args()


def base_url_from_port(host: str, port: int) -> str:
    """Build an STS2MCP API base URL from host and port."""
    return f"http://{host}:{port}/api/v1"


def training_client_urls(args: argparse.Namespace) -> list[str]:
    """Resolve CLI client-port/base-url arguments into API base URLs."""
    urls = list(args.base_urls or [])
    urls.extend(base_url_from_port(args.client_host, port) for port in args.client_ports or [])
    if not urls:
        urls.append(base_url_from_port(args.client_host, DEFAULT_CLIENT_PORT))
    return urls


class SharedTrainingState:
    """Synchronize one trainable agent across multiple training clients."""

    def __init__(self, agent: Agent, battle_agent_type: str) -> None:
        self.agent = agent
        self.battle_agent_type = normalize_battle_agent_type(battle_agent_type)
        self.checkpoint_dir = battle_agent_checkpoint_dir(self.battle_agent_type)
        self.latest_path = battle_latest_path(self.battle_agent_type)
        self.agent_lock = threading.RLock()
        self.last_backup_index = agent.battle_agent.trained_steps // TRAINING_BACKUP_INTERVAL

    def save_milestone_if_needed(self) -> None:
        """Save periodic latest and milestone checkpoints after enough steps."""
        backup_index = self.agent.battle_agent.trained_steps // TRAINING_BACKUP_INTERVAL
        if backup_index <= self.last_backup_index:
            return

        self.last_backup_index = backup_index
        backup_step = backup_index * TRAINING_BACKUP_INTERVAL
        save_battle_checkpoint(
            self.agent,
            self.latest_path,
            f"training milestone {backup_step}",
        )
        save_battle_checkpoint(
            self.agent,
            battle_backup_path(backup_step, self.battle_agent_type),
            f"training milestone backup {backup_step}",
        )


def state_signature(raw_state: dict | None) -> str:
    """Return a stable string signature for comparing raw states."""
    return json_dumps_stable(raw_state or {})


def json_dumps_stable(value: object) -> str:
    """Serialize a value with stable key ordering for state comparisons."""
    return json.dumps(value, sort_keys=True, default=str)


def is_connection_error_text(error: object) -> bool:
    """Return whether an error string looks like a transient client disconnect."""
    text = str(error)
    return (
        "Request failed:" in text
        or "Connection refused" in text
        or "Connection aborted" in text
        or "Read timed out" in text
        or "Max retries exceeded" in text
        or "Failed to establish a new connection" in text
    )


def is_connection_action_error(info: dict) -> bool:
    """Return whether step info represents an action-time connection failure."""
    return bool(
        info.get("action_error")
        and is_connection_error_text(info.get("error", ""))
    )


def is_main_menu(raw_state: dict) -> bool:
    """Return whether the backend is at the main menu."""
    return raw_state.get("state_type") == "menu" and raw_state.get("menu_screen") == "main"


def has_continue_option(raw_state: dict) -> bool:
    """Return whether the main menu can continue the previous run."""
    return "continue" in raw_state.get("options", [])


def update_waiting_client_status(
    dashboard: LiveTrainingDashboard | None,
    client_id: str,
    base_url: str,
    episode: int,
    reason: object,
) -> None:
    """Publish a reconnect-waiting status to the dashboard."""
    if dashboard is None:
        return
    dashboard.update_client_status(
        client_id,
        base_url,
        f"Waiting for reconnect: {reason}",
        episode,
    )


def wait_for_responsive_state(
    game: Game,
    client_id: str,
    base_url: str,
    dashboard: LiveTrainingDashboard | None,
    stop_event: threading.Event,
    episode: int,
    reason: object,
) -> dict | None:
    """Poll a client until it returns state or the stop event is set."""
    update_waiting_client_status(dashboard, client_id, base_url, episode, reason)
    warned_at = 0.0
    while not stop_event.is_set():
        try:
            raw_state = game.get_state()
            if dashboard is not None:
                dashboard.update_client_status(client_id, base_url, "Reconnected", episode)
            logging.info("%s reconnected to %s", client_id, base_url)
            return raw_state
        except Exception as exc:
            now = time.time()
            if now - warned_at >= 10.0:
                warned_at = now
                logging.warning("%s waiting for %s to reconnect: %s", client_id, base_url, exc)
            update_waiting_client_status(dashboard, client_id, base_url, episode, exc)
            time.sleep(RECONNECT_POLL_SECONDS)
    return None


def recover_client_state_after_disconnect(
    game: Game,
    client_id: str,
    base_url: str,
    dashboard: LiveTrainingDashboard | None,
    stop_event: threading.Event,
    episode: int,
    previous_state: dict | None,
    reason: object,
) -> tuple[dict | None, bool]:
    """Recover a client after disconnect and decide whether to restart episode."""
    previous_signature = state_signature(previous_state)
    next_reason = reason

    while not stop_event.is_set():
        raw_state = wait_for_responsive_state(
            game,
            client_id,
            base_url,
            dashboard,
            stop_event,
            episode,
            next_reason,
        )
        if raw_state is None:
            return None, False

        if previous_state is not None and state_signature(raw_state) == previous_signature:
            if dashboard is not None:
                dashboard.update_client_status(client_id, base_url, "Training", episode)
            return raw_state, False

        if is_main_menu(raw_state) and has_continue_option(raw_state):
            try:
                continued_state = game._menu_select_state("continue")
                if dashboard is not None:
                    dashboard.update_client_status(client_id, base_url, "Training", episode)
                logging.info("%s resumed episode %d with main-menu continue", client_id, episode)
                return continued_state, False
            except Exception as exc:
                next_reason = exc
                update_waiting_client_status(dashboard, client_id, base_url, episode, exc)
                time.sleep(RECONNECT_POLL_SECONDS)
                continue

        if is_main_menu(raw_state):
            logging.info(
                "%s returned to main menu without continue; current episode will restart",
                client_id,
            )
            return raw_state, True

        if dashboard is not None:
            dashboard.update_client_status(client_id, base_url, "Training", episode)
        return raw_state, False

    return None, False


def start_or_recover_episode(
    game: Game,
    client_id: str,
    base_url: str,
    dashboard: LiveTrainingDashboard | None,
    stop_event: threading.Event,
    episode: int,
) -> dict | None:
    """Reset a game or recover an existing state after connection failures."""
    while not stop_event.is_set():
        try:
            return game.reset()
        except Exception as exc:
            raw_state, restart_episode = recover_client_state_after_disconnect(
                game,
                client_id,
                base_url,
                dashboard,
                stop_event,
                episode,
                None,
                exc,
            )
            if raw_state is None:
                return None
            if restart_episode:
                continue
            return raw_state
    return None


def run_training_client(
    client_id: str,
    base_url: str,
    shared: SharedTrainingState,
    dashboard: LiveTrainingDashboard | None,
    episode_log: EpisodeLogWriter,
    tensorboard: TensorBoardLogger,
    stop_event: threading.Event,
) -> None:
    """Run the training loop for one STS2MCP client."""
    game = Game(character=0, base_url=base_url)
    player = Player(character=game.character)
    reward_model = BattleProgressReward()
    agent = shared.agent
    episode = 1
    final_client_status = "Stopped"

    try:
      while not stop_event.is_set():
        if dashboard is not None:
            dashboard.update_client_status(client_id, base_url, "Resetting", episode)
            dashboard.wait_if_paused()
        raw_state = start_or_recover_episode(
            game,
            client_id,
            base_url,
            dashboard,
            stop_event,
            episode,
        )
        if raw_state is None:
            break
        reward_model.reset(raw_state)
        final_raw_state = raw_state
        restart_episode = False
        done = False
        episode_reward = 0.0
        battle_reward = 0.0
        current_battle_reward = 0.0
        current_battle_steps = 0
        current_battle_hp_lost = 0
        current_battle_potions_used = 0
        battle_number = 0
        episode_steps = 0
        battle_steps = 0
        update_count = 0
        battle_wins = 0
        battle_losses = 0
        losses = []
        action_counts = {}
        with shared.agent_lock:
            epsilon_start = agent.battle_agent.epsilon
            replay_start = len(agent.battle_agent.replay_buffer)

        logging.info("%s starting episode %d base_url=%s", client_id, episode, base_url)

        while raw_state.get("state_type")!="game_over" and not stop_event.is_set():
            if dashboard is not None:
                dashboard.wait_if_paused()
            prev_raw_state = raw_state
            refresh_player_detail_for_map(game, player, raw_state)

            with shared.agent_lock:
                forced_end_turn = is_forced_end_turn_state(agent, raw_state)
                skipped_agent = should_skip_agent(raw_state)
                if forced_end_turn:
                    action = {"type": "end_turn"}
                    q_values = forced_end_turn_q_values(raw_state)
                    action_selection = forced_end_turn_action_selection()
                elif skipped_agent:
                    action = {"type": "proceed"}
                    q_values = current_q_values(agent, prev_raw_state, action)
                    action_selection = action_selection_details(
                        agent,
                        prev_raw_state,
                        action,
                        skipped_agent,
                        q_values,
                    )
                else:
                    policy_state = {
                        "screen_type": raw_state.get("state_type"),
                        "raw_state": raw_state,
                    }
                    action = agent.choose_action(policy_state)
                    q_values = current_q_values(agent, prev_raw_state, action)
                    action_selection = action_selection_details(
                        agent,
                        prev_raw_state,
                        action,
                        skipped_agent,
                        q_values,
                    )
                
            try:
                next_raw_state, done, info = game.step(action)
            except Exception as exc:
                recovered_state, restart_episode = recover_client_state_after_disconnect(
                    game,
                    client_id,
                    base_url,
                    dashboard,
                    stop_event,
                    episode,
                    prev_raw_state,
                    exc,
                )
                if recovered_state is None or restart_episode:
                    break
                raw_state = recovered_state
                final_raw_state = recovered_state
                continue

            if info.get("action_error"):
                reward, reward_details = reward_model.action_error_reward(
                    info.get("error", "client request failed")
                )
                if is_connection_action_error(info):
                    recovered_state, restart_episode = recover_client_state_after_disconnect(
                        game,
                        client_id,
                        base_url,
                        dashboard,
                        stop_event,
                        episode,
                        prev_raw_state,
                        info.get("error", "client request failed"),
                    )
                    if recovered_state is None or restart_episode:
                        break
                    raw_state = recovered_state
                    final_raw_state = recovered_state
                    continue
            else:
                reward, reward_details = reward_model.compute(
                    prev_raw_state,
                    next_raw_state,
                    action,
                )
            auto_steps = []
            if next_raw_state is not None and not done:
                try:
                    for advance_forced_states in (
                        advance_forced_end_turn_states,
                    ):
                        if done:
                            break
                        (
                            next_raw_state,
                            auto_reward,
                            auto_done,
                            new_auto_steps,
                        ) = advance_forced_states(
                            game,
                            agent,
                            reward_model,
                            next_raw_state,
                        )
                        auto_steps.extend(new_auto_steps)
                        reward += auto_reward
                        done = done or auto_done
                except Exception as exc:
                    recovered_state, restart_episode = recover_client_state_after_disconnect(
                        game,
                        client_id,
                        base_url,
                        dashboard,
                        stop_event,
                        episode,
                        next_raw_state,
                        exc,
                    )
                    if recovered_state is None or restart_episode:
                        break
                    raw_state = recovered_state
                    final_raw_state = recovered_state
                    continue
            final_raw_state = next_raw_state
            reward_details = fold_reward_details(
                reward_details,
                reward,
                auto_steps,
            )
            training_info = None
            if not forced_end_turn:
                with shared.agent_lock:
                    training_info = agent.train_from_step(
                        prev_raw_state,
                        action,
                        reward,
                        next_raw_state,
                        done,
                        reward_details,
                    )
                    try:
                        shared.save_milestone_if_needed()
                    except Exception as exc:
                        logging.exception(
                            "Could not save battle agent milestone checkpoint at trained_steps=%d: %s",
                            agent.battle_agent.trained_steps,
                            exc,
                        )
            episode_reward += reward
            folded_step_count = 1 + len(auto_steps)
            episode_steps += folded_step_count

            loss = None
            if reward_details.get("type") == "battle":
                step_battle_reward = reward_details.get("total", reward)
                battle_reward += step_battle_reward
                current_battle_reward += step_battle_reward
                current_battle_steps += folded_step_count
                current_battle_hp_lost = reward_details.get("hp_lost", current_battle_hp_lost)
                if reward_details.get("potion_used", False):
                    current_battle_potions_used += 1
                if reward_details.get("result") == "won":
                    battle_wins += 1
                if reward_details.get("result") == "lost":
                    battle_losses += 1

                if reward_details.get("result") in {"won", "lost"}:
                    battle_number += 1
                    result = reward_details.get("result")
                    logging.info(
                        "Episode %d battle %d finished: result=%s battle_reward=%.2f "
                        "steps=%d hp=%s->%s hp_lost=%d gold_lost=%d max_hp_lost=%d "
                        "potions_used=%d hp_penalty=%.2f gold_penalty=%.2f "
                        "max_hp_penalty=%.2f win_reward=%.2f",
                        episode,
                        battle_number,
                        result,
                        current_battle_reward,
                        current_battle_steps,
                        reward_details.get("battle_start_hp"),
                        reward_details.get("next_hp"),
                        current_battle_hp_lost,
                        reward_details.get("gold_lost", 0),
                        reward_details.get("max_hp_lost", 0),
                        current_battle_potions_used,
                        reward_details.get("hp_penalty", 0.0),
                        reward_details.get("gold_penalty", 0.0),
                        reward_details.get("max_hp_penalty", 0.0),
                        reward_details.get("win_reward", 0.0),
                    )
                    current_battle_reward = 0.0
                    current_battle_steps = 0
                    current_battle_hp_lost = 0
                    current_battle_potions_used = 0

            if training_info is not None:
                battle_steps += 1
                loss = training_info["loss"]
                action_count_key = (
                    action.get("action_key")
                    or action_selection.get("action_key")
                    or training_info["action_type"]
                )
                action_counts[action_count_key] = action_counts.get(action_count_key, 0) + 1
                if training_info["updated"]:
                    update_count += 1
                    losses.append(loss)
                    with shared.agent_lock:
                        train_step = agent.battle_agent.trained_steps
                        train_epsilon = agent.battle_agent.epsilon
                        train_replay_size = len(agent.battle_agent.replay_buffer)
                        train_learn_steps = agent.battle_agent.learn_steps
                    tensorboard.add_scalars(
                        f"clients/{client_id}/train",
                        {
                            "loss": loss,
                            "reward": reward,
                            "epsilon": train_epsilon,
                            "replay_size": train_replay_size,
                            "learn_steps": train_learn_steps,
                        },
                        train_step,
                    )

            with shared.agent_lock:
                print_replay_size = len(agent.battle_agent.replay_buffer)
                print_epsilon = agent.battle_agent.epsilon
            print(
                client_id,
                action,
                "reward=", reward,
                "reward_details=", reward_details,
                "done=", done,
                "loss=", loss,
                "replay=", print_replay_size,
                "epsilon=", round(print_epsilon, 4),
            )
            if dashboard is not None:
                with shared.agent_lock:
                    epsilon = agent.battle_agent.epsilon
                    replay_size = len(agent.battle_agent.replay_buffer)
                    trained_steps = agent.battle_agent.trained_steps
                    learn_steps = agent.battle_agent.learn_steps
                dashboard.update_step(
                    client_id,
                    base_url,
                    episode,
                    episode_steps,
                    prev_raw_state,
                    next_raw_state,
                    action,
                    reward,
                    reward_details,
                    done,
                    episode_reward,
                    battle_reward,
                    current_battle_reward,
                    current_battle_steps,
                    battle_steps,
                    update_count,
                    battle_wins,
                    battle_losses,
                    loss,
                    epsilon,
                    replay_size,
                    trained_steps,
                    learn_steps,
                    action_counts,
                    action_selection,
                    q_values,
                )
            if done:
                break

            raw_state = next_raw_state
            time.sleep(0.1)

        if stop_event.is_set():
            break

        if restart_episode:
            if dashboard is not None:
                dashboard.update_client_status(client_id, base_url, "Starting new episode", episode)
            logging.info("%s episode %d discarded after reconnect; starting new episode", client_id, episode)
            episode += 1
            time.sleep(0.5)
            continue

        avg_loss = sum(losses) / len(losses) if losses else None
        min_loss = min(losses) if losses else None
        max_loss = max(losses) if losses else None
        with shared.agent_lock:
            epsilon_end = agent.battle_agent.epsilon
            replay_end = len(agent.battle_agent.replay_buffer)
        final_run = final_raw_state.get("run", {})
        final_floor = final_run.get("floor", 0)
        final_act = final_run.get("act", 0)
        final_state_type = final_raw_state.get("state_type")
        action_summary = " ".join(
            f"{action_type}:{count}"
            for action_type, count in sorted(action_counts.items())
        ) or "none"
        loss_summary = (
            f"avg_loss={avg_loss:.4f} min_loss={min_loss:.4f} max_loss={max_loss:.4f}"
            if avg_loss is not None
            else "avg_loss=n/a min_loss=n/a max_loss=n/a"
        )

        logging.info(
            "%s episode %d study summary: reward=%.2f battle_reward=%.2f steps=%d battle_steps=%d updates=%d "
            "%s epsilon=%.4f->%.4f replay=%d->%d wins=%d losses=%d final_floor=%s final_act=%s "
            "final_state=%s actions=%s",
            client_id,
            episode,
            episode_reward,
            battle_reward,
            episode_steps,
            battle_steps,
            update_count,
            loss_summary,
            epsilon_start,
            epsilon_end,
            replay_start,
            replay_end,
            battle_wins,
            battle_losses,
            final_floor,
            final_act,
            final_state_type,
            action_summary,
        )

        logging.info(
            "%s episode %d ended; returning to main menu and starting a new run",
            client_id,
            episode,
        )
        episode_result = {
            "client_id": client_id,
            "base_url": base_url,
            "episode": episode,
            "reward": episode_reward,
            "battle_reward": battle_reward,
            "steps": episode_steps,
            "battle_steps": battle_steps,
            "updates": update_count,
            "final_floor": final_floor,
            "final_act": final_act,
            "final_state_type": final_state_type,
            "avg_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "replay_start": replay_start,
            "replay_end": replay_end,
            "wins": battle_wins,
            "losses": battle_losses,
            "action_counts": dict(action_counts),
        }
        episode_log.append(episode_result)
        with shared.agent_lock:
            tensorboard_step = agent.battle_agent.trained_steps
        tensorboard.add_scalars(
            f"clients/{client_id}/episode",
            {
                "reward": episode_reward,
                "battle_reward": battle_reward,
                "steps": episode_steps,
                "battle_steps": battle_steps,
                "updates": update_count,
                "final_floor": final_floor,
                "final_act": final_act,
                "avg_loss": avg_loss,
                "min_loss": min_loss,
                "max_loss": max_loss,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "replay_start": replay_start,
                "replay_end": replay_end,
                "wins": battle_wins,
                "losses": battle_losses,
            },
            tensorboard_step,
        )
        for action_key, count in action_counts.items():
            tensorboard.add_scalar(
                f"clients/{client_id}/actions/{action_key}",
                count,
                tensorboard_step,
            )
        tensorboard.flush()
        if dashboard is not None:
            dashboard.add_episode_result(episode_result)
        if episode % SAVE_INTERVAL == 0:
            try:
                with shared.agent_lock:
                    save_battle_checkpoint(
                        agent,
                        shared.latest_path,
                        f"{client_id} episode {episode}",
                    )
            except Exception as exc:
                logging.exception(
                    "Could not save battle agent model after %s episode %d: %s",
                    client_id,
                    episode,
                    exc,
                )

        episode += 1
        time.sleep(0.5)
    except Exception as exc:
        final_client_status = f"Error: {exc}"
        logging.exception("%s stopped with error: %s", client_id, exc)
    finally:
        if dashboard is not None:
            dashboard.update_client_status(client_id, base_url, final_client_status, episode)


def main():
    """Run the training CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    dashboard_enabled = args.live or args.live_html is not None
    dashboard = None
    if dashboard_enabled:
        dashboard = LiveTrainingDashboard(
            args.live_html,
            args.live_ws_host,
            args.live_ws_port,
            args.live_http_host,
            args.live_http_port,
        )
        logging.info(
            "Serving live training dashboard at %s websocket=%s",
            dashboard.http_url,
            dashboard.websocket_url,
        )
        if args.live_html is not None:
            logging.info("Wrote static training dashboard copy to %s", args.live_html)

    episode_log = EpisodeLogWriter(args.episode_log)
    logging.info("Writing episode summaries to %s", episode_log.path)

    battle_agent_type = normalize_battle_agent_type(args.battle_agent)
    tensorboard = TensorBoardLogger(
        args.tensorboard_logdir,
        f"training_{battle_agent_type}_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    if tensorboard.enabled:
        logging.info("Writing TensorBoard training metrics to %s", tensorboard.log_dir)

    agent = Agent(battle_agent_type=battle_agent_type)
    checkpoint_dir = battle_agent_checkpoint_dir(battle_agent_type)
    latest_path = battle_latest_path(battle_agent_type)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if latest_path.exists():
        try:
            agent.battle_agent.load(str(latest_path))
            logging.info(
                "Loaded %s battle agent model from %s",
                battle_agent_type,
                latest_path,
            )
        except Exception as exc:
            logging.warning(
                "Could not load %s battle agent model from %s; starting fresh. error=%s",
                battle_agent_type,
                latest_path,
                exc,
            )
    else:
        logging.info(
            "No %s battle agent checkpoint found at %s; starting fresh",
            battle_agent_type,
            latest_path,
        )

    shared = SharedTrainingState(agent, battle_agent_type)
    stop_event = threading.Event()
    client_urls = training_client_urls(args)
    threads = []

    if dashboard is not None:
        for index, base_url in enumerate(client_urls, start=1):
            dashboard.update_client_status(f"client-{index}", base_url, "Starting", 1)

    try:
        for index, base_url in enumerate(client_urls, start=1):
            client_id = f"client-{index}"
            thread = threading.Thread(
                target=run_training_client,
                args=(
                    client_id,
                    base_url,
                    shared,
                    dashboard,
                    episode_log,
                    tensorboard,
                    stop_event,
                ),
                name=f"trainer-{client_id}",
            )
            thread.start()
            threads.append(thread)
            logging.info("Started %s for %s", client_id, base_url)

        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        logging.info("Stopping training clients...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5.0)
    finally:
        stop_event.set()
        if dashboard is not None:
            dashboard.finish()
            dashboard.close()
        tensorboard.close()


if __name__ == "__main__":
    main()
