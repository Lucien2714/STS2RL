"""Command-line entry point for evaluating saved battle-agent checkpoints."""

import argparse
import logging
from pathlib import Path
import threading

from sts2rl.checkpoints.manager import CHECKPOINT_DIR

from sts2rl.evaluation.checkpoints import find_checkpoints
from sts2rl.evaluation.config import (
    DEFAULT_EPISODES,
    DEFAULT_GAME_MODE,
    DEFAULT_LIVE_HTTP_HOST,
    DEFAULT_LIVE_HTTP_PORT,
    DEFAULT_LIVE_WS_HOST,
    DEFAULT_LIVE_WS_PORT,
    DEFAULT_MAX_STEPS,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
)
from sts2rl.evaluation.dashboard import LiveEvaluationDashboard
from sts2rl.evaluation.runner import evaluate_checkpoint, write_csv
from sts2rl.evaluation.seeds import evaluation_client_episode_seeds

DEFAULT_CLIENT_PORT = 15526


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate each saved battle-agent checkpoint without training."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--character", type=int, default=0)
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
            "STS2 MCP API port for an evaluation client. Specify once per client, "
            "for example --client-port 15526 --client-port 15527 --client-port 15528."
        ),
    )
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        default=None,
        help=(
            "Full STS2 MCP API base URL for one evaluation client. Can be specified "
            "multiple times; defaults to http://localhost:15526/api/v1 when no clients are set."
        ),
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--game-mode",
        choices=["standard", "daily", "custom"],
        default=DEFAULT_GAME_MODE,
    )
    parser.add_argument(
        "--seed",
        default=None,
        help=(
            "Optional base seed for deterministic client/episode seeds. "
            "Seeds are generated in client order: seed, seed_2, seed_3, and so on."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum steps per evaluation episode. Use 0 for no step limit.",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--csv", type=Path, default=None)
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset",
        dest="reset",
        action="store_true",
        default=True,
        help="Reset/start a fresh run before each evaluation episode. This is the default.",
    )
    reset_group.add_argument(
        "--no-reset",
        "--current-state",
        dest="reset",
        action="store_false",
        help="Start evaluation episodes from the current backend game state instead of resetting.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Serve the live evaluation dashboard over HTTP.",
    )
    parser.add_argument(
        "--auto-pause",
        action="store_true",
        help="Automatically pause the live dashboard before each checkpoint and between episodes.",
    )
    parser.add_argument(
        "--live-html",
        type=Path,
        default=None,
        help="Optionally write a static copy of the live dashboard HTML to this file.",
    )
    parser.add_argument(
        "--live-http-host",
        default=DEFAULT_LIVE_HTTP_HOST,
        help="Host interface for the live dashboard HTTP server.",
    )
    parser.add_argument(
        "--live-http-port",
        type=int,
        default=DEFAULT_LIVE_HTTP_PORT,
        help="Port for the live dashboard HTTP server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--live-ws-host",
        default=DEFAULT_LIVE_WS_HOST,
        help="Host interface for the live dashboard WebSocket server.",
    )
    parser.add_argument(
        "--live-ws-port",
        type=int,
        default=DEFAULT_LIVE_WS_PORT,
        help="Port for the live dashboard WebSocket server. Use 0 to choose a free port.",
    )
    return parser.parse_args()


def base_url_from_port(host: str, port: int) -> str:
    """Build an STS2MCP API base URL from host and port."""
    return f"http://{host}:{port}/api/v1"


def evaluation_client_urls(args: argparse.Namespace) -> list[str]:
    """Resolve CLI client-port/base-url arguments into API base URLs."""
    urls = list(args.base_urls or [])
    urls.extend(base_url_from_port(args.client_host, port) for port in args.client_ports or [])
    if not urls:
        urls.append(base_url_from_port(args.client_host, DEFAULT_CLIENT_PORT))
    return urls


class MultiClientEpisodePause:
    """Barrier that synchronizes multi-client evaluation between episodes."""

    def __init__(
        self,
        client_count: int,
        episodes: int,
        dashboard: LiveEvaluationDashboard | None,
        pause_between_episodes: bool,
    ) -> None:
        self.client_count = client_count
        self.episodes = episodes
        self.dashboard = dashboard
        self.pause_between_episodes = pause_between_episodes
        self.condition = threading.Condition()
        self.completed = 0
        self.generation = 0
        self.aborted = False

    def abort(self) -> None:
        """Wake waiting clients and mark the barrier as aborted."""
        with self.condition:
            self.aborted = True
            self.condition.notify_all()

    def wait(
        self,
        client_id: str,
        base_url: str,
        checkpoint_name: str,
        episode_index: int,
    ) -> None:
        """Wait until every client finishes the current episode."""
        if self.dashboard is not None:
            self.dashboard.update_client_status(
                client_id,
                base_url,
                f"Finished episode {episode_index}; waiting for other clients",
                checkpoint_name,
                episode_index,
            )

        with self.condition:
            if self.aborted:
                raise RuntimeError("Evaluation episode sync aborted")

            generation = self.generation
            self.completed += 1
            if self.completed >= self.client_count:
                self.completed = 0
                self.generation += 1
                if self.dashboard is not None and self.pause_between_episodes:
                    self.dashboard.pause_between_episodes(checkpoint_name, episode_index)
                self.condition.notify_all()
            else:
                while not self.aborted and generation == self.generation:
                    self.condition.wait(timeout=0.5)

            if self.aborted:
                raise RuntimeError("Evaluation episode sync aborted")

        if self.dashboard is not None:
            waiting_status = (
                "Waiting to finish checkpoint"
                if episode_index >= self.episodes
                else f"Waiting to start episode {episode_index + 1}"
            )
            next_status = "Finalizing" if episode_index >= self.episodes else "Evaluating"
            next_episode = episode_index if episode_index >= self.episodes else episode_index + 1
            self.dashboard.update_client_status(
                client_id,
                base_url,
                waiting_status,
                checkpoint_name,
                episode_index,
            )
            self.dashboard.wait_if_paused()
            self.dashboard.update_client_status(
                client_id,
                base_url,
                next_status,
                checkpoint_name,
                next_episode,
            )


def main() -> None:
    """Run the checkpoint evaluation CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        logging.warning("No checkpoints found in %s", args.checkpoint_dir)
        return

    dashboard_enabled = args.live or args.live_html is not None
    dashboard = None
    if dashboard_enabled:
        dashboard = LiveEvaluationDashboard(
            args.live_html,
            args.live_ws_host,
            args.live_ws_port,
            args.live_http_host,
            args.live_http_port,
            initial_paused=args.auto_pause,
        )
    if dashboard is not None:
        logging.info(
            "Serving live evaluation dashboard at %s websocket=%s",
            dashboard.http_url,
            dashboard.websocket_url,
        )
        if args.live_html is not None:
            logging.info("Wrote static dashboard copy to %s", args.live_html)

    rows = []
    client_urls = evaluation_client_urls(args)
    client_episode_seeds = evaluation_client_episode_seeds(
        args.episodes,
        len(client_urls),
        args.seed,
    )
    seed_groups = [
        f"client-{index}=[{', '.join(seeds)}]"
        for index, seeds in enumerate(client_episode_seeds, start=1)
    ]
    seed_summary = "; ".join(seed_groups)
    logging.info("Evaluation episode seeds: %s", seed_summary)
    print(f"Evaluation episode seeds: {seed_summary}")
    try:
        for checkpoint_path in checkpoints:
            if dashboard is not None and args.auto_pause:
                dashboard.set_paused(True)
            if dashboard is not None:
                for index, base_url in enumerate(client_urls, start=1):
                    dashboard.update_client_status(
                        f"client-{index}",
                        base_url,
                        "Ready",
                        checkpoint_path.name,
                        1,
                    )
            logging.info("Evaluating checkpoint %s", checkpoint_path)
            try:
                result = evaluate_checkpoint_clients(
                    checkpoint_path,
                    client_urls,
                    args,
                    dashboard,
                    client_episode_seeds,
                )
            except Exception as exc:
                logging.exception("Could not evaluate checkpoint %s: %s", checkpoint_path, exc)
                continue

            rows.append(result)
            if dashboard is not None:
                dashboard.add_checkpoint_result(result)
            print_checkpoint_result(result)

        if args.csv is not None:
            write_csv(args.csv, rows)
            logging.info("Wrote evaluation results to %s", args.csv)

        if dashboard is not None:
            dashboard.finish()
    finally:
        if dashboard is not None:
            dashboard.close()


def evaluate_checkpoint_clients(
    checkpoint_path: Path,
    client_urls: list[str],
    args: argparse.Namespace,
    dashboard: LiveEvaluationDashboard | None,
    client_episode_seeds: list[list[str]],
) -> dict:
    """Evaluate one checkpoint using one or more STS2MCP clients."""
    if len(client_urls) == 1:
        return evaluate_checkpoint(
            checkpoint_path,
            args.episodes,
            args.character,
            client_urls[0],
            args.timeout,
            args.game_mode,
            client_episode_seeds[0],
            args.max_steps,
            args.sleep,
            args.reset,
            dashboard,
            client_id="client-1",
            pause_between_episodes=args.auto_pause,
        )

    results: list[dict] = []
    errors: list[tuple[str, Exception]] = []
    lock = threading.Lock()
    threads = []
    sync_between_episodes = (not args.reset) or args.auto_pause
    episode_gate = (
        MultiClientEpisodePause(
            len(client_urls),
            args.episodes,
            dashboard,
            args.auto_pause,
        )
        if sync_between_episodes
        else None
    )

    def worker(index: int, base_url: str) -> None:
        """Evaluate the checkpoint on one client thread."""
        client_id = f"client-{index}"
        episode_seeds = client_episode_seeds[index - 1]
        if dashboard is not None:
            dashboard.update_client_status(client_id, base_url, "Waiting", checkpoint_path.name, 1)
        try:
            result = evaluate_checkpoint(
                checkpoint_path,
                args.episodes,
                args.character,
                base_url,
                args.timeout,
                args.game_mode,
                episode_seeds,
                args.max_steps,
                args.sleep,
                args.reset,
                dashboard,
                client_id=client_id,
                pause_between_episodes=False,
                after_episode=episode_gate.wait if episode_gate is not None else None,
            )
            with lock:
                results.append(result)
            if dashboard is not None:
                dashboard.update_client_status(
                    client_id,
                    base_url,
                    "Finished",
                    checkpoint_path.name,
                    args.episodes,
                )
        except Exception as exc:
            logging.exception("%s could not evaluate checkpoint %s: %s", client_id, checkpoint_path, exc)
            if episode_gate is not None:
                episode_gate.abort()
            with lock:
                errors.append((client_id, exc))
            if dashboard is not None:
                dashboard.update_client_status(
                    client_id,
                    base_url,
                    f"Error: {exc}",
                    checkpoint_path.name,
                    None,
                )

    for index, base_url in enumerate(client_urls, start=1):
        thread = threading.Thread(
            target=worker,
            args=(index, base_url),
            name=f"eval-client-{index}",
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if not results:
        joined_errors = "; ".join(f"{client}: {error}" for client, error in errors)
        raise RuntimeError(f"All evaluation clients failed: {joined_errors}")

    return aggregate_client_results(checkpoint_path, results, client_urls)


def aggregate_client_results(
    checkpoint_path: Path,
    results: list[dict],
    client_urls: list[str],
) -> dict:
    """Combine per-client checkpoint summaries into one weighted result."""
    episode_count = sum(result.get("episodes", 0) for result in results)
    episode_count = max(1, episode_count)

    weighted_keys = (
        "avg_reward",
        "avg_battle_reward",
        "avg_steps",
        "avg_battle_steps",
        "avg_floor",
    )
    aggregate = dict(results[0])
    for key in weighted_keys:
        aggregate[key] = (
            sum(result.get(key, 0.0) * result.get("episodes", 0) for result in results)
            / episode_count
        )

    battle_wins = sum(result.get("battle_wins", 0) for result in results)
    battle_losses = sum(result.get("battle_losses", 0) for result in results)
    aggregate.update(
        {
            "checkpoint": checkpoint_path.name,
            "path": str(checkpoint_path),
            "client_id": "all",
            "base_url": ",".join(client_urls),
            "clients": len(results),
            "episodes": sum(result.get("episodes", 0) for result in results),
            "battle_wins": battle_wins,
            "battle_losses": battle_losses,
            "battle_win_rate": battle_wins / max(1, battle_wins + battle_losses),
            "max_floor": max((result.get("max_floor", 0) for result in results), default=0),
            "timeouts": sum(result.get("timeouts", 0) for result in results),
            "seeds": ",".join(result.get("seeds", "") for result in results if result.get("seeds")),
            "ending_steps": ",".join(
                result.get("ending_steps", "")
                for result in results
                if result.get("ending_steps")
            ),
        }
    )
    return aggregate


def print_checkpoint_result(result: dict) -> None:
    """Print a compact one-line evaluation summary."""
    print(
        f"{result['checkpoint']}: "
        f"trained_steps={result['trained_steps']} "
        f"learn_steps={result['learn_steps']} "
        f"game_mode={result['game_mode']} "
        f"start_mode={result['start_mode']} "
        f"clients={result.get('clients', 1)} "
        f"seeds=[{result['seeds']}] "
        f"avg_reward={result['avg_reward']:.2f} "
        f"avg_battle_reward={result['avg_battle_reward']:.2f} "
        f"avg_steps={result['avg_steps']:.1f} "
        f"ending_steps=[{result['ending_steps']}] "
        f"battle_win_rate={result['battle_win_rate']:.2%} "
        f"wins={result['battle_wins']} losses={result['battle_losses']} "
        f"avg_floor={result['avg_floor']:.1f} max_floor={result['max_floor']} "
        f"timeouts={result['timeouts']}"
    )
