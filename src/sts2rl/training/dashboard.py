"""Live HTTP/WebSocket dashboard adapter for training telemetry."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from sts2rl.evaluation.dashboard import LiveEvaluationHttpServer, LiveEvaluationWebSocketServer

TRAINING_DASHBOARD_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")


class LiveTrainingDashboard:
    """Serve and broadcast live multi-client training progress."""

    def __init__(
        self,
        path: Path | None,
        websocket_host: str,
        websocket_port: int,
        http_host: str,
        http_port: int,
    ) -> None:
        self.path = path
        self.started_at = time.time()
        self.status = "Starting training"
        self.paused = False
        self._pause_condition = threading.Condition()
        self._data_lock = threading.RLock()
        self.current_step: dict = {}
        self.client_steps: dict[str, dict] = {}
        self.client_recent_steps: dict[str, list[dict]] = {}
        self.episode_rows: list[dict] = []
        self.recent_steps: list[dict] = []
        self.websocket_server = LiveEvaluationWebSocketServer(
            websocket_host,
            websocket_port,
            self.snapshot,
            self.handle_control_message,
        )
        self.websocket_server.start()
        self.http_server = LiveEvaluationHttpServer(http_host, http_port, self)
        self.http_server.start()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.write_html()
        self.broadcast()

    @property
    def websocket_url(self) -> str:
        """Return the public WebSocket URL for live updates."""
        return f"ws://{self.websocket_server.host}:{self.websocket_server.port}"

    @property
    def http_url(self) -> str:
        """Return the public HTTP URL for the dashboard."""
        return f"http://{self.http_server.host}:{self.http_server.port}/"

    def update_step(
        self,
        client_id: str,
        base_url: str,
        episode: int,
        episode_steps: int,
        raw_state: dict,
        next_raw_state: dict,
        action: dict,
        reward: float,
        reward_details: dict,
        done: bool,
        episode_reward: float,
        battle_reward: float,
        current_battle_reward: float,
        current_battle_steps: int,
        battle_steps: int,
        update_count: int,
        battle_wins: int,
        battle_losses: int,
        loss: float | None,
        epsilon: float,
        replay_size: int,
        trained_steps: int,
        learn_steps: int,
        action_counts: dict,
        action_selection: dict,
        q_values: dict,
    ) -> None:
        run = next_raw_state.get("run", {})
        self.status = "Paused" if self.paused else "Training"
        current_step = {
            "client_id": client_id,
            "base_url": base_url,
            "client_status": "Training",
            "episode": episode,
            "episode_steps": episode_steps,
            "from_state": raw_state.get("state_type"),
            "to_state": next_raw_state.get("state_type"),
            "action": action,
            "reward": reward,
            "reward_details": reward_details,
            "done": done,
            "episode_reward": episode_reward,
            "battle_reward": battle_reward,
            "current_battle_reward": current_battle_reward,
            "current_battle_steps": current_battle_steps,
            "battle_steps": battle_steps,
            "update_count": update_count,
            "battle_wins": battle_wins,
            "battle_losses": battle_losses,
            "loss": loss,
            "epsilon": epsilon,
            "replay_size": replay_size,
            "trained_steps": trained_steps,
            "learn_steps": learn_steps,
            "action_counts": dict(action_counts),
            "action_selection": action_selection,
            "q_values": q_values,
            "floor": run.get("floor", 0),
            "act": run.get("act", 0),
        }
        recent_step = {
            "client_id": client_id,
            "base_url": base_url,
            "episode": episode,
            "step": episode_steps,
            "reward": reward,
            "episode_reward": episode_reward,
            "battle_reward": battle_reward,
            "loss": loss,
            "epsilon": epsilon,
            "replay_size": replay_size,
            "trained_steps": trained_steps,
            "learn_steps": learn_steps,
        }
        with self._data_lock:
            self.current_step = current_step
            self.client_steps[client_id] = current_step
            self.recent_steps.append(recent_step)
            self.recent_steps = self.recent_steps[-300:]
            client_recent_steps = self.client_recent_steps.setdefault(client_id, [])
            client_recent_steps.append(recent_step)
            self.client_recent_steps[client_id] = client_recent_steps[-300:]
        self.broadcast()

    def add_episode_result(self, result: dict) -> None:
        """Record and broadcast a completed episode summary."""
        self.status = "Paused" if self.paused else f"Finished episode {result['episode']}"
        with self._data_lock:
            self.episode_rows.append(result)
            self.episode_rows = self.episode_rows[-100:]
        self.broadcast()

    def update_client_status(
        self,
        client_id: str,
        base_url: str,
        status: str,
        episode: int | None = None,
    ) -> None:
        """Update the status row for a training client."""
        with self._data_lock:
            current = dict(self.client_steps.get(client_id, {}))
            current.update(
                {
                    "client_id": client_id,
                    "base_url": base_url,
                    "client_status": status,
                    "episode": episode if episode is not None else current.get("episode"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            self.client_steps[client_id] = current
        self.broadcast()

    def handle_control_message(self, message: dict) -> None:
        """Apply pause and resume commands received from the dashboard UI."""
        command = message.get("command")
        if command == "set_paused":
            self.set_paused(bool(message.get("paused")))
        elif command == "toggle_pause":
            self.set_paused(not self.paused)

    def set_paused(self, paused: bool) -> None:
        """Set the pause state and wake waiters when resuming."""
        with self._pause_condition:
            self.paused = paused
            if not paused:
                self._pause_condition.notify_all()
        self.status = "Paused" if paused else "Training"
        self.broadcast()

    def resume_waiters(self) -> None:
        """Resume any training threads waiting on the pause condition."""
        with self._pause_condition:
            self.paused = False
            self._pause_condition.notify_all()

    def wait_if_paused(self) -> None:
        """Block the caller while the live dashboard is paused."""
        announced = False
        with self._pause_condition:
            while self.paused:
                if not announced:
                    self.status = "Paused"
                    self.broadcast()
                    announced = True
                self._pause_condition.wait(timeout=0.5)

    def finish(self) -> None:
        """Mark training as stopped and broadcast the final state."""
        self.resume_waiters()
        self.status = "Training stopped"
        self.broadcast()

    def close(self) -> None:
        """Stop dashboard servers and release paused training threads."""
        self.resume_waiters()
        self.http_server.stop()
        self.websocket_server.stop()

    def snapshot(self) -> dict:
        """Return a thread-safe copy of the dashboard state."""
        with self._data_lock:
            current_step = dict(self.current_step)
            client_steps = {client_id: dict(step) for client_id, step in self.client_steps.items()}
            client_recent_steps = {
                client_id: list(steps) for client_id, steps in self.client_recent_steps.items()
            }
            recent_steps = list(self.recent_steps)
            episode_rows = list(self.episode_rows)
        return {
            "status": self.status,
            "started_at": self.started_at,
            "elapsed": int(time.time() - self.started_at),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paused": self.paused,
            "current_step": current_step,
            "client_steps": client_steps,
            "recent_steps_by_client": client_recent_steps,
            "recent_steps": recent_steps,
            "episode_rows": episode_rows,
        }

    def broadcast(self) -> None:
        """Broadcast the latest snapshot to connected WebSocket clients."""
        self.websocket_server.broadcast(self.snapshot())

    def write_html(self) -> None:
        """Write a static copy of the dashboard HTML when requested."""
        if self.path is None:
            return
        self.path.write_text(self.render_html(), encoding="utf-8")

    def render_html(self) -> str:
        """Render dashboard HTML with the active WebSocket URL embedded."""
        template = TRAINING_DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
        return template.replace("__WS_URL__", json.dumps(self.websocket_url))
