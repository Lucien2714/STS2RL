"""Live HTTP and WebSocket dashboard for checkpoint evaluation."""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Callable

DASHBOARD_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    """Threading server configured for quick local dashboard restarts."""

    allow_reuse_address = True
    daemon_threads = True


class LiveEvaluationHttpHandler(http.server.BaseHTTPRequestHandler):
    """Serve dashboard HTML and health snapshots over HTTP."""

    def do_GET(self) -> None:
        """Handle dashboard and health-check GET requests."""
        dashboard: LiveEvaluationDashboard = self.server.dashboard
        if self.path in {"/", "/evaluation_live.html"}:
            self.send_text(dashboard.render_html(), "text/html; charset=utf-8")
            return
        if self.path == "/health":
            self.send_text(json.dumps(dashboard.snapshot()), "application/json")
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP request logging."""
        return

    def send_text(self, body: str, content_type: str) -> None:
        """Send a plain text or HTML response with no-store caching."""
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


class LiveEvaluationHttpServer:
    """Small background HTTP server for the live dashboard."""

    def __init__(self, host: str, port: int, dashboard: "LiveEvaluationDashboard") -> None:
        self.server = ReusableThreadingTCPServer((host, port), LiveEvaluationHttpHandler)
        self.server.dashboard = dashboard
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start serving HTTP requests in a background thread."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the HTTP server and close its listening socket."""
        self.server.shutdown()
        self.server.server_close()


class LiveEvaluationWebSocketHandler(socketserver.BaseRequestHandler):
    """Handle one WebSocket client connection."""

    def handle(self) -> None:
        """Perform handshake, register the client, and read control messages."""
        socket_server: LiveEvaluationWebSocketServer = self.server.dashboard_server
        try:
            headers = self.read_headers()
            key = headers.get("sec-websocket-key")
            if key is None:
                return
            self.send_handshake(key)
            socket_server.add_client(self.request)
            self.read_until_close(socket_server)
        finally:
            socket_server.remove_client(self.request)

    def read_headers(self) -> dict[str, str]:
        """Read and parse the initial HTTP upgrade headers."""
        data = b""
        self.request.settimeout(2.0)
        while b"\r\n\r\n" not in data and len(data) < 8192:
            chunk = self.request.recv(1024)
            if not chunk:
                break
            data += chunk

        headers = {}
        lines = data.decode("utf-8", errors="ignore").split("\r\n")
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return headers

    def send_handshake(self, key: str) -> None:
        """Send the RFC 6455 WebSocket upgrade response."""
        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        self.request.sendall(response.encode("ascii"))

    def read_until_close(self, socket_server: "LiveEvaluationWebSocketServer") -> None:
        """Read WebSocket frames until the client closes or the server stops."""
        self.request.settimeout(1.0)
        while not socket_server.closed:
            try:
                header = self.read_exact(2)
            except socket.timeout:
                continue
            except OSError:
                break
            if not header:
                break
            try:
                opcode = header[0] & 0x0F
                masked = bool(header[1] & 0x80)
                length = header[1] & 0x7F
                if length == 126:
                    length = int.from_bytes(self.read_exact(2), "big")
                elif length == 127:
                    length = int.from_bytes(self.read_exact(8), "big")
                mask = self.read_exact(4) if masked else b""
                payload = self.read_exact(length) if length else b""
            except OSError:
                break
            if masked and payload:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                break
            if opcode == 0x9:
                self.request.sendall(encode_websocket_frame(payload, opcode=0xA))
            if opcode == 0x1 and payload:
                socket_server.handle_message(payload)

    def read_exact(self, length: int) -> bytes:
        """Read exactly length bytes unless the socket closes first."""
        data = b""
        while len(data) < length:
            chunk = self.request.recv(length - len(data))
            if not chunk:
                break
            data += chunk
        return data


class LiveEvaluationWebSocketServer:
    """Minimal local WebSocket broadcaster for dashboard snapshots."""

    def __init__(
        self,
        host: str,
        port: int,
        snapshot_provider: Callable[[], dict],
        message_handler: Callable[[dict], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.snapshot_provider = snapshot_provider
        self.message_handler = message_handler
        self.clients: set[socket.socket] = set()
        self.lock = threading.Lock()
        self.closed = False
        self.server = ReusableThreadingTCPServer((host, port), LiveEvaluationWebSocketHandler)
        self.server.dashboard_server = self
        self.host, self.port = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start accepting WebSocket clients in a background thread."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the WebSocket server and close all client sockets."""
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        with self.lock:
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass

    def add_client(self, client: socket.socket) -> None:
        """Register a client and immediately send the current snapshot."""
        with self.lock:
            self.clients.add(client)
        self.send_to_client(client, self.snapshot_provider())

    def remove_client(self, client: socket.socket) -> None:
        """Remove a disconnected client from the broadcast set."""
        with self.lock:
            self.clients.discard(client)

    def broadcast(self, payload: dict) -> None:
        """Send a JSON payload to all connected clients."""
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            self.send_to_client(client, payload)

    def send_to_client(self, client: socket.socket, payload: dict) -> None:
        """Send one JSON payload and drop the client on socket failure."""
        try:
            client.sendall(encode_websocket_frame(json.dumps(payload, default=str).encode("utf-8")))
        except OSError:
            self.remove_client(client)

    def handle_message(self, payload: bytes) -> None:
        """Decode and dispatch a JSON control message from a client."""
        if self.message_handler is None:
            return
        try:
            message = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if isinstance(message, dict):
            self.message_handler(message)


def encode_websocket_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """Encode a server-to-client WebSocket frame."""
    header = bytes([0x80 | opcode])
    length = len(payload)
    if length < 126:
        return header + bytes([length]) + payload
    if length < 65536:
        return header + bytes([126]) + length.to_bytes(2, "big") + payload
    return header + bytes([127]) + length.to_bytes(8, "big") + payload


class LiveEvaluationDashboard:
    """Serve and broadcast live evaluation progress and pause controls."""

    def __init__(
        self,
        path: Path | None,
        websocket_host: str,
        websocket_port: int,
        http_host: str,
        http_port: int,
        initial_paused: bool = False,
    ) -> None:
        self.path = path
        self.started_at = time.time()
        self.status = "Waiting to start evaluation"
        self.paused = initial_paused
        self._pause_condition = threading.Condition()
        self._data_lock = threading.RLock()
        self.current_step: dict = {}
        self.current_steps: dict[str, dict] = {}
        self.rows: list[dict] = []
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
        checkpoint_name: str,
        episode_index: int,
        client_id: str,
        base_url: str,
        run_seed: str | None,
        step: int,
        raw_state: dict,
        next_raw_state: dict,
        action: dict,
        reward: float,
        episode_reward: float,
        battle_reward: float,
        battle_wins: int,
        battle_losses: int,
        done: bool,
        q_values: dict | None = None,
    ) -> None:
        """Record and broadcast one evaluation step."""
        run = next_raw_state.get("run", {})
        self.status = "Paused" if self.paused else "Evaluating"
        current_step = {
            "checkpoint": checkpoint_name,
            "client_id": client_id,
            "base_url": base_url,
            "episode": episode_index,
            "seed": run_seed or "normal",
            "step": step,
            "from_state": raw_state.get("state_type"),
            "to_state": next_raw_state.get("state_type"),
            "action": action,
            "reward": reward,
            "episode_reward": episode_reward,
            "battle_reward": battle_reward,
            "battle_wins": battle_wins,
            "battle_losses": battle_losses,
            "floor": run.get("floor", 0),
            "act": run.get("act", 0),
            "done": done,
            "q_values": q_values or {},
        }
        with self._data_lock:
            self.current_step = current_step
            self.current_steps[client_id] = current_step
        self.broadcast()

    def add_checkpoint_result(self, result: dict) -> None:
        """Record and broadcast a completed checkpoint summary."""
        self.status = "Paused" if self.paused else f"Finished {result['checkpoint']}"
        with self._data_lock:
            self.rows.append(result)
        self.broadcast()

    def update_client_status(
        self,
        client_id: str,
        base_url: str,
        status: str,
        checkpoint_name: str | None = None,
        episode_index: int | None = None,
    ) -> None:
        """Update the dashboard status for one evaluation client."""
        with self._data_lock:
            current = dict(self.current_steps.get(client_id, {}))
            current.update(
                {
                    "client_id": client_id,
                    "base_url": base_url,
                    "client_status": status,
                    "checkpoint": checkpoint_name or current.get("checkpoint"),
                    "episode": episode_index
                    if episode_index is not None
                    else current.get("episode"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            self.current_steps[client_id] = current
        self.broadcast()

    def handle_control_message(self, message: dict) -> None:
        """Apply pause/resume commands received from dashboard clients."""
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
        self.status = "Paused" if paused else "Evaluating"
        self.broadcast()

    def pause_between_episodes(self, checkpoint_name: str, episode_index: int) -> None:
        """Pause the dashboard after an episode when auto-pause is enabled."""
        with self._pause_condition:
            self.paused = True
        self.status = (
            f"Episode {episode_index} complete for {checkpoint_name}; waiting to start eval"
        )
        self.broadcast()

    def resume_waiters(self) -> None:
        """Resume any evaluation threads blocked by pause state."""
        with self._pause_condition:
            self.paused = False
            self._pause_condition.notify_all()

    def wait_if_paused(self) -> None:
        """Block the caller while the dashboard is paused."""
        announced = False
        with self._pause_condition:
            while self.paused:
                if not announced:
                    self.status = "Paused"
                    self.broadcast()
                    announced = True
                self._pause_condition.wait(timeout=0.5)

    def finish(self) -> None:
        """Mark evaluation complete and broadcast the final state."""
        self.resume_waiters()
        self.status = "Evaluation complete"
        self.broadcast()

    def close(self) -> None:
        """Stop dashboard servers and release paused evaluation threads."""
        self.resume_waiters()
        self.http_server.stop()
        self.websocket_server.stop()

    def snapshot(self) -> dict:
        """Return a thread-safe copy of dashboard state."""
        with self._data_lock:
            current_step = dict(self.current_step)
            current_steps = {
                client_id: dict(step) for client_id, step in self.current_steps.items()
            }
            rows = list(self.rows)
        return {
            "status": self.status,
            "started_at": self.started_at,
            "elapsed": int(time.time() - self.started_at),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paused": self.paused,
            "current_step": current_step,
            "current_steps": current_steps,
            "rows": rows,
        }

    def broadcast(self) -> None:
        """Broadcast the latest snapshot to WebSocket clients."""
        self.websocket_server.broadcast(self.snapshot())

    def write_html(self) -> None:
        """Write a static copy of the dashboard HTML when requested."""
        if self.path is None:
            return
        self.path.write_text(self.render_html(), encoding="utf-8")

    def render_html(self) -> str:
        """Render dashboard HTML with the active WebSocket URL embedded."""
        template = DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
        return template.replace("__WS_URL__", json.dumps(self.websocket_url))
