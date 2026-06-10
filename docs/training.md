# Training

Training uses `sts2rl.training.cli`.

Default behavior:

- starts standard singleplayer runs
- uses one shared agent across all configured clients
- battle actions use `training=True`
- battle transitions are written to replay
- DQN updates run after battle transitions
- latest checkpoint is saved every 10 episodes
- milestone backups are saved every 20,000 trained steps
- reconnect recovery is preserved from the prototype

Run:

```bash
uv run sts2rl-train
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `--client-host` | string | `localhost` | Host used when building STS2MCP API URLs from `--client-port`. |
| `--client-port`, `--mcp-port` | integer, repeatable | `15526` | STS2MCP API port for a game client. Repeat this flag to train with multiple running clients. |
| `--base-url` | URL, repeatable | none | Full STS2MCP API base URL for one client. Use this when host/port construction is not enough. |
| `--episode-log` | path | `logs/training_episodes.jsonl` | JSONL file that receives one summary record per completed episode. |
| `--live` | flag | off | Start the live training dashboard HTTP and WebSocket servers. |
| `--live-html` | path | none | Write a static copy of the live dashboard HTML to this path. Also enables dashboard setup. |
| `--live-http-host` | string | `127.0.0.1` | Interface for the live dashboard HTTP server. |
| `--live-http-port` | integer | `8774` | Port for the live dashboard HTTP server. Use `0` to choose a free port. |
| `--live-ws-host` | string | `127.0.0.1` | Interface for the live dashboard WebSocket server. |
| `--live-ws-port` | integer | `8775` | Port for the live dashboard WebSocket server. Use `0` to choose a free port. |

Client selection rules:

- If no client option is provided, training uses `http://localhost:15526/api/v1`.
- Each `--client-port` creates one URL using `--client-host`.
- Each `--base-url` is used exactly as provided.
- `--client-port` and `--base-url` can be combined; all resolved clients share one trainable agent.

Examples:

```bash
uv run sts2rl-train --client-port 15526
uv run sts2rl-train --client-port 15526 --client-port 15527
uv run sts2rl-train --base-url http://localhost:15526/api/v1
uv run sts2rl-train --live --live-http-port 8774 --live-ws-port 8775
uv run sts2rl-train --episode-log logs/experiment_a.jsonl
```

## Outputs

- Latest checkpoint: `checkpoints/battle_agent_latest.pt`
- Milestone checkpoints: `checkpoints/battle_agent_step_<trained_steps>.pt`
- Episode summaries: the `--episode-log` JSONL path
- Optional live dashboard: printed HTTP URL when `--live` or `--live-html` is used
