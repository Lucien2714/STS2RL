# Training

Training uses `sts2rl.training.cli`.

Default behavior:

- starts standard singleplayer runs
- uses one shared agent across all configured clients
- battle actions use `training=True`
- battle transitions are written to replay
- the selected battle agent updates after battle transitions
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
| `--battle-agent` | `DQN` or `PPO` | `DQN` | Battle agent implementation to train. |
| `--client-host` | string | `localhost` | Host used when building STS2MCP API URLs from `--client-port`. |
| `--client-port`, `--mcp-port` | integer, repeatable | `15526` | STS2MCP API port for a game client. Repeat this flag to train with multiple running clients. |
| `--base-url` | URL, repeatable | none | Full STS2MCP API base URL for one client. Use this when host/port construction is not enough. |
| `--episode-log` | path | `logs/training_episodes.jsonl` | JSONL file that receives one summary record per completed episode. |
| `--tensorboard-logdir` | path | none | Optional TensorBoard log directory for training metrics. |
| `--policy` | `flat` or `learned` | `flat` | Battle policy module. `learned` trains a per-card embedding inside the policy; separate schema and checkpoint directory. See [card-embedding.md](card-embedding.md). |
| `--step-delay` | float | `0.0` | Seconds to sleep between steps. Was an unconditional `0.1`, which spent minutes per episode asleep; raise it only to watch a run. |
| `--no-step-print` | flag | off | Suppress the per-step console line. Episode summaries still print. |
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
uv run sts2rl-train --battle-agent PPO --client-port 15526
uv run sts2rl-train --client-port 15526 --client-port 15527
uv run sts2rl-train --base-url http://localhost:15526/api/v1
uv run sts2rl-train --live --live-http-port 8774 --live-ws-port 8775
uv run sts2rl-train --episode-log logs/experiment_a.jsonl
uv run sts2rl-train --tensorboard-logdir runs/training
uv run sts2rl-train --battle-agent PPO --policy learned --client-port 15526
uv run sts2rl-train --client-port 15526 --no-step-print   # fastest console
tensorboard --logdir runs/training
```

## Throughput

The loop is game-client bound, so the things that used to waste wall clock were
all on our side and have been removed or made optional:

- No unconditional `0.1 s` sleep per step (`--step-delay`, default `0`).
- Q-values and action-selection metadata are computed only when the live
  dashboard is attached — they cost a forward pass over every candidate.
- The per-step console line no longer serializes the full reward-details dict,
  and can be turned off entirely with `--no-step-print`.
- One `agent_lock` acquisition per step for all agent-state reads, instead of one
  per consumer.
- DQN and PPO updates score a whole minibatch in one forward pass instead of one
  pass per sample (`tests/test_batched_updates.py` pins the equivalence).

## Outputs

- Latest DQN checkpoint: `checkpoints/battleAgent/DQN/battleagent_latest.pt`
- DQN milestone checkpoints: `checkpoints/battleAgent/DQN/battleagent_step_<trained_steps>.pt`
- Latest PPO checkpoint: `checkpoints/battleAgent/PPO/battleagent_latest.pt`
- PPO milestone checkpoints: `checkpoints/battleAgent/PPO/battleagent_step_<trained_steps>.pt`
- Episode summaries: the `--episode-log` JSONL path
- Optional TensorBoard event files under `--tensorboard-logdir`
- Optional live dashboard: printed HTTP URL when `--live` or `--live-html` is used
