# Evaluation

Evaluation uses `sts2rl.evaluation.cli`.

Default behavior:

- evaluates every checkpoint in `checkpoints/`
- uses custom seeded runs by default
- pre-generates `episodes * clients` seeds
- assigns each client its own episode seeds
- reuses the same seed assignment for every checkpoint
- battle actions use `training=False`
- no replay writes, model updates, or checkpoint saves occur
- clients do not wait between episodes unless `--no-reset` or `--auto-pause`

Run:

```bash
uv run sts2rl-evaluate --episodes 3 --client-port 15526
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `--checkpoint-dir` | path | `checkpoints` | Directory scanned for `.pt` battle-agent checkpoints. Step checkpoints are evaluated before the latest checkpoint. |
| `--episodes` | integer | `3` | Number of episodes to run for each checkpoint on each client. |
| `--character` | integer | `0` | Character index passed to the environment. Current mapping is `0=IRONCLAD`, `1=SILENT`, `2=REGENT`, `3=NECROBINDER`, `4=DEFECT`. |
| `--client-host` | string | `localhost` | Host used when building STS2MCP API URLs from `--client-port`. |
| `--client-port`, `--mcp-port` | integer, repeatable | `15526` | STS2MCP API port for one evaluation client. Repeat for multi-client evaluation. |
| `--base-url` | URL, repeatable | none | Full STS2MCP API base URL for one evaluation client. |
| `--timeout` | float | `20.0` | HTTP timeout in seconds for STS2MCP requests. |
| `--game-mode` | choice | `custom` | Run mode for reset episodes: `standard`, `daily`, or `custom`. |
| `--seed` | string | random | Base seed for deterministic seed allocation. Episode seeds become `seed`, `seed_2`, `seed_3`, and so on. |
| `--max-steps` | integer | `0` | Maximum steps per episode. `0` means no step limit. |
| `--sleep` | float | `0.3` | Seconds to sleep between evaluation steps. |
| `--csv` | path | none | Optional CSV output path for checkpoint summaries. |
| `--reset` | flag | on | Reset/start a fresh run before each evaluation episode. |
| `--no-reset`, `--current-state` | flag | off | Start episodes from the current backend game state instead of resetting. |
| `--live` | flag | off | Serve the live evaluation dashboard over HTTP. |
| `--auto-pause` | flag | off | Pause the live dashboard before each checkpoint and between episodes. |
| `--live-html` | path | none | Write a static copy of the live dashboard HTML to this path. Also enables dashboard setup. |
| `--live-http-host` | string | `127.0.0.1` | Interface for the live dashboard HTTP server. |
| `--live-http-port` | integer | `8764` | Port for the live dashboard HTTP server. Use `0` to choose a free port. |
| `--live-ws-host` | string | `127.0.0.1` | Interface for the live dashboard WebSocket server. |
| `--live-ws-port` | integer | `8765` | Port for the live dashboard WebSocket server. Use `0` to choose a free port. |

Client selection rules:

- If no client option is provided, evaluation uses `http://localhost:15526/api/v1`.
- Each `--client-port` creates one URL using `--client-host`.
- Each `--base-url` is used exactly as provided.
- Multiple clients evaluate the same checkpoint in parallel and aggregate results.

Seed behavior:

- With `--reset` and `--game-mode custom` or `daily`, each episode receives a generated seed.
- With `--seed EVALRUN --episodes 3 --client-port 15526 --client-port 15527`, client 1 gets `EVALRUN`, `EVALRUN_2`, `EVALRUN_3`; client 2 gets `EVALRUN_4`, `EVALRUN_5`, `EVALRUN_6`.
- The same seed allocation is reused for every checkpoint so checkpoint comparisons are fairer.

Examples:

```bash
uv run sts2rl-evaluate --episodes 3 --client-port 15526
uv run sts2rl-evaluate --episodes 5 --seed EVALRUN --csv results/eval.csv
uv run sts2rl-evaluate --client-port 15526 --client-port 15527 --episodes 3
uv run sts2rl-evaluate --checkpoint-dir checkpoints --max-steps 200 --sleep 0
uv run sts2rl-evaluate --no-reset --live --auto-pause
```

## Outputs

- Console summary for each evaluated checkpoint
- Optional CSV from `--csv`
- Optional live dashboard when `--live` or `--live-html` is used
