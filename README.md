# STS2RL

STS2RL is a reinforcement-learning project for Slay the Spire 2 using the
STS2MCP HTTP API. It is a package-oriented rewrite of the earlier prototype,
with explicit environment, agent, training, and evaluation boundaries.

## Quick Start

Install dependencies with uv from this directory:

```bash
uv sync
```

Run training:

```bash
uv run sts2rl-train
```

Optionally pretrain the battle agent on recorded human play before training
(behavioral cloning), then let training resume from the pretrained checkpoint:

```bash
uv run sts2rl-pretrain --recordings /path/to/recordings --battle-agent DQN
```

Evaluate checkpoints:

```bash
uv run sts2rl-evaluate --episodes 3 --client-port 15526
```

Evaluate with multiple clients and deterministic seed allocation:

```bash
uv run sts2rl-evaluate --episodes 3 --client-port 15526 --client-port 15527 --seed EVALRUN
```

## Command Parameters

Training most often needs only the STS2MCP client location:

| Parameter | Default | Description |
|---|---:|---|
| `--client-port`, `--mcp-port` | `15526` | STS2MCP API port. Repeat once per running game client. |
| `--client-host` | `localhost` | Host used with `--client-port`. |
| `--base-url` | built from host/port | Full API URL, such as `http://localhost:15526/api/v1`. Repeat once per client. |
| `--live` | off | Serve the live training dashboard. |
| `--episode-log` | `logs/training_episodes.jsonl` | JSONL output for completed episode summaries. |

Evaluation adds checkpoint, seed, and episode controls:

| Parameter | Default | Description |
|---|---:|---|
| `--checkpoint-dir` | `checkpoints` | Directory containing `.pt` checkpoints. |
| `--episodes` | `3` | Episodes to run per checkpoint per client. |
| `--game-mode` | `custom` | Run mode: `standard`, `daily`, or `custom`. |
| `--seed` | random | Base seed for deterministic evaluation seed allocation. |
| `--max-steps` | `0` | Episode step limit. `0` means no limit. |
| `--csv` | none | Optional CSV output path for evaluation summaries. |
| `--reset` / `--no-reset` | `--reset` | Start each episode from a fresh run, or use the current backend state. |

See [Training](docs/training.md) and [Evaluation](docs/evaluation.md) for the full parameter lists.

## Documentation

- [Architecture](docs/architecture.md)
- [Environment](docs/environment.md)
- [Agents](docs/agents.md)
- [Training](docs/training.md)
- [Pretraining (Behavioral Cloning)](docs/pretraining.md)
- [Evaluation](docs/evaluation.md)
- [Seeds](docs/seeds.md)
- [Rewards](docs/rewards.md)
- [Checkpoints](docs/checkpoints.md)
- [Development](docs/development.md)
