# STS2RL

STS2RL is an environment-first reinforcement-learning project for Slay the
Spire 2 using the STS2MCP HTTP API.

The supported foundation includes the STS2MCP client, typed game actions,
environment reset/step behavior, complete observations, deterministic
structured tokenization, full-map DAG encoding, and end-to-end candidate-action
PPO over dynamic legal actions.

## Development

Install dependencies and run the current contract tests:

```bash
uv sync
uv run pytest
```

Start a resumable training run against a local STS2MCP server:

```bash
uv run python scripts/train.py --run-dir runs/ironclad-001 --total-episodes 100
```

Metrics are written to JSONL and TensorBoard. See
[Training](docs/training.md) for checkpoint and resume behavior.

To run the game clients without a window, and to give each client its own save
directory when running several, see [无渲染运行游戏客户端](docs/headless.md):

```powershell
pwsh -File scripts/start_game_clients.ps1 -Count 4 -Headless
```

See [Architecture](docs/architecture.md), [Environment](docs/environment.md),
[Agents](docs/agents.md), [Training](docs/training.md),
[Rewards](docs/rewards.md), [Headless clients](docs/headless.md), and the
bundled [STS2MCP API reference](docs/raw-full.md).
