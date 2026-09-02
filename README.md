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

See [Architecture](docs/architecture.md), [Environment](docs/environment.md),
[Agents](docs/agents.md), [Rewards](docs/rewards.md), and the bundled
[STS2MCP API reference](docs/STS2MCP-raw-full.md).
