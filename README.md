# STS2RL

STS2RL is an environment-first reinforcement-learning project for Slay the
Spire 2 using the STS2MCP HTTP API.

The project is currently rebuilding its agent, training, and evaluation layers.
The supported foundation is the STS2MCP client, typed game actions, environment
reset/step behavior, bundled game data, and state-encoder scaffold.

## Development

Install dependencies and run the current contract tests:

```bash
uv sync
uv run pytest
```

See [Architecture](docs/architecture.md), [Environment](docs/environment.md),
[Rewards](docs/rewards.md), and the bundled
[STS2MCP API reference](docs/STS2MCP-raw-full.md).
