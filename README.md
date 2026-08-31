# STS2RL

STS2RL is an environment-first reinforcement-learning project for Slay the
Spire 2 using the STS2MCP HTTP API.

The project is rebuilding its training and evaluation layers. The supported
foundation now includes the STS2MCP client, typed game actions, environment
reset/step behavior, dynamic legal-action candidates, candidate-action PPO,
bundled game data, and the retained state-encoder scaffold.

## Development

Install dependencies and run the current contract tests:

```bash
uv sync
uv run pytest
```

See [Architecture](docs/architecture.md), [Environment](docs/environment.md),
[Agents](docs/agents.md), [Rewards](docs/rewards.md), and the bundled
[STS2MCP API reference](docs/STS2MCP-raw-full.md).
