# Development

Install the development environment and run the contract tests:

```bash
uv sync
uv run pytest
```

During the environment-first rebuild:

1. Keep STS2MCP HTTP details inside `sts2rl.env.mcp_client`.
2. Represent commands with `sts2rl.actions.GameAction`.
3. Keep `GameEnv` independent from encoders, rewards, and agents.
4. Add contract tests before introducing a new upper layer.
