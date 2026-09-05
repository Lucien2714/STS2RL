# Development

Install the development environment and run the contract tests:

```bash
uv sync
uv run pytest
uv run ruff check .
```

During the environment-first rebuild:

1. Keep STS2MCP HTTP details inside `sts2rl.env.mcp_client`.
2. Represent commands with `sts2rl.actions.GameAction`.
3. Keep `GameEnv` independent from encoders, rewards, and agents.
4. Add contract tests before introducing a new upper layer.
5. Keep training artifacts under a dedicated run directory; never commit them.

Validate external input at the layer that owns it: HTTP responses in the client
and environment, raw game fields in the tokenizer/action provider, and persisted
configuration in the training runtime. Internal code should trust typed objects
and avoid repeating type checks or wrapping ordinary lookup errors. Keep checks
that prevent silent errors, such as tensor/mask shape mismatches, invalid graph
references, incompatible checkpoints, and incorrect PPO action sequencing.
Verify model output shapes in tests rather than rechecking them on every forward
pass.
