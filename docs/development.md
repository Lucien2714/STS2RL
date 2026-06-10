# Development

Recommended checks:

```bash
uv sync --dev
uv run python -m compileall src scripts tests
uv run sts2rl-evaluate --help
uv run sts2rl-train --help
uv run pytest
```

When adding a new algorithm:

1. Implement or reuse an agent interface in `sts2rl.agents.base`.
2. Put algorithm internals under `sts2rl.algorithms`.
3. Keep environment stepping in `env`.
4. Keep neural-network `forward()` methods inside model modules.
5. Keep action decisions behind `choose_action()`.
