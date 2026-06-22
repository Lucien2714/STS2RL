# Development

Recommended checks:

```bash
uv sync --dev
uv run python -m compileall src scripts tests
uv run sts2rl-evaluate --help
uv run sts2rl-train --help
uv run pytest
```

When adding a new battle algorithm:

1. Subclass `sts2rl.agents.battle.base.CandidateActionAgent` (reuses the shared
   `BattleStateEncoder` and the encode/candidate surface).
2. Reuse `BattleStateEncoder` in `sts2rl.encoders` rather than re-encoding state.
3. Keep environment stepping in `env`.
4. Keep neural-network `forward()` methods inside the agent's model module.
5. Keep action decisions behind `choose_action()`.
