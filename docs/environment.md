# Environment

`sts2rl.env.game_env.GameEnv` is the main environment wrapper. It owns an
`STS2Client`, exposes `reset()` and `step()`, and uses an `ActionDispatcher` to
turn agent action dictionaries into STS2MCP API calls.

Reset behavior is menu-driven. Standard training starts a normal singleplayer
run. Evaluation can request custom seeded runs and uses the seeded menu flow:

```text
main -> singleplayer -> custom(seed) -> custom_run -> embark
```

`GameEnv.reset()` must never return a menu state as the initial run state. If it
cannot leave the menu flow, it raises an `STS2ClientError`.

