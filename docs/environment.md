# Environment

`sts2rl.env.GameEnv` is the raw STS2MCP environment boundary. It owns or accepts
an `STS2Client`, navigates reset menus, dispatches typed `GameAction` objects,
and validates the state returned by the backend.

```python
from sts2rl.actions import GameAction
from sts2rl.env import GameEnv, ResetSpec

with GameEnv(base_url="http://localhost:15526/api/v1") as env:
    state = env.reset(ResetSpec(game_mode="custom", run_seed="ABC"))
    result = env.step(GameAction("end_turn"))
    next_state = result.raw_state
```

The environment deliberately does not encode state or calculate reward. Those
layers consume `RawState` after the raw transition boundary is stable.

`GameEnv.step()` accepts only `GameAction` and returns an immutable `EnvStep`:

- `raw_state`: the next validated STS2MCP state.
- `done`: whether the backend reached `game_over`.
- `info`: API response metadata or a structured client action error.

Invalid actions and dispatcher programming errors raise immediately. An
`STS2ClientError` raised while executing a valid action is returned as
`info["action_error"] = True` when the current state can still be fetched.
