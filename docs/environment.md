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

`EpisodeRunner` turns that raw boundary into the Agent-facing observation:

- it fetches player detail once after reset;
- it fetches it once for every nonterminal next state;
- a state refresh refreshes raw state and player detail together;
- terminal observations use `player_detail=None` and make no detail request;
- an `STS2ClientError` while loading required detail becomes an explicit
  `ObservationError` instead of silently training without the permanent deck.

Reward models continue to receive raw previous/next states. `EpisodeResult`
also keeps raw initial/final states, while each `Transition` stores exactly the
`GameObservation` passed to the Agent.

`GameEnv.step()` accepts only `GameAction` and returns an immutable `EnvStep`:

- `raw_state`: the next validated STS2MCP state.
- `done`: whether the backend reached `game_over`.
- `info`: API response metadata or a structured client action error.

Invalid actions and dispatcher programming errors raise immediately. An
`STS2ClientError` raised while executing a valid action is returned as
`info["action_error"] = True` when the current state can still be fetched.

Reset menu navigation lives in `ResetController`. Every menu transition is a
`MenuSelectAction` sent through the same dispatcher as normal environment
actions. Calling `reset()` while a run is already active raises by default;
use `ResetSpec(allow_active_run=True)` only when reusing that run is intended.
