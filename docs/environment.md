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

`EpisodeRunner` turns that raw boundary into the Agent-facing observation by
pairing the state with `GET /api/v1/player`, the only source of the run-level
master deck. That snapshot is reused for the duration of a battle, because no
mid-battle screen can add, remove, or upgrade a card; off-battle steps refetch
it. A build that does not serve the endpoint warns once and continues without
the deck.

`GameEnv.step` also issues no follow-up read: every action response embeds the
resulting state, and a rejected action returns the unchanged state alongside
its error. Only a response whose state could not be read falls back to a
`GET`.

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

## Two backends, one API

`GameEnv(backend=...)` picks what the client on the other end is:

- `game` (default): a real STS2MCP client. A run starts by navigating its
  menus, which is what `ResetController` exists for.
- `sim`: [STS2Simulator](../../STS2Simulator/README.md), which hosts the game's
  own rules engine with no Godot engine and serves the same API. It has no
  menus, so `SimResetController` starts a run in one `POST /api/v1/sim/reset`
  and gets back the first state the agent can act on.

Everything after reset is the same code: the same dispatcher, the same embedded
state on every response, the same `action_error` on a rejected action. That is
the point of the simulator serving this API rather than a bespoke one -- the
encoder, the action space, and the reward never learn which one they are
talking to.

Two differences are worth knowing:

- **Seeds are required.** The simulator has no menu to leave the choice to and
  no "whatever the game rolls" mode, so `--seed-pool` or `--run-seed` is
  mandatory. In exchange the seed is never ignored: `reused_active_run` is
  always false, because a reset always starts the run it was asked for rather
  than joining one already in progress.
- **The action delay is zero by default.** `action_delay_seconds` paces the
  *next* request while a real client is still resolving the last one. The
  simulator answers only once the game has settled, so there is nothing to
  pace.

`ResetSpec.sim_mode` chooses what an episode plays: `run` follows the seed's map
act by act, `gauntlet` plays hallway fights with no map at all, which is a
combat-only task rather than a shorter run. `sim_max_fights` bounds the
gauntlet. Both are ignored by a real client.
