# Rewards

Reward interfaces live in `sts2rl.env.rewards`.

The default reward model is `ScopedRewardModel`, which separates the scalar used
for battle-agent learning from the scalar used for run-progress learning and
metrics:

- `BattleOutcomeReward`: battle win/loss plus small resource penalties
- `RunProgressReward`: floor, act, and death progress over the whole run
- `ScopedRewardModel`: returns both scopes and their sum

`compute(prev_state, next_state, action)` returns `(total, details)`, where
`details` includes:

- `battle_reward`: reward passed to the battle agent
- `run_reward`: reward passed to non-battle screen agents
- `total`: `battle_reward + run_reward`, used for episode metrics
- `battle_details` and `run_details`: per-scope diagnostics

Current battle reward:

- battle win: `+1.0`
- battle loss: `-1.0`
- HP loss: `-0.02` per player HP lost this step
- potion use/discard: `-0.05`
- gold lost during battle: `-0.005` per gold, applied when the battle resolves
- max HP lost during battle: `-0.10` per max HP, applied when the battle resolves
- enemy damage, enemy kills, and unused energy are diagnostic-only by default

Current run reward:

- floor progress: `+0.02` per floor
- act progress: `+1.0` per act
- game over: `-1.0`

Battle transitions still terminate the battle-agent sub-episode when the reward
details report `result` as `won` or `lost`.
