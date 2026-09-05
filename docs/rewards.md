# Rewards

Reward interfaces live in `sts2rl.env.rewards`; every magnitude lives in
`sts2rl.env.constants`.

`BattleProgressReward` scores two kinds of transition:

**In battle** (`monster`, `elite`, `boss`, `hand_select`):

- enemy HP removed, and a bonus per enemy killed
- player HP lost this step
- battle win reward or loss penalty, paid once per battle
- gold and max-HP lost over the whole battle, charged when it resolves
- potion use cost
- end-turn penalty proportional to unspent energy

**Outside battle**:

- floor progress
- HP gained or lost
- a game-over penalty

## Scale

The magnitudes are deliberately kept within roughly one order of magnitude of
each other. PPO shares one encoder trunk between the policy and value heads, so
an outcome term far larger than the per-step terms makes the value loss
dominate the shared gradient and wash out the policy signal. When retuning,
scale the whole block rather than a single entry.
`tests/test_rewards.py` asserts both the individual terms and that
relationship.

## Boundary

The boundary is `RewardModel.compute(prev_state, next_state, action)`.
`GameEnv` does not invoke reward code. `EpisodeRunner` combines raw environment
transitions with a `RewardModel` and passes the resulting reward to the agent.

A battle resolves exactly once: after a win or loss is scored, subsequent
transitions still sitting on the reward screen return zero with
`already_resolved` set, so one victory cannot be paid twice.

Rewards from forced steps — states with exactly one legal candidate, which the
agent executes without recording a rollout entry — are folded into the
preceding recorded decision by `CandidatePPOAgent`, so no reward is lost.
