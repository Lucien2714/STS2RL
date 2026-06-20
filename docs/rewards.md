# Rewards

Reward interfaces live in `sts2rl.env.rewards`.

The current battle reward is a conservative shaped signal:

- floor progress reward
- battle win reward: `+100`
- battle loss penalty: `-100`
- enemy damage reward: `+0.2` per HP lost
- enemy kill reward: `+5`
- HP loss penalty: `-2` per player HP lost
- potion use penalty: `-1`
- gold and max HP loss penalties
- end-turn unused energy penalty: `-0.2` per unused energy

The win/loss outcome is intentionally the main signal. Damage, kills, unused
energy, and potion use are smaller nudges so the policy does not overfit to
short-term damage or spending every point of energy.

The long-term boundary is `RewardModel.compute(prev_state, next_state, action)`.
During this migration, the stateful battle bookkeeping remains in `GameEnv`.
