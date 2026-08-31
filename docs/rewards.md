# Rewards

Reward interfaces live in `sts2rl.env.rewards`.

The current reward behavior is preserved from the prototype:

- floor progress reward
- HP loss penalty
- battle win reward
- battle loss penalty
- enemy damage and kill rewards
- potion use penalty
- gold and max HP loss penalties
- end-turn unused energy penalty

The boundary is `RewardModel.compute(prev_state, next_state, action)`.
`GameEnv` does not invoke reward code. `EpisodeRunner` combines raw environment
transitions with a `RewardModel` and passes the resulting reward to the agent;
feature encoding remains an agent-side dependency.

