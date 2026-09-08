# Rewards

Reward interfaces live in `sts2rl.env.rewards`; every magnitude lives in
`sts2rl.env.constants`.

`RunProgressReward` scores climbing the spire, and nothing else:

- **+1.0** per node entered (`NODE_PROGRESS_REWARD`)
- **+10.0** per boss defeated (`BOSS_VICTORY_REWARD`)
- **-0.01** per step (`STEP_COST`), so standing still is never free

A node is counted as an increase in `run.floor` between the two states. A boss
victory spans many transitions, so a pending flag is raised while the boss is
alive and consumed exactly once when it is not — a win cannot be paid twice,
and dying to the boss clears the flag rather than paying it.

## What is deliberately not scored

Enemy damage, kills, gold, potions, unspent energy, and HP were all weighted by
hand in earlier models. They are means, not ends, and tuning them is how a
reward model stops matching the objective.

HP is the instructive one, because it looked defensible and was not. Scoring it
paid for healing immediately, so the agent rested at every rest site rather
than upgrading a card — upgrading pays nothing this step. HP does matter:
running out ends the run and with it all further progress. But it matters
*terminally*, and pricing a terminal consideration per step is precisely how
reward hacking starts.

A node cleared at 1 HP now scores exactly what one cleared untouched scores;
the difference shows up as a shorter run. `tests/test_rewards.py` pins that
equality so the term cannot creep back.

## Scale

Everything is expressed relative to one node, so the block scales by changing
`NODE_PROGRESS_REWARD` and keeping the ratios. One node must stay worth far
more than a step, or the step charge dominates; `tests/test_rewards.py` asserts
that ratio too.

Returns are monotone in progress — roughly 26 for an act, 80 for a full win —
which is what makes them tractable for the critic. An earlier hand-weighted
model produced episode returns to +131 with `value_loss` running 69–1580
against a `policy_loss` of ±0.5. Measured over 200 episodes the current model
gives `value_loss` 0.09–0.79 and a policy that climbs from floor 3.6 to 10.8.

Because returns still grow with the policy, the critic predicts a scaled
return rather than the raw one. See the `_ReturnScale` discussion in
[Training](training.md) and in CLAUDE.md.

## Boundary

The boundary is `RewardModel.compute(prev_state, next_state, action)`.
`GameEnv` does not invoke reward code. `EpisodeRunner` combines raw environment
transitions with a `RewardModel` and passes the resulting reward to the agent.

A rejected action changed nothing in the game, so
`RewardModel.action_error_reward` scores it zero rather than running `compute`
over two identical states.

Each client gets its own `RunProgressReward`, because the model tracks a
pending boss per run.

Rewards from forced steps — states with exactly one legal candidate, which the
agent executes without recording a rollout entry — are folded into the
preceding recorded decision by `CandidatePPOAgent`, so no reward is lost.
