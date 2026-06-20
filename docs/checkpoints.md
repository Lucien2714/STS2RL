# Checkpoints

Checkpoint helpers live in `sts2rl.checkpoints.manager` and
`sts2rl.evaluation.checkpoints`.

Battle checkpoints are partitioned by agent implementation:

- `checkpoints/battleAgent/DQN/battleagent_latest.pt`
- `checkpoints/battleAgent/DQN/battleagent_step_N.pt`
- `checkpoints/battleAgent/PPO/battleagent_latest.pt`
- `checkpoints/battleAgent/PPO/battleagent_step_N.pt`

Evaluation sorts `battleagent_step_N.pt` numerically, then evaluates latest,
then any other `.pt` files in the selected checkpoint directory.
