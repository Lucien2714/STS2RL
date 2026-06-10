# Checkpoints

Checkpoint helpers live in `sts2rl.checkpoints.manager` and
`sts2rl.evaluation.checkpoints`.

Current names are preserved:

- `checkpoints/battle_agent_latest.pt`
- `checkpoints/battle_agent_step_N.pt`

Evaluation sorts `battle_agent_step_N.pt` numerically, then evaluates latest,
then any other `.pt` files.

