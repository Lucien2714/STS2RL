# Architecture

STS2RL separates the project into stable layers:

- `env/`: STS2MCP client integration, reset flow, environment stepping, reward boundaries.
- `agents/`: abstract agent interfaces and concrete screen agents.
- `algorithms/`: reusable algorithm internals such as DQN components.
- `training/`: training CLI, runner, telemetry, dashboard, and episode logs.
- `evaluation/`: checkpoint evaluation, seeded custom runs, dashboard, and CSV output.
- `checkpoints/`: checkpoint paths and discovery helpers.
- `metrics/`: typed result and transition objects.
- `data/`: static game data and lookup maps.

The current migration preserves the original runtime behavior while moving code
under the `sts2rl` package and adding explicit extension points for future
algorithms and reward models.

