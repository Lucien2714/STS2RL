# Architecture

STS2RL is being rebuilt from the environment boundary outward. The current
supported layers are:

- `actions/`: typed actions and their dispatch to STS2MCP client methods.
- `env/`: HTTP client integration, reset navigation, and raw environment steps.
- `data/`: bundled static game data and lookup maps.
- `encoder/`: the state-encoder interface retained for the next phase.
- `agents/`: dynamic legal actions, the minimal agent contract, candidate PPO,
  and the episode runner.

The environment returns raw game state. Encoding and reward calculation remain
outside it. The first agent layer consumes that contract without introducing a
fixed global action table.
