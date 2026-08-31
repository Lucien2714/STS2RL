# Architecture

STS2RL is being rebuilt from the environment boundary outward. The current
supported layers are:

- `actions/`: typed actions and their dispatch to STS2MCP client methods.
- `env/`: HTTP client integration, reset navigation, and raw environment steps.
- `data/`: bundled static game data and lookup maps.
- `encoder/`: the state-encoder interface retained for the next phase.

The environment returns raw game state. Encoding, reward calculation, agents,
training, and evaluation are separate layers and will be reintroduced only after
the environment contract is stable.
