# Architecture

STS2RL is being rebuilt from the environment boundary outward. The current
supported layers are:

- `actions/`: typed actions and their dispatch to STS2MCP client methods.
- `env/`: HTTP client integration, reset navigation, and raw environment steps.
- `data/`: bundled static game data and lookup maps.
- `encoder/`: deterministic structured tokenization, entity Transformer,
  conditioned map DAG encoder, and shared state/action encoding.
- `agents/`: dynamic legal actions, the minimal agent contract, end-to-end
  candidate PPO, and the episode runner.

The environment returns a raw game state and the runner pairs each non-terminal
state with full player detail. Encoding and reward calculation remain outside
the environment. PPO consumes the resulting `GameObservation`, scores only its
current structured legal actions, and trains the full encoder without
introducing a fixed global action table.
