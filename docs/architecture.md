# Architecture

STS2RL is built from the environment boundary outward. The layers are:

- `actions/`: typed actions and their dispatch to STS2MCP client methods.
- `env/`: HTTP client integration, reset navigation, raw environment steps, and
  the reward model.
- `data/`: bundled static game data.
- `encoder/`: the shared column schema, deterministic structured tokenization,
  the entity Transformer, the conditioned map DAG encoder, and the composed
  state/action encoder.
- `agents/`: dynamic legal actions, the minimal agent contract, end-to-end
  candidate PPO, and the episode runner.
- `training/`: multi-episode orchestration, JSONL and TensorBoard metrics, and
  versioned atomic checkpoints.

The environment returns a raw game state and the runner pairs it with player
detail when the mod serves it, reusing one snapshot for the duration of a
battle since the deck cannot change mid-battle. Detail is optional: builds
without the endpoint simply train without the master deck. Encoding and reward
calculation stay outside the environment. PPO consumes the resulting `GameObservation`, scores only its
current structured legal actions, and trains the full encoder without a fixed
global action table.

`encoder/schema.py` is the one place that defines which columns the model sees
and which vocabulary each indexes. The tokenizer and the trainable encoders
both import it and neither owns it, so the two cannot drift apart.

The training runtime owns process-level concerns: experiment counters,
checkpoint compatibility, logging, interruption recovery, and CLI construction.
It does not move those responsibilities into `GameEnv` or the model encoder.
`TrainingState` is the single record of every counter; the agent holds the live
values during training and the trainer copies them at episode boundaries.

## Known gaps

- Encoding is batched within a decision but not across rollout steps, so a PPO
  update still runs one encoder forward per step. Padding whole steps together
  is the remaining throughput work, and it is also what would make a GPU worth
  using — the tensors are currently too small for one.
- There is no evaluation CLI and no behavioral-cloning pretraining.
