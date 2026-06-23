# Architecture

STS2RL separates the project into stable layers:

- `env/`: STS2MCP client integration, reset flow, environment stepping, reward boundaries.
- `agents/`: abstract agent interfaces and concrete agents. `TrainableScreenAgent`
  (`agents/base.py`; `BattleAgent` is a back-compat alias) is the trainable-agent
  interface. The candidate-action agents `DQNCandidateAgent` / `PPOCandidateAgent`
  (`agents/candidate_dqn_agent.py` / `candidate_ppo_agent.py`) subclass
  `CandidateActionAgent`, which *composes* a screen encoder; each agent owns only
  its network and training logic. A bare instance defaults to the battle encoder
  and is the battle agent; the `agents/battle/` modules are re-export shims keeping
  the legacy `BattleDQNAgent` / `BattlePPOAgent` names. See
  [ADR-0001](adr/0001-trainable-screen-agents.md) and
  [ADR-0002](adr/0002-screen-neutral-agent-naming.md).
- `encoders/`: **model-free** (no torch) state/action encoding and legal-action
  enumeration. `BattleStateEncoder` for battle, plus one `CandidateEncoder`
  subclass per non-battle screen (map, reward, shop, rest, event). The orchestrator
  routes a screen to its trainable agent when it has legal candidates, else to a
  rule-based policy.
- `models/`: reusable **learned** `nn.Module` components (e.g. `CardModelEncoder`,
  a learned per-card embedding — see [card-embedding.md](card-embedding.md)). Kept
  separate from the model-free `encoders/`.
- `training/`: training CLI, runner, telemetry, dashboard, and episode logs.
- `evaluation/`: checkpoint evaluation, seeded custom runs, dashboard, and CSV output.
- `checkpoints/`: checkpoint paths and discovery helpers.
- `metrics/`: typed result and transition objects.
- `data/`: static game data and lookup maps.

Architecture decisions are recorded in [docs/adr/](adr/README.md).

