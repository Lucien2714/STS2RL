# Architecture

STS2RL separates the project into stable layers:

- `env/`: STS2MCP client integration, reset flow, environment stepping, reward boundaries.
- `agents/`: abstract agent interfaces and the *algorithm* agents.
  `TrainableScreenAgent` (`agents/base.py`) is the trainable-agent interface. The
  candidate-action agents `DQNCandidateAgent` / `PPOCandidateAgent`
  (`agents/candidate_dqn_agent.py` / `candidate_ppo_agent.py`) subclass
  `CandidateActionAgent`, which *composes* an action space + encoder pair; each
  agent owns only its update rule, optimizer, buffers, and checkpoint IO, and
  accepts a swappable policy module. A bare instance defaults to the battle
  components and is the battle agent; screen agents are built from the
  orchestrator's `SCREEN_SPECS` registry. See
  [ADR-0001](adr/0001-trainable-screen-agents.md),
  [ADR-0002](adr/0002-screen-neutral-agent-naming.md), and
  [ADR-0007](adr/0007-action-space-policy-algorithm-split.md).
- `action_spaces/`: **torch-free game rules** — legal-action candidate
  enumeration, stable action keys, and fallbacks, one `ActionSpace` per screen
  (battle, map, reward, shop, rest, event). Imports only stdlib + `sts2rl.data`.
- `encoders/`: **model-free** (no torch) state/action featurization.
  `BattleStateEncoder` for battle, plus one `CandidateEncoder` subclass per
  non-battle screen. The orchestrator routes a screen to its trainable agent when
  it has legal candidates, else to a rule-based policy.
- `models/`: reusable **learned** `nn.Module` components: the candidate-scoring
  policy modules (`models/policies.py` — `CandidateQNetwork`,
  `CandidatePPOPolicy`, shared `score()` contract), the learned-featurizer
  variants (`models/learned_policies.py`), and building blocks such as
  `CardModelEncoder` (a learned per-card embedding, wired in behind
  `--policy learned` — see [card-embedding.md](card-embedding.md)). Kept separate
  from the model-free `encoders/`.
- `flow/`: the one shared implementation of a game step (`flow/step_loop.py` —
  `decide_action` / `apply_action`), used by both training and evaluation, plus
  the predicates in `flow/battle_flow.py` that decide which transitions the agent
  is asked about and trained on.
- `training/`: training CLI, runner, telemetry, dashboard, and episode logs.
- `evaluation/`: checkpoint evaluation, seeded custom runs, dashboard, and CSV output.
- `checkpoints/`: checkpoint paths and discovery helpers.
- `metrics/`: typed result and transition objects.
- `data/`: static game data and lookup maps.

Architecture decisions are recorded in [docs/adr/](adr/README.md).

