# ADR-0007: Split agents into action space, policy module, and algorithm

- Status: Accepted
- Date: 2026-07-19
- Related: ADR-0001, ADR-0004; amends ADR-0002 and ADR-0003. Full design
  rationale: [agent-architecture-redesign.md](../agent-architecture-redesign.md)

## Context

A candidate-action agent has four independent concerns: legal-move enumeration
(game rules), representation (features), network architecture, and the learning
algorithm. They were collapsed onto two axes: the *encoder* carried rules +
representation, and `DQNCandidateAgent` / `PPOCandidateAgent` carried
architecture + algorithm (+ checkpointing + rollout plumbing). Two consequences:
the learned `CardModelEncoder` (ADR-0004) had no seam to plug into — `encoders/`
is torch-free by rule and the agents' optimizers covered only the hardcoded
networks — and the algorithm classes owned far more than their names claimed.
There was also a latent circular import: `import sts2rl.encoders` crashed when
imported before `sts2rl.agents` (`battle_encoder → agents.selection →
agents/__init__ → orchestrator → battle_encoder`).

## Decision

Separate the agents into three composable roles (executed as Phases 0–2 of the
design note; Phase 3 — learned featurizers — is deferred):

- **`action_spaces/`** (new package, torch-free; imports only stdlib +
  `sts2rl.data`): one `ActionSpace` per screen owning `candidates()`,
  `action_key()`, and `fallback()`. `agents/selection.py` moved here verbatim,
  fixing the circular import. Rule helpers shared with featurization
  (`action_item`, `action_target_index`, target-type logic, caps) live here as
  module functions; the battle encoder imports them (`encoders → action_spaces`
  is the sanctioned direction).
- **`encoders/`** keep only representation: `encode_state` / `encode_action`,
  dims, and the checkpoint schema strings.
- **`models/policies.py`**: `CandidateQNetwork` and `CandidatePPOPolicy` moved
  here unchanged, exposing a shared differentiable
  `score(state_action: Tensor[K, D]) -> Tensor[K]` contract (BC's `bc_score` and
  all scoring paths route through it). The agents accept `policy=` to swap
  architectures; the optimizer covers `policy.parameters()`, so a future policy
  module embedding `CardModelEncoder` trains end-to-end with no special-casing.
- **`DQNCandidateAgent` / `PPOCandidateAgent`** are now algorithm facades:
  update rule, optimizer, buffers/collectors, checkpoint IO. The DQN target
  network is a `deepcopy` of the composed policy module.
- **Screen agents are registry configurations, not classes**: orchestrator
  `SCREEN_SPECS` (`screen → ScreenSpec(action_space, encoder)`) +
  `create_screen_agent(screen, type, **kwargs)` replace the ten per-screen
  subclasses. The `agents/battle/` shim modules, the `*Battle*` aliases
  (ADR-0002), and the `BattleAgent` base alias are retired; tests import the
  canonical names.

## Checkpoint compatibility

No migration needed: the policy-module classes kept their attribute names
(`net`, `actor`, `critic`) and remain the object saved under
`"model_state_dict"`, so `state_dict` keys, schema strings
(`candidate_action_v3`, `candidate_action_ppo_v3`, `{screen}_{dqn|ppo}_v1`),
and checkpoint paths are byte-identical. Verified by loading fixed-seed
pre-refactor checkpoints and asserting exact Q/logit equality;
`tests/test_policies.py` pins the key sets and schema strings so future drift
fails loudly.

## Consequences

- Learned representations finally have a seam: a policy module owns any learned
  featurizer, and its parameters are automatically optimized (the deferred
  Phase 3 wires `CardModelEncoder` in behind a new schema string).
- The algorithm classes match their names; swapping architectures is a
  constructor argument, not a subclass.
- The public agent surface (`valid_action_candidates`, `encode_*`, `bc_score`,
  `save`/`load`, counters, factories) is unchanged, so `training/`,
  `evaluation/`, and `flow/` needed no edits.
- Behavior was verified unchanged at every phase against a golden capture of
  (state vector, ordered candidate keys, action vectors) over the test fixtures.
- Importers of the retired aliases/modules (`BattleDQNAgent`,
  `agents.battle.*`, `MapDQNAgent`, `SCREEN_AGENTS`, …) must switch to the
  canonical names/factories — a deliberate break confined to tests and docs in
  this repo (ADR-0002 anticipated the eventual removal).
