# Design note: separating representation, model, and algorithm in the agents

- Status: **Accepted** — ratified as
  [ADR-0007](adr/0007-action-space-policy-algorithm-split.md); Phases 0–2 are
  implemented, Phase 3 (learned featurizers) is a deferred follow-up
- Date: 2026-07-15 (proposed) / 2026-07-19 (accepted)
- Related: ADR-0001 (candidate-action framework), ADR-0003 (encoder ABC),
  ADR-0004 (learned card embedding), [card-embedding.md](card-embedding.md)

This note motivated the restructuring of the trainable agents; ADR-0007 records
the decision as implemented (the `action_spaces/` package name and the retired
compat shims were settled during review).

## 1. The problem

A candidate-action agent has **four** independent concerns:

1. **Legal-move enumeration** — game rules: which actions are valid now, and a
   stable key per action. Pure logic, no learning, no torch.
2. **Representation** — turning raw state + a candidate action into features.
3. **Network architecture** — the `nn.Module` mapping features → scores/value.
4. **Learning algorithm** — the update rule (DQN replay / PPO clip) + optimizer.

Today these collapse onto **two** axes:

| Concern | Where it lives today |
|---|---|
| 1 Legal moves | inside the *encoder* — `valid_action_candidates`, `action_key`, `_fallback_action`, `_public_action` |
| 2 Representation | inside the *encoder* — `encode_state`, `encode_action`, `state_size`/`action_feature_size` |
| 3 Architecture | **hardcoded** inside the *agent* — `CandidateQNetwork` / `CandidatePPOPolicy`, built in `__init__` |
| 4 Algorithm | inside the *agent* — `DQNCandidateAgent` / `PPOCandidateAgent` |

Two things fall out of this that we want to fix (the two concerns that motivated
this note):

**(A) Learned models have nowhere to plug in.** `CardModelEncoder` (ADR-0004) is a
learned `nn.Module` stranded in `models/` that *nothing imports*. It cannot live in
`encoders/` (torch-free by rule), and it cannot be trained even if we called it,
because the agent's optimizer only covers `self.model.parameters()`
(`candidate_dqn_agent.py`, `candidate_ppo_agent.py`). ADR-0004's own Consequences
section names this: wiring it in is "an action-schema bump that … touches the
encoder and both networks." That friction is the abstraction under strain.

**(B) The algorithm class owns too much.** `DQNCandidateAgent` is named after one
concern (the update rule) but also owns the network architecture, the
candidate-scoring paradigm, tensorization, checkpoint IO, and rollout plumbing.
The one thing allowed to vary — the encoder — is the one that can carry no learned
parameters. Architecture, which is at least as worth varying as features, is the
axis that is *not* a first-class object.

Non-goals of this note: changing the candidate-action paradigm itself (it is a good
fit for STS's variable action space), changing reward shaping, or changing the
multi-client concurrency model.

## 2. Target architecture

Three top-level roles, cleanly separable:

```
ActionSpace     (rules, torch-free)   enumerate(state) -> [candidate];  action_key(action)
PolicyModule    (nn.Module)           score(state, candidates) -> Tensor[K]   # Q or logits
                                      value(state) -> Tensor[]                # PPO only
Algorithm       (update rule)         owns optimizer(policy.parameters()), buffers, checkpoint IO
```

- **ActionSpace** is the model-free half of today's encoder, lifted out. It is
  identical across DQN/PPO and across hand-crafted vs learned representations, and
  it belongs next to `actions`/`env`, not the model.

- **PolicyModule** is the swappable, *trainable* brain. It is a single `nn.Module`
  that internally composes a **featurizer** and a **scoring net**:
  - `FlatConcatPolicy(HandcraftedFeaturizer, MLP)` reproduces today exactly
    (featurizer has no parameters; only the MLP does).
  - `LearnedPolicy(CardModelEncoder + pooling, …)` is now a drop-in: because it is
    one `nn.Module`, *all* its parameters are returned by `policy.parameters()`.

- **Algorithm** (`DQN`, `PPO`) wraps a PolicyModule and owns **only** the update
  rule, the optimizer over `policy.parameters()`, the replay/rollout buffers, the
  exploration/GAE bookkeeping, and checkpoint IO. It knows nothing about screens
  or architectures. DQN's target network is a *copy of the PolicyModule* — a target
  net is an algorithm concern, which is another sign this seam is the right one.

An **agent** is then just a binding `(ActionSpace, PolicyModule, Algorithm)` created
from a registry entry — not a hand-written subclass.

### How this fixes the two concerns

- **(A) Learned models plug in and train end-to-end.** A learned featurizer is part
  of the PolicyModule, so its parameters are already in the optimizer. Behavioral
  cloning and RL both flow gradients into the embedding with no special-casing. The
  orphan `models/CardModelEncoder` gets a home.
- **(B) The algorithm owns only the algorithm.** Architecture and representation
  move into PolicyModule; rules move into ActionSpace. `DQN`/`PPO` become small.

### Free win (not a primary goal)

The per-screen 5×2 class product (`MapDQNAgent`, `MapPPOAgent`, … + battle aliases)
collapses into registry entries: `map = (MapActionSpace, map_policy_factory, DQN|PPO)`.
The ten near-empty subclasses can become thin compat shims or be deleted per the
ADR-0002 backward-compat stance.

## 3. Interface sketch (illustrative, not final)

```python
class ActionSpace(Protocol):            # torch-free
    def candidates(self, raw_state) -> list[dict]: ...   # each has 'action' + 'action_key'
    def action_key(self, action, raw_state=None) -> str: ...
    def fallback(self, raw_state) -> dict: ...

class PolicyModule(nn.Module):          # the swappable, trainable brain
    def score(self, raw_state, candidates) -> torch.Tensor: ...   # [K]  differentiable
    def value(self, raw_state) -> torch.Tensor: ...               # []   (ActorCritic only)
    # parameters() spans featurizer + net -> the optimizer sees everything

class Algorithm(ABC):                   # DQN / PPO
    def __init__(self, action_space, policy: PolicyModule, **hp):
        self.optimizer = Adam(self.policy.parameters(), ...)
    def choose_action(self, raw_state, *, training): ...
    def observe(...); def train_step(...) -> float | None
    def new_rollout(self); def buffered_steps(self)
    def save(path); def load(path)
```

- BC unifies to one loss in the trainer: `softmax(policy.score(state, candidates))`
  with cross-entropy to the human-chosen candidate. Today's per-agent `bc_score`
  becomes just `PolicyModule.score`; DQN (score = Q) and PPO (score = logits) and
  learned featurizers all satisfy it identically.
- DQN needs only `score`; PPO additionally needs `value`. Express as two protocols
  (`ScorePolicy`, `ActorCriticPolicy`); the algorithm declares which it requires.

## 4. Compatibility with existing consumers

These call sites must keep working (verified in `training/`, `evaluation/`):

| Caller | Uses | Plan |
|---|---|---|
| `pretrain.py` | `encode_state`, `encode_action`, `valid_action_candidates`, `bc_score`, `model_input_size` | `encode_*` move to the featurizer; BC calls `policy.score`. Keep thin delegating shims on the agent for one release. |
| `orchestrator.Agent` | `new_rollout`, `remember`, `train_step`, `encode_*`, `candidate_action_vectors`, `buffered_steps`, `epsilon`, `learn_steps` | Provided by `Algorithm` (buffers/counters) + `ActionSpace`/`PolicyModule` (encode/candidates), surfaced through the agent facade. |
| `telemetry.py`, `evaluation/runner.py` | `current_q_values` | Stays on the agent facade; computed from `policy.score` (+ softmax for PPO). |
| `orchestrator` factories | `create_battle_agent`, `create_screen_agent(s)` | Same signatures; build from the registry instead of the subclass map. |

## 5. Checkpoints

Checkpoint compatibility is the sharp edge; loads are gated by the schema strings
`candidate_action_v3` / `candidate_action_ppo_v3`.

- **Structural refactor (Phases 0–2) is checkpoint-preserving.** The hand-crafted
  featurizer + the *same* MLP produce an identical tensor graph. The only risk is
  `state_dict` **key prefixes** shifting when the MLP is nested under the
  PolicyModule (e.g. `net.0.weight` → `net.net.0.weight`).
  *Ratified in implementation (ADR-0007): no key remap was needed at all* — the
  policy modules moved to `models/policies.py` as-is (same class, same `net` /
  `actor` / `critic` attributes) and remain the object saved under
  `"model_state_dict"`, so keys are byte-identical. `tests/test_policies.py` pins
  the key sets so future changes fail loudly.
- **Learned featurizers (Phase 3) are a schema bump.** New schema string (e.g.
  `candidate_action_learned_v1`); old checkpoints are retrained. ADR-0004 already
  anticipated exactly this, so it is an accepted, isolated cost — and now opt-in
  per agent config instead of forced on everyone.

## 6. Migration phases (each independently shippable & green)

- **Phase 0 — split rules out of the encoder.** Extract `ActionSpace` from
  `CandidateEncoder` / `BattleStateEncoder`; the encoder keeps only representation.
  Behavior-identical, no tensors touched, checkpoint-safe. Fixes the
  rules-vs-representation fusion on its own.
- **Phase 1 — introduce PolicyModule + shrink the algorithm.** Move the networks +
  tensorization into `FlatConcatPolicy`; `DQNCandidateAgent`/`PPOCandidateAgent`
  become `DQN`/`PPO` algorithms taking a PolicyModule. Optimizer over
  `policy.parameters()` (empty featurizer params ⇒ identical to today). Add the
  checkpoint key-remap. Preserves behavior *and* checkpoints.
- **Phase 2 — registry replaces the 5×2 subclasses.** `create_*` build from a
  config table; delete/shim the ten subclasses and battle aliases.
- **Phase 3 — learned featurizers become first-class (the payoff).** Wrap
  `CardModelEncoder` in a `LearnedFeaturizer`; drop it into a PolicyModule; BC + RL
  train it end-to-end. New schema, opt-in per agent.

Phases 0–2 are pure refactors that leave the models numerically unchanged; only
Phase 3 changes what is learned.

## 7. Alternatives considered

- **Leave it; wire `CardModelEncoder` in ad hoc.** Rejected: reproduces the
  "algorithm owns everything" coupling and repeats the encoder/both-networks edit
  ADR-0004 flagged, once per learned component.
- **Make the network a constructor arg only (compose the net, keep the encoder as
  is).** Half-measure: it makes architecture swappable but leaves representation
  torch-free, so a *learned* featurizer still cannot join the optimizer. Does not
  fix concern (A).
- **One mega-agent class with strategy flags.** Rejected: moves the coupling into
  conditionals; the whole point is to separate the axes into composable objects.

## 8. Open questions (resolved at ratification)

- Flat vs structured PolicyModule input: Phases 0–2 keep the flat
  `[state ⧺ action]` tensor contract (`score(state_action)`), matching today's
  numerics. Structured per-entity input remains open for Phase 3, where the
  featurizer joins the policy module anyway.
- `ActionSpace` home: a new `action_spaces/` package (torch-free; imports only
  stdlib + `sts2rl.data`), which also absorbed `agents/selection.py` and fixed
  the `encoders → agents` circular-import edge.
- Compat shims: retired immediately (`agents/battle/`, `*Battle*` aliases,
  `BattleAgent`, the ten per-screen subclasses, `SCREEN_AGENTS`); tests updated
  to the canonical names.
