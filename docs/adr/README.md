# Architecture Decision Records

This directory records notable architecture/design decisions for STS2RL. Each ADR
captures one decision: the context that forced it, what we chose, and the
consequences (good and bad). ADRs are immutable once accepted — if a later
decision overrides one, add a new ADR and mark the old one *Superseded by ADR-N*.

Format: a trimmed [Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
template — **Status / Context / Decision / Consequences** (plus *Alternatives*
where they matter).

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-trainable-screen-agents.md) | Generalize the candidate-action framework to all screens | Accepted |
| [0002](0002-screen-neutral-agent-naming.md) | Screen-neutral agent naming with backward-compatible battle shims | Accepted |
| [0003](0003-candidate-encoder-abc.md) | Enforce the screen-encoder contract with an ABC | Accepted |
| [0004](0004-learned-card-embedding.md) | Learned per-card embedding (`CardModelEncoder`) | Accepted |
| [0005](0005-adopt-ruff.md) | Adopt ruff for formatting and linting | Accepted |
| [0006](0006-split-battle-and-run-rewards.md) | Split battle and run reward scopes | Accepted |
| [0007](0007-action-space-policy-algorithm-split.md) | Split agents into action space, policy module, and algorithm | Accepted |
