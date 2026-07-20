# ADR-0002: Screen-neutral agent naming with backward-compatible battle shims

- Status: Accepted (amended by ADR-0007: the transitional `agents/battle/` shim
  modules and `*Battle*` aliases have since been retired; the canonical
  `DQNCandidateAgent` / `PPOCandidateAgent` names remain)
- Date: 2026-06-22
- Related: ADR-0001

## Context

After ADR-0001 the same classes drive every screen, but they were still named for
battle: `BattleDQNAgent` / `BattlePPOAgent` (networks `BattleQNetwork` /
`BattlePPOPolicy`), reused for non-battle screens through `*CandidateAgent`
aliases, and the trainable base interface was `BattleAgent`. The canonical names
*lied* — a "Battle" class was driving the shop screen — which is a real source of
confusion for a generic framework.

## Decision

Make the screen-neutral names canonical and keep every battle name as a
backward-compatible alias:

- Concrete agents move to `agents/candidate_dqn_agent.py` /
  `agents/candidate_ppo_agent.py` as `DQNCandidateAgent` / `PPOCandidateAgent`,
  with networks `CandidateQNetwork` / `CandidatePPOPolicy`. The old
  `agents/battle/dqn_agent.py` / `ppo_agent.py` become **thin re-export shims**
  (the pattern `agents/battle/base.py` already used), preserving
  `BattleDQNAgent` / `BattlePPOAgent` / `DQNBattleAgent` / `PPOBattleAgent`.
- The base interface `BattleAgent` (`agents/base.py`) is renamed
  `TrainableScreenAgent`, with `BattleAgent = TrainableScreenAgent` kept as an
  alias.

## Consequences

- Names now match reality; the framework reads as screen-agnostic.
- Fully backward compatible: every prior import path still resolves through the
  aliases/shims, so no call sites, tests, or external scripts break.
- **Checkpoints are unaffected**: the action schema string comes from the encoder,
  and `state_dict` keys are derived from module *attributes*, not class names, so
  renaming classes/networks cannot invalidate existing `.pt` files.
- The `battle/` shim modules are transitional cruft; they can be removed in a
  future breaking release once nothing imports the battle paths.
