# ADR-0001: Generalize the candidate-action framework to all screens

- Status: Accepted
- Date: 2026-06-22
- Related: ADR-0002, ADR-0003

## Context

The trainable candidate-action machinery (encode a state, enumerate legal action
*candidates*, score each, pick one) existed only for battle. Every non-battle
screen — map, reward, shop, rest, event — was handled by a hand-written
rule-based policy with no ability to learn. We wanted those screens to be
trainable too, reusing the battle machinery rather than reimplementing it per
screen.

## Decision

Generalize the framework along the axis that actually varies between screens —
the **encoder** — and share everything else:

- Extract the screen-agnostic base `CandidateActionAgent`
  (`agents/candidate_agent.py`); it *composes* an encoder and owns device,
  counters, and the encode/candidate delegation. The DQN/PPO algorithms are
  screen-independent and unchanged.
- Add a `CandidateEncoder` base (`encoders/base.py`) and one encoder per screen
  (`map_encoder`, `reward_encoder`, `shop_encoder`, `rest_encoder`,
  `event_encoder`), each implementing `valid_action_candidates`, `encode_state`,
  `encode_action`, `action_key`.
- The orchestrator routes a non-battle screen to its trainable agent **only when
  that screen has legal candidates**, otherwise to the rule-based policy; it
  trains only on transitions the agent actually chose.
- Training shares one model/optimizer per screen across clients
  (`SharedTrainingState`), with per-client rollouts and per-screen checkpoints,
  mirroring the battle agent. A `--screen-agent {DQN,PPO,none}` CLI flag selects
  the implementation.

## Consequences

- Non-battle screens become trainable with zero new algorithm code; rule-based
  policies remain as the always-available fallback.
- `--screen-agent` defaults to `PPO`, so a plain `sts2rl-train` now also trains
  five screen agents (a behavioral change vs. the prior rule-based-only default,
  and it pairs a DQN battle agent with PPO screen agents). Pass `none` to restore
  the old behavior.
- More checkpoints to manage (one latest/backup set per screen).
- The battle encoder (`BattleStateEncoder`) predates this base and keeps its own
  copies of the shared helpers; it is intentionally **not** migrated, to avoid
  perturbing the trained battle action schema. The base extraction therefore only
  half-applies to battle.
