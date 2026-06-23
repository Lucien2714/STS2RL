# ADR-0006: Split battle and run reward scopes

- Status: Accepted
- Date: 2026-06-23

## Context

The original `BattleProgressReward` mixed battle progress, resource changes, floor
progress, and death penalties into one per-transition scalar. That made the reward
easy to log, but awkward for learning:

- The battle agent needs short-horizon feedback for winning fights while preserving
  HP and consumables.
- Non-battle screen agents need long-horizon run-progress feedback for choices such
  as map nodes, rewards, shops, rests, and events.
- A single dense reward over-incentivized local combat heuristics such as damage,
  kills, and unused-energy spending, while giving weak credit to run-level choices.

Battle transitions were already treated as sub-episodes by ending the battle-agent
bootstrap when a fight is won or lost, so the reward boundary should match that
training boundary.

## Decision

Replace the single reward model with scoped reward components:

- `BattleOutcomeReward`: battle-only terminal/resource reward.
- `RunProgressReward`: whole-run floor, act, and death reward.
- `ScopedRewardModel`: coordinator that returns `(total, details)` where
  `details["battle_reward"]` trains battle-controlled states,
  `details["run_reward"]` trains non-battle screen states, and `details["total"]`
  remains the metric/logging reward.

Keep `BattleProgressReward` as a backward-compatible alias for the scoped model.

The chosen reward scales are:

- Battle win/loss: `+1.0` / `-1.0`.
- Battle HP loss: `-0.02` per player HP lost on that step.
- Potion use/discard: `-0.05`.
- Battle gold loss on resolution: `-0.005` per gold.
- Battle max HP loss on resolution: `-0.10` per max HP.
- Run floor progress: `+0.02` per floor.
- Run act progress: `+1.0` per act delta.
- Game over: `-1.0`.

Enemy damage, enemy kills, and unused energy remain diagnostic fields but no longer
contribute to the training reward by default.

## Consequences

- Battle learning now receives a small, normalized fight outcome/resource signal
  instead of a large dense combat heuristic.
- Screen-agent learning receives run-progress reward instead of battle-local reward.
- Episode and evaluation metrics can report total reward, battle reward, and run
  reward separately.
- Existing checkpoint tensor shapes remain compatible, but reward-scale
  comparability with older checkpoints is intentionally broken.
- Future reward experiments can tune battle and run scopes independently without
  changing environment stepping or action dispatch.
