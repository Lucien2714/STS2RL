# Offline demonstration data

Behavior cloning needs decisions, not runs: one screen, the complete set of
actions that were legal on it, and which of them the human picked. The recorder
writes runs; `sts2rl-bc-clean` turns them into decisions and says exactly what
it dropped and why.

## The whole path

```powershell
# 1. recorded runs -> decisions
uv run sts2rl-bc-clean --records ..\gameplay_records --out data\bc\v1 --verify

# 2. decisions -> cloned actor weights
uv run sts2rl-bc-train --dataset data\bc\v1 --out runs\bc-v1

# 3. cloned weights -> a PPO run that does not start from random
uv run sts2rl-train --run-dir runs\ppo-v1 --init-encoder runs\bc-v1\bc_best.pt
```

Recording itself needs no command: `record_human_actions` in the mod's
`STS2_MCP.conf` writes a record per run while you play.

## Clean a dataset

`--records` takes a directory of `*.json` records or one such file. The output
directory gets two files:

| file | contents |
|---|---|
| `decisions.jsonl.gz` | one JSON object per trainable decision |
| `manifest.json` | dataset version, per-run provenance, and the rejection ledger |

`--verify` reads the dataset back and tokenizes every decision, which is the
only real proof the encoder can accept it. Without it the cleaner never loads
the encoder at all.

`--only-victories` and `--min-floor N` drop whole runs before their steps are
read. Cloning copies the demonstrator, so a run is a quality decision before it
is a quantity one: the early floors of a run that died on floor 6 are ordinary
play, but everything from the mistake that killed it onward demonstrates
losing. Excluded runs are listed in the manifest under `excluded_runs` rather
than dropped silently — "563 decisions" means something different depending on
whether it came from two runs or from twenty with eighteen filtered out.

## What a record gives us

Each recorded step carries the three things one decision needs, and they map
onto existing types without translation:

| record field | maps to |
|---|---|
| `state` | `GameObservation.raw_state` |
| `player_detail` | `GameObservation.player_detail` (the `/player` payload — the only source of the run-level master deck) |
| `action` `{action, args}` | `GameAction(action_type, **args)` |

Records are written with a UTF-8 BOM, so they are read as `utf-8-sig`. Only
`schema_version: 2` is accepted; a newer recorder is refused rather than read
optimistically, because a silently renamed field would produce a dataset that
trains on the wrong thing.

## Every step is accounted for

`kept` plus the ledger always equals the number of recorded steps. A step is
attributed to the **first** reason that fires, and the order is part of the
meaning:

| reason | what it means |
|---|---|
| `terminal_no_action` | the last step of a run, which is a state and not a decision |
| `no_observable_effect` | the next recorded state is identical, so the game refused the action |
| `no_candidates` | `LegalActionProvider` cannot automate the screen at all |
| `unmatched_action` | the human's action is not in the candidate set |
| `forced_single_candidate` | one candidate, whose cross entropy is exactly zero |

Two of these deserve their reasons written down.

**`no_observable_effect` before `unmatched_action`.** Refused-and-unchanged is
the same signature `EpisodeRunner` treats as a stall rather than a transition:
the action was legal, the screen was simply not ready, and learning from it
would teach that resting at a rest site does nothing. It has to come first
because it and `unmatched_action` describe the same rejections from opposite
sides — a potion reward claimed against a full belt is both "refused by the
game" and "not offered, because the provider skips it" — and attributing those
to the action space would overstate what the action space costs. Ordered
correctly on the one run available, `unmatched_action` falls from 29 to 17 and
`params_differ` drops to zero: every surviving mismatch is an action type the
provider withholds on purpose.

**`forced_single_candidate`.** A single-choice distribution has zero cross
entropy and zero gradient, so these are not training data.
`CandidatePPOAgent.choose_action` drops them from the rollout for the same
reason.

### The action space's cost stays visible

The manifest names up to `--max-unmatched-examples` mismatches per run rather
than only counting them. A pooled count says "17 demonstrated actions were
unreachable"; it does not say that seven of those were `proceed` on a reward
screen, which is a documented, deliberate exclusion, versus something nobody
has accounted for. On the bundled run all 17 are deliberate:

| count | screen / action | why it is withheld |
|---|---|---|
| 7 | `rewards` / `proceed` | a reward screen is claimed out, not walked away from |
| 7 | `card_select` / `cancel_selection` | cancelling a selection is a loop, not a choice |
| 2 | `rewards` / `discard_potion` | a potion is only discardable once the belt is full |
| 1 | `hand_select` / `combat_select_card` | the prompt was already full |

## Why the dataset is stored before tokenization

One hard reason and one soft one.

`TokenizedDecision` cannot be pickled at all — `TokenizedState.entities` is a
mappingproxy — so there is no tensor cache to write even if we wanted one.

And `encoder/schema.py` is the single home of every model-visible column, so a
dataset written before tokenization survives a column being added there.
Tokenizing costs about 1.1 ms per decision against about 2.3 ms for one encoder
forward pass, so paying it per epoch is not the bottleneck, and `BCDataset`
tokenizes on demand without caching.

## The labels are re-verified, not trusted

`expert_index` is a *position* in an ordered candidate list. An action space
that gained, lost, or reordered a candidate would move that label onto a
different action, and nothing downstream would notice — the encoder would
happily clone the policy onto the wrong choices.

So the cleaner stores the candidate set as a snapshot, and `BCDataset.load`
re-derives it from every stored `raw_state` with the *current*
`LegalActionProvider` and compares. A mismatch is a `StaleDatasetError` naming
the run and step; re-cleaning takes seconds. This is checked against real data
rather than a fingerprint, which is why it is exact.

`dataset_format_version` covers the shape of a decision line itself, on the
same bargain `CheckpointManager.FORMAT_VERSION` makes.

## A reload breaks the transition chain

`next_step_index` and `steps_to_run_end` are reserved for a future critic and
are not used yet. `next_step_index` is null both at the end of a run *and*
where the next step is one the recorder marked `resumed`: after a reload the
following state is not what this action produced, and a reward read as a
difference between those two states would invent progress that never happened.

Note that a run can be save-scummed — the bundled one reports
`num_reloads: 3` — but repeated states with conflicting actions are usually not
that. `skip_card_reward` puts the card back on the rewards screen *unchanged*,
so entering the card screen, declining, and returning produces a byte-identical
screen with a different action next time. That is real partial observability,
not a label conflict, and the cleaner deliberately does not deduplicate it.

## The holdout is split by run

`BCDataset.split_by_run` never splits by decision. Decisions inside one run
share a deck, a map, and often a fight, so a decision-level split puts near
duplicates of the training data in the holdout and reports a number that
measures nothing.

With too few runs to split there is nothing honest to return, so the decisions
are cut at the tail and the caller gets a warning saying the result is a sanity
check and **not** a measure of generalization.

## Cloning

`sts2rl-bc-train` is one cross entropy over the actor's candidate scores
against the index the human picked. The model does not change; only where the
gradient comes from. A minibatch runs one forward pass per decision and stacks
the losses before a single backward — `GameEncoder` already batches a
decision's whole candidate set, and padding several decisions together is
cross-step batching, which the codebase leaves as open work rather than
inventing here.

### Only the actor is trained

The critic is left untouched, for three reasons that all point the same way.

It predicts in `_ReturnScale`'s normalized space, and an expert's returns sit
at roughly twenty times an early policy's. Hand PPO a critic calibrated to the
expert's divisor and let `_ReturnScale` relearn the divisor from what the new
policy actually earns, and the critic reads its own predictions at the wrong
scale — worse than one that was never saved.

The value target is also orders of magnitude larger than a cross entropy, and
`GameEncoder` is shared, so a value term here would leave the encoder mostly
trained by the critic when the actor is the whole point.

And with few runs the critic would be learning "how far is left on this map",
which does not survive a new map the way "what to play here" does.

The cleaned dataset keeps `next_step_index` and `steps_to_run_end`, so adding
it later is a column and a loss term rather than a re-clean.

### A top-1 number needs its baselines

Candidate sets average about eight actions, so every epoch reports two
baselines beside the accuracy:

| baseline | on the bundled run |
|---|---|
| uniform choice over candidates | 17.6% |
| always take the first candidate | **32.7%** |

The second is high because candidate order is structured — in combat
`play_card` runs in hand order, and the first playable card is often the right
one. An accuracy in the thirties is therefore not evidence of anything.

Accuracy is also reported per `state_type`, because the screens are not one
problem: combat is 65% of the decisions at 8–12 candidates, while `map` and
`event` average barely two.

### The artifact is weights, not a checkpoint

`bc_best.pt` holds the encoder state dict, the vocabulary fingerprint, and the
`EncoderConfig` — and deliberately no optimizer state and no counters. Adam's
state belongs to a different objective and the counters to a run that never
happened. `sts2rl-train --init-encoder` refuses it if either the fingerprint or
the encoder config disagrees, the same two checks a PPO checkpoint makes and
for the same reason: both produce weights that load and then behave as
something they are not. `--init-encoder` cannot be combined with `--resume`.

The saved artifact is the epoch with the **lowest holdout cross entropy** —
not the last, and not the most accurate. Training accuracy keeps climbing after
the holdout has turned, so the final weights would be the most overfitted of
the run. And accuracy is the wrong thing to pick on: on six runs it wandered
within about 1.5 points from epoch 1 onward, so its maximum was noise, while
the loss was lowest at epoch 1 and 8% higher at the accuracy-chosen epoch — a
policy growing more confident without growing more right. PPO *samples* from
this policy, so an overconfident start has too little entropy to explore;
calibration is what transfers, and cross entropy is what measures it.

## Non-combat `use_potion`

Combat used to be the only screen offering `use_potion`, which left
demonstrated actions unreachable. The gate that replaced it is a list of
**observed (potion, screen) pairs**, `NON_COMBAT_POTIONS` in
`agents/action_space.py`, and it only grows from evidence:

| potion | screens | evidence |
|---|---|---|
| `FOUL_POTION` | `shop`, `fake_merchant` | three drinks in one traced run; the fake merchant's fight is documented as started by it |
| `BLOOD_POTION` | `shop`, `rest_site` | one drink each in two traced runs, both accepted (HP 51→66, 32→48) |

A pair is never widened by analogy. Blood Potion on a rewards screen is
plausible and unobserved, and plausible is exactly what the first gate was.

### The first gate, and what it cost

No API field says whether a potion works outside combat: `can_use_in_combat` is
`true` on every potion the game reports, and there is no out-of-combat twin.
The first version inferred it from `target_type` — anything not aimed at an
enemy — on `shop`, `fake_merchant`, `rewards` and `rest_site`.

A live PPO run proved it wrong. A Skill Potion (`target_type: Self`) offered on
a rewards screen was refused with **"Potion 'Skill Potion' can only be used in
combat"**. The refusal consumed nothing and left the screen unchanged, so it
was a loop, not a wasted potion: once PPO had raised that action's
probability, each episode truncated on it, left the run alive, and handed it to
the next episode, which joined under `allow_active_run` and truncated on it
again at step 0. Sixteen of the 38 episodes from 118 to 155 went that way, and
the seeded experiment was measuring reused runs by the time it was stopped.

Two lessons carry beyond potions. An inferred legality rule has to fail by
offering *nothing*, because the cost of a wrong offer is not one bad step but a
loop. And a rising `truncated` count with `reused_run` tracking it is the
signature to watch for — it showed up here within 40 episodes of the first
refusal.

## There is not enough data yet

The bundled record is **one run**: 704 steps, 563 trainable decisions. A
128-dimensional transformer on 563 samples will overfit, and with one run there
is no honest holdout at all — `split_by_run` falls back to a tail split and
says so.

The pipeline is built for many runs and splits by run from the start, but the
data volume has to come from recording more. Until there are a dozen or more
runs, treat BC accuracy as a sanity check on the plumbing rather than as a
measure of anything.

For scale: that 26 MB record compresses to a 257 KB dataset, about 456 bytes
per decision, so a hundred runs is a few tens of megabytes.
