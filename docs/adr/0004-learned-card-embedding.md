# ADR-0004: Learned per-card embedding (`CardModelEncoder`)

- Status: Accepted
- Date: 2026-06-22
- Related: [Card embedding design note](../card-embedding.md)

## Context

Cards are currently encoded crudely in `BattleStateEncoder`: a scaled ordinal
index scalar (`get_card_index(...) / map_size`) per hand card and count vectors
over `card_id × upgraded` for piles. Feeding the card index as a magnitude
implies a false ordering over the 576 card ids (card #500 is not "more" than card
#1). Gerigk & Engels, *Learning Various Strategies for Dominion with Deep RL*
(AIIDE 2023), instead represent each card with a learned low-dimensional
embedding pooled as a multiset. We want the same: a learned vector per card.

## Decision

Add a standalone `CardModelEncoder` (`nn.Module`, `models/card_encoder.py`) that
maps a card to a learned 16-d vector from **only** its id, `is_upgraded`, and
enchantment, using *factored* embeddings:

- `nn.Embedding` tables for the **categorical** factors — card id (16-d) and
  enchantment (8-d). The table rows *are* the learned embedding; the `DataIdMap`
  JSON lookup only supplies the integer row index. Row 0 of each table is reserved
  for unknown/none.
- `is_upgraded` is **binary**, so it is fed as a raw 0/1 scalar (no embedding).
- The factors are concatenated and passed through `Linear(id+ench+1 → 16)`.
- It lives in a new `models/` package, **not** in `encoders/`, because `encoders/`
  is deliberately model-free (no torch) and this is a learned component.
- `save`/`load` persist the weights with a schema string and a config block
  (dims + card/enchantment vocab sizes); `load` rejects a mismatch, since the rows
  are keyed by card index and a different vocabulary would silently mis-map.
- Imports are restricted to `sts2rl.data.*` + `torch` (no `agents`/`encoders`) to
  avoid the known `encoders → agents` circular-import edge.

This is delivered **standalone** — not yet wired into the battle encoder or the
agent networks.

## Alternatives considered

- **Deterministic 16-d feature vector** (no learned params): rejected — not an
  embedding in the paper's sense, and from id alone it degenerates to a fixed
  projection of a one-hot or the ordinal scalar we are replacing.
- **Single composite-id embedding table** keyed by `(id, upgrade, enchant)`:
  rejected — vocabulary explosion, mostly-unseen rows.
- **1-d output per factor**: rejected for the categorical factors — collapses 576
  cards onto a line and re-imposes a false ordering; 1-d is fine only for the
  binary upgrade flag (which is why it stays a raw scalar).

## Consequences

- A richer, learnable card representation, reusable as a building block (e.g. a
  future enemy/relic embedding would follow the same pattern in `models/`).
- Using it in training is a **deferred follow-up**: `BattleStateEncoder` must emit
  card *indices* and the DQN/PPO networks must embed them — an action-schema bump
  that invalidates existing battle checkpoints and touches the encoder and both
  networks.
- The embedding is learned, so it carries its own parameters; the JSON lookup is
  necessary but not sufficient (it only yields the index).
