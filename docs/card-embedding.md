# Card embedding (`CardModelEncoder`)

A learned, low-dimensional representation of a single card, inspired by Gerigk &
Engels, *Learning Various Strategies for Dominion with Deep Reinforcement
Learning* (AIIDE 2023), which embeds each card and pools cards as a multiset. The
decision record is [ADR-0004](adr/0004-learned-card-embedding.md); this note is
the how-it-works / how-to-use companion.

Module: `sts2rl/models/card_encoder.py` — `class CardModelEncoder(nn.Module)`.

## What it produces

A card → a learned `out_dim`-d vector (default 16), derived from **only three
factors**: the card's id, `is_upgraded`, and enchantment. Nothing else (cost,
type, rarity, description) feeds it.

## Why a learned embedding, not the JSON lookup

These are two different tables and both are needed:

```
data/json (DataIdMap):   "STRIKE"  ──►  497              string → index   (fixed, not learned)
nn.Embedding.weight:        497    ──►  [0.21, -0.4, …]  index → 16 numbers (LEARNED, trained)
```

The JSON answers *"which row is this card?"* (an integer). `nn.Embedding` is a
`(vocab+1) × dim` matrix of **trainable** numbers whose row is that card's vector;
it starts random and is shaped by backprop, so similar cards drift to nearby
vectors. Feeding the raw index `497` (or `497/576`) as a feature instead would
re-impose a meaningless ordering over the 576 ids — the very thing this replaces.

## Factored design (concat + linear)

The three factors are combined by concatenation followed by a linear projection:

| Factor | Kind | Representation | Width |
|--------|------|----------------|-------|
| card id | categorical (576) | `nn.Embedding(576+1, 16)` | 16 |
| enchantment | categorical (22) | `nn.Embedding(22+1, 8)` | 8 |
| `is_upgraded` | binary | raw `0.0/1.0` scalar | 1 |

`concat([id_emb, ench_emb, upgrade_flag]) → Linear(25 → 16)`.

The categorical factors get multi-dimensional **embeddings** (a 1-d output would
collapse them and re-impose a false ordering). `is_upgraded` is one bit, so a
single scalar carries all of its information — no embedding needed. Embedding
**row 0** of each table is reserved for "unknown / none", so an unrecognized card
id or a card with no enchantment maps cleanly to a dedicated row.

## API

- `encode_card(card) -> Tensor[out_dim]` — the "card object in, vector out" entry
  point. Accepts a `Card`, a raw card dict, or a `CardIdentity` (all funnel
  through `CardIdentity`).
- `forward(card_idx, ench_idx, upgraded) -> Tensor[..., out_dim]` — the batched,
  differentiable tensor path that training will use directly.
- `card_to_indices(card) -> (card_idx, ench_idx, upgraded)` — the model-free
  bridge from a card object to factor indices, reusing `get_card_index` /
  `get_data_index_or_default` from `data/loader.py`.
- `save(path)` / `load(path)` — persist/restore weights with a schema + config
  guard; `load` raises on a dim/vocab mismatch (the rows are keyed by card index,
  so a different vocabulary would silently mis-map).

## Boundaries and dependencies

- Lives in `models/` (reusable learned components), **not** `encoders/`, which is
  intentionally model-free (no torch) — see
  [architecture.md](architecture.md).
- Imports only `sts2rl.data.*` + `torch`. It deliberately does not import
  `agents`/`encoders`, which keeps it clear of the existing `encoders → agents`
  circular-import edge (`encoders/battle_encoder.py` imports
  `agents/selection.py`).

## Status: wired in, opt-in (`--policy learned`)

`CardModelEncoder` is integrated as ADR-0007 Phase 3, behind an opt-in flag rather
than forced on every agent:

- `encoders/learned_battle_encoder.LearnedBattleStateEncoder` appends
  `(card_index, enchantment_index, upgraded)` channels per hand slot to the state
  vector, and one such triple to the action vector, *after* the handcrafted
  features. The handcrafted prefix is byte-identical, so the added channels are
  strictly extra information.
- `models/learned_policies` slices those channels back out, looks them up in an
  owned `CardModelEncoder`, mean-pools the hand embeddings over occupied slots,
  and concatenates everything before scoring.
- Because the embedding is a submodule of the policy, `Adam(policy.parameters())`
  already covers it: it trains end-to-end under behavioral cloning *and* RL with
  no special-casing in either update rule.

The index convention (row 0 reserved for unknown/none) is defined once, torch-free,
as `data.card.card_factor_indices` — the featurizer that writes the indices and the
embedding tables that consume them cannot drift apart.

Enable with `--policy learned` on `sts2rl-train` / `sts2rl-pretrain`. This is an
action-schema bump (`candidate_action_learned_v1`), so learned and flat checkpoints
are mutually unloadable — they live in separate directories
(`checkpoints/battleAgent/PPO-learned/`) and existing flat checkpoints are
untouched.

The embedding's parameters ride along in the agent's checkpoint;
`CardModelEncoder.save/load` remains for using or pretraining the module on its own.
