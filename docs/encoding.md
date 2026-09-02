# Structured Game Encoding

This document describes the data contract between raw STS2MCP observations,
the deterministic tokenizer, trainable game encoder, and PPO agent.

The value types and non-map state tokenizer described here are implemented.
Candidate-action tokenization, map tokenization, and `GameEncoder` belong to
later refactor stages and are explicitly marked as such below.

## Data flow

```text
GameEnv raw state + player-detail
                 |
                 v
         GameObservation
                 |
                 | GameTokenizer
                 v
  TokenizedState / TokenizedDecision
                 |
                 | GameEncoder (not implemented yet)
                 v
 state embedding + candidate embeddings
                 |
                 v
        candidate-action PPO
```

The boundary is deliberately split into deterministic and trainable work:

- `GameVocabulary` assigns stable categorical indices.
- `numeric.py` parses and normalizes scalar values.
- `GameTokenizer` organizes non-map state values without trainable parameters.
- The token classes store and validate the resulting structured snapshot.
- `GameEncoder` will learn embeddings from that snapshot.

## GameObservation

`GameObservation` combines two STS2MCP responses:

```python
GameObservation(
    raw_state=raw_state,
    player_detail=player_detail,
)
```

`raw_state` contains the current screen, legal-decision context, player state,
and combat or map data. `player_detail` additionally provides the complete
permanent deck. It may be `None` for a terminal state where no active run
remains.

`GameObservation` is a frozen dataclass, but the raw dictionaries inside it are
not copied or recursively frozen. Callers must treat them as read-only.

The current `Agent` and `EpisodeRunner` still accept `RawState`; their migration
to `GameObservation` belongs to a later stage.

## Vocabulary indices

Categorical strings cannot be passed directly to `torch.nn.Embedding`.
`GameVocabulary` maps them to deterministic integer indices:

```python
vocabulary.lookup("cards", "BASH")
vocabulary.lookup("rarities", "Uncommon")
vocabulary.event_option_index("ABYSSAL_BATHS", "Immerse")
```

Every table reserves:

```text
0 = PAD      field or category is absent
1 = UNKNOWN  field is present but not in the bundled vocabulary
2+           real values
```

Lookups apply `strip().casefold()`. Real IDs are sorted by normalized value, so
indices do not depend on JSON record order. An unambiguous bundled display name
is an alias of its canonical ID; for example, `"Uppercut"` and `"UPPERCUT"`
resolve to the same card index. Ambiguous names shared by multiple IDs and
runtime names absent from bundled data remain `UNKNOWN`.

Vocabulary tables describe independent factors. The encoder will not allocate
one entry for every card, upgrade, enchantment, zone, and cost combination. A
card instead carries separate IDs and numeric values for those factors.

## Numeric features

All numeric model inputs are represented by a float32 value and a boolean
presence mask:

```text
value = 0, mask = true   observed numeric zero
value = 0, mask = false  source value was missing or invalid
```

The available conversions are:

- `linear_feature(value)` for small values with a meaningful scale.
- `signed_log_feature(value)` for unbounded or long-tailed counts.
- `ratio_feature(value, maximum)` for relative state such as HP percentage.
- `pack_numeric(features)` for creating the value and mask tensors.

Transformation is selected per field. It is not correct to apply logarithmic
compression to every number. For example, energy is naturally small and can be
linear, while gold benefits from logarithmic compression.

Invalid strings, booleans, NaN, infinity, and invalid ratio denominators are
represented as missing rather than silently converted into a valid zero.

## GameTokenizer state schema

`GameTokenizer.tokenize_state(observation)` creates a CPU `TokenizedState`.
Its output widths and column order are public constants:

```python
GLOBAL_CATEGORICAL_FIELDS
GLOBAL_NUMERIC_FIELDS
ENTITY_CATEGORICAL_FIELDS[kind]
ENTITY_NUMERIC_FIELDS[kind]
```

Consumers must use these names instead of duplicating numeric column offsets.
The categorical field names describe semantics; each column is looked up in
the corresponding vocabulary table.

Global categorical columns are `state_type` and `character`. Global numeric
columns are:

```text
act, floor, ascension,
hp, max_hp, hp_ratio, block, gold,
energy, max_energy, energy_ratio, stars,
max_potion_slots, deck_count,
draw_pile_count, discard_pile_count, exhaust_pile_count,
orb_slots, orb_empty_slots
```

Small bounded values such as act, ascension, energy, stars, and slot counts use
linear values. Counts whose useful range can grow, such as HP, block, gold,
floor, deck size, and pile size, use signed `log1p`. HP and energy also include
ratio columns with independently validated denominators.

The raw state is authoritative for live player and combat values. Player detail
fills fields absent from the raw player and is the sole source for the full
permanent deck. The raw run fields similarly override player-detail run fields.

Every supported entity kind is present in `TokenizedState.entities`, even when
it has zero rows. Empty tensors retain the kind's correct feature width. The
implemented kinds and primary content are:

| Kind | Categorical identity/context | Numeric state |
|---|---|---|
| `player` | character, owner type | resources and pile counts |
| `card` | card/type/rarity/zones/target/enchantment/selection type | costs, upgrades, count, semantic position, flags |
| `relic` | relic, rarity, zone | counter |
| `potion` | potion, target, zone | combat usability |
| `orb` | orb, zone | passive/evoke values and position |
| `pet` | monster ID, owner type, zone | HP, block, position |
| `enemy` | monster ID, owner type, zone | HP, block, position |
| `power` | power/type/owner type/zone | amount |
| `intent` | intent, zone | numeric label when parseable, position |
| `reward` | reward type and typed item IDs | gold amount |
| `shop_item` | category and typed item/card attributes | price and availability flags |
| `event_option` | event and canonical `(event_id, title)` option | option flags |
| `rest_option` | option ID and zone | enabled flag |
| `bundle` | selection type and zone | card count |
| `crystal_cell` | revealed item type and zone | coordinates, ratios, cell/item flags |
| `crystal_tool` | tool and zone | usability and selection flags |

`power.owners` points to the owning player, pet, or enemy row. `intent.owners`
points to its enemy row. A bundle's `children` points to the card rows contained
inside that bundle.

Only positions with game meaning are numeric features: hand order, orb order,
combatant formation, intent order, and Crystal Sphere coordinates. Raw action
handles such as card indices, potion slots, reward indices, and option indices
are deliberately excluded. Later action tokenization resolves those handles to
`EntityReference` values.

Descriptions, prompts, labels that are not plain numeric values, keyword prose,
and other free text are ignored. The state tokenizer also deliberately returns
`game_map=None`; complete map DAG parsing is a separate stage.

## EntityReference

An `EntityReference` is a semantic pointer:

```python
EntityReference(kind="card", index=3)
```

For ordinary entities this addresses row 3 of:

```python
state.entities["card"]
```

For a map node:

```python
EntityReference(kind="map_node", index=5)
```

the index addresses row 5 of `state.game_map`.

Indices are local to one tokenized state. They are not stable identities across
environment steps. Stable identity comes from vocabulary IDs stored in the
referenced row.

The `kind` is part of the reference so a card reference cannot accidentally
address a relic or enemy batch.

## TokenizedEntityBatch

All entities of one kind are stored in dense matrices:

```python
TokenizedEntityBatch(
    kind="card",
    categorical=Tensor[card_count, card_categorical_width],
    numeric=Tensor[card_count, card_numeric_width],
    numeric_mask=Tensor[card_count, card_numeric_width],
    active=Tensor[card_count],
    active_mask=Tensor[card_count],
    owners=(...),
    children=(...),
)
```

Tensor requirements are:

| Field | dtype | shape |
|---|---|---|
| `categorical` | `torch.long` | `[entity_count, categorical_width]` |
| `numeric` | `torch.float32` | `[entity_count, numeric_width]` |
| `numeric_mask` | `torch.bool` | same as `numeric` |
| `active` | `torch.bool` | `[entity_count]` |
| `active_mask` | `torch.bool` | `[entity_count]` |

The active fields form a three-state value:

| `active` | `active_mask` | meaning |
|---:|---:|---|
| true | true | explicitly active |
| false | true | explicitly inactive |
| false | false | active state unknown |
| true | false | invalid |

This distinction is important for one-use relics. The current STS2MCP relic
schema exposes `counter` but does not document `active`, `used`, or
`uses_remaining`. Until such a value is explicitly available, a relic's active
state must remain unknown rather than being guessed from its description.

`owners` and `children` preserve relationships outside the matrices. There is
one owner entry and one child tuple per entity row. Planned examples include a
power referencing its owning enemy and a bundle referencing its contained card
rows.

### Card multiplicity

Unordered zones group observationally identical card copies and include a
numeric `copy_count`. For example:

```text
draw pile: Strike, Strike, Strike+, Defend

card rows:
  Strike,  zone=draw, upgrade=0, count=2
  Strike,  zone=draw, upgrade=1, count=1
  Defend,  zone=draw, upgrade=0, count=1
```

Copies are grouped only when all observed runtime fields match, including ID,
zone, costs, upgrade level, enchantment, and target type.

Planned grouping rules are:

| source | representation |
|---|---|
| permanent deck | group identical states and store `copy_count` |
| draw pile | group identical states and store `copy_count` |
| discard pile | group identical states and store `copy_count` |
| exhaust pile | group identical states and store `copy_count` |
| hand | keep every actionable card as a separate row |
| selection screen | keep every selectable card as a separate row |
| orbs | keep every orb separately and preserve order |

The API documents draw-pile entries in display order rather than actual draw
order, so treating them as an unordered multiset does not discard an observable
top-to-bottom order.

## TokenizedState

`TokenizedState` is the complete model-visible snapshot for one environment
state:

```python
TokenizedState(
    global_categorical=Tensor[global_categorical_width],
    global_numeric=Tensor[global_numeric_width],
    global_numeric_mask=Tensor[global_numeric_width],
    entities={
        "card": card_batch,
        "relic": relic_batch,
        "enemy": enemy_batch,
    },
    game_map=tokenized_map_or_none,
)
```

Global categorical values include state type and character. Global numeric
values include the run and player scalars listed in the tokenizer schema above.
Their feature order is fixed by the public tokenizer constants.

`entities` is copied into an immutable mapping. Each mapping key must equal its
batch's `kind`. Owner and child references are validated when the state is
constructed.

Entity row order is part of the state snapshot. It does not need to remain the
same across later environment states because PPO stores the complete tokenized
decision used during sampling.

## TokenizedMap

The map is stored separately because it is a directed graph rather than an
unordered entity collection:

| Field | dtype | shape | meaning |
|---|---|---|---|
| `node_categorical` | long | `[N, C]` | node type and other categorical IDs |
| `node_numeric` | float32 | `[N, F]` | coordinates, distances, summaries |
| `node_numeric_mask` | bool | `[N, F]` | presence for numeric values |
| `edge_index` | long | `[2, E]` | parent-to-child edges |
| `topological_order` | long | `[N]` | every parent before its children |
| `reachable_mask` | bool | `[N]` | nodes still reachable from this state |
| `candidate_indices` | long | `[A]` | node row for every current map choice |
| `boss_indices` | long | `[B]` | terminal boss node rows |
| `candidate_type_counts` | float32 | `[A, T]` | full-subtree type counts |

`current_index` contains the current node row or `None` when no concrete current
position exists.

Every `edge_index` column is `[parent, child]`, moving from the current floor
toward a boss. The map encoder will traverse `topological_order` in reverse so
each parent can aggregate child representations that already contain their own
future descendants.

`TokenizedMap` validates shapes, index ranges, topological permutation, and
parent-before-child ordering. Building the graph, resolving `(col, row)`, and
computing reachability remain responsibilities of the future tokenizer.

## TokenizedAction

`TokenizedAction` represents one complete candidate action:

```python
TokenizedAction(
    action_type=Tensor[()],
    numeric=Tensor[action_numeric_width],
    numeric_mask=Tensor[action_numeric_width],
    source=EntityReference("card", 2),
    target=EntityReference("enemy", 0),
)
```

`action_type` is a scalar long vocabulary index. Numeric action fields are
reserved for parameters with semantic numeric meaning, such as a grid
coordinate.

Screen-local handles such as `card_index`, potion slot, or option index are not
numeric model features. The future tokenizer will resolve them to a semantic
source or target reference. End-turn, proceed, confirm, and skip actions can
legitimately have no references.

## TokenizedDecision

`TokenizedDecision` joins one state to the complete ordered candidate set:

```python
TokenizedDecision(
    state=state,
    actions=(play_card, use_potion, end_turn),
)
```

It requires at least one action and validates every action reference against
the state.

Candidate order must remain unchanged because PPO stores the selected candidate
index and old log probability. Reordering or regenerating candidates during an
update would make that probability refer to a different action.

## Device movement and rollout ownership

Every token container provides a non-mutating `.to(device)` method:

```python
gpu_decision = cpu_decision.to("cuda")
```

The returned dataclasses contain moved tensors; the original CPU tensors remain
in the rollout. `EntityReference` values contain no tensors and are reused.

Entity batches substantially reduce rollout overhead. A kind with 30 entities
uses a few matrices instead of 30 dataclasses containing three or more tiny
tensors. The tokenizer will only create rows for entities present in the
current observation; it never materializes all possible card-state
combinations.

## Responsibility summary

```text
GameObservation   raw state plus complete player detail
GameVocabulary    stable string-to-index identity
numeric.py        finite values, scaling, and missing masks
GameTokenizer     deterministic non-map state parsing; actions/map planned
TokenizedState    validated model-visible state snapshot
TokenizedMap      validated full map DAG
TokenizedAction   one semantic structured candidate
TokenizedDecision state plus ordered dynamic candidate set
GameEncoder       trainable state/action representation (planned)
PPO               candidate selection and learning
```
