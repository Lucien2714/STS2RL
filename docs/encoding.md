# Structured Game Encoding

This document describes the data contract between raw STS2MCP observations,
the deterministic tokenizer, trainable game encoder, and PPO agent.

The value types, complete tokenizer, composed trainable `GameEncoder`, runner,
and PPO rollout integration described here are implemented.

## Data flow

```text
GameEnv raw state + /player deck
                 |
                 v
         GameObservation
                 |
                 | GameTokenizer
                 v
  TokenizedState / TokenizedDecision
                 |
                 | GameEncoder
                 v
 state embedding + candidate embeddings
                 |
                 v
        candidate-action PPO
```

The boundary is deliberately split into deterministic and trainable work:

- `GameVocabulary` assigns stable categorical indices from the bundled JSON
  tables, which are generated upstream by
  [spire-codex](https://github.com/ptrlrd/spire-codex) from the game itself.
- `numeric.py` parses and normalizes scalar values.
- `GameTokenizer` organizes state, map, and action values without trainable
  parameters.
- The token classes store and validate the resulting structured snapshot.
- `GameEncoder` learns embeddings from that snapshot during PPO updates.

## GameObservation

`GameObservation` combines the two STS2MCP responses the agent needs:

```python
GameObservation(raw_state=raw_state, player_detail=player_detail)
```

`raw_state` contains the current screen, legal-decision context, player state,
and combat or map data. `player_detail` is the `GET /api/v1/player` response;
its `deck` block is the only source of the run-level master deck, since game
state exposes the piles only during combat. It is `None` on terminal states,
before a run starts, and on builds without the endpoint.

`GameObservation` is a frozen dataclass, but the raw dictionaries inside it are
not copied or recursively frozen. Callers must treat them as read-only.

`Agent`, `Transition`, and `EpisodeRunner` use `GameObservation`. Reward models
and `EpisodeResult.initial_state/final_state` deliberately retain raw-state
inputs because rewards and external run summaries do not require model tokens.

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
floor, and pile size, use signed `log1p`. HP and energy also include
ratio columns with independently validated denominators.

The raw state is the only source of player and combat values. Fields the API
omits for a given character or screen — stars outside Regent, orb slots outside
Defect, energy outside combat — arrive masked as missing rather than zero.

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
and other free text are ignored. Non-map states return `game_map=None`; map
states contain the validated full DAG described below.

## GameTokenizer action schema

`GameTokenizer.tokenize_decision(observation, candidates)` tokenizes the state
once and preserves the exact candidate order. Runtime handles are used only to
resolve references:

| Action | Source | Target |
|---|---|---|
| `play_card` | hand card | optional enemy |
| `use_potion` | potion | optional enemy |
| `discard_potion` | potion | none |
| `combat_select_card` | hand-selection card | none |
| `claim_reward` | reward | none |
| `select_card_reward` | reward card | none |
| `choose_event_option` | event option | none |
| `choose_rest_option` | rest option | none |
| `shop_purchase` | shop item | none |
| `select_card` | selection card | none |
| `select_bundle` | bundle | none |
| `select_relic` | offered relic | none |
| `claim_treasure_relic` | treasure relic | none |
| `crystal_sphere_set_tool` | chosen tool | none |
| `crystal_sphere_click_cell` | currently selected tool, when present | cell |

Confirm, skip, proceed, end-turn, dialogue-advance, cancel, and Crystal Sphere
proceed actions have no entity references. The fixed action numeric schema is
`ACTION_NUMERIC_FIELDS = ("x", "y")`; only a Crystal cell click populates these
coordinates, while every other action has false masks for both fields.

Raw `card_index`, `slot`, option `index`, target `entity_id`, and tool strings do
not become model features. Duplicate or missing handles are not resolved by
list position. They raise `TokenizationError`, whose message includes both the
state type and complete action payload.

`choose_map_node` first resolves its temporary `next_options.index`, then stores
the corresponding coordinate node as its target. `menu_select` remains outside
the learning action space because reset navigation owns menus and no menu-option
entity exists.

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

## EntityTransformer

`EntityTransformer` is the first trainable encoding layer. Its default
configuration is:

```python
EncoderConfig(
    hidden_dim=128,
    entity_layers=2,
    entity_heads=4,
    entity_ff_dim=256,
    dropout=0.0,
)
```

`hidden_dim` must be divisible by `entity_heads`. Nonzero dropout is rejected,
because independently sampled encoder noise would contaminate PPO's old/new
probability ratio.

Each entity kind has separate embeddings for its categorical columns and a
separate numeric projection. Numeric projection input concatenates values,
missing masks, `active`, and `active_mask`. A shared entity-type embedding and
a shared zone embedding are then added. The global fields similarly use their
own categorical embeddings and numeric projection, added to a learned
`[STATE]` token.

Relationships are reduced before global self-attention:

- power rows are mean-aggregated into their player/enemy/pet owner through a
  learned projection;
- intent rows are mean-aggregated into their enemy owner;
- bundle child cards are mean-aggregated into their bundle;
- relation rows and bundled child rows are not duplicated in the Transformer
  sequence, although their base row embeddings remain addressable in the
  structured result.

`[STATE]` and the remaining entity rows pass through the configured Transformer
layers. There is no positional embedding. Reordering an unordered entity batch
therefore leaves the state embedding invariant and permutes that batch's row
outputs in the same way. Positions that genuinely matter were already encoded
as explicit numeric fields by the tokenizer.

The result is:

```python
EncodedEntities(
    state_embedding=Tensor[hidden_dim],
    entity_embeddings={kind: Tensor[entity_count, hidden_dim]},
)
```

`EncodedEntities.reference()` resolves ordinary `EntityReference` values. Map
nodes remain the responsibility of the separate DAG encoder.

## TokenizedMap

The map is stored separately because it is a directed graph rather than an
unordered entity collection:

| Field | dtype | shape | meaning |
|---|---|---|---|
| `node_categorical` | long | `[N, C]` | node type and other categorical IDs |
| `node_numeric` | float32 | `[N, F]` | coordinates, distances, summaries |
| `node_numeric_mask` | bool | `[N, F]` | presence for numeric values |
| `edge_index` | long | `[2, E]` | parent-to-child edges |
| `reachable_mask` | bool | `[N]` | nodes still reachable from this state |
| `candidate_indices` | long | `[A]` | node row for every current map choice |
| `boss_indices` | long | `[B]` | terminal boss node rows |
| `candidate_type_counts` | float32 | `[A, T]` | full-subtree type counts |

`current_index` contains the current node row or `None` when no concrete current
position exists.

Node identity is the exact `(col, row)` pair. Ordinary nodes and the singular
or plural boss records are merged by that identity, and node rows are ordered
deterministically by `(row, col)`. Duplicate ordinary node coordinates,
duplicate candidate coordinates, invalid coordinates, unresolved children,
and unresolved candidates raise `TokenizationError`.

Every `edge_index` column is `[parent, child]`, moving from the current floor
toward a boss. `MapDAGEncoder` groups nodes by their longest distance to a sink
and processes one level at a time, so each parent aggregates child
representations that already contain their own future descendants. The
tokenizer still runs a topological sort internally to detect cycles and to
compute boss distances, but it is not carried in the token.

`GameTokenizer` builds and validates the graph with these rules:

- Each edge is read from a node's full `children` list, not the one-level
  `next_options.leads_to` preview.
- A deterministic Kahn topological sort detects cycles.
- Reachability begins at the current candidates (or after the current node when
  no candidates exist) and excludes the current and visited nodes.
- Reverse-topological dynamic programming computes shortest and longest edge
  distance from every node to any boss. Nodes without a boss path use a false
  numeric mask for both distances.
- Each candidate's type-count row visits its complete descendant subgraph and
  counts shared DAG nodes once for that candidate.

The exact map columns are public as `MAP_CATEGORICAL_FIELDS` and
`MAP_NUMERIC_FIELDS`. Numeric fields contain coordinates, current/visited/
candidate/boss/reachable flags, and shortest/longest Boss distances.

`TokenizedMap` then independently validates shapes and index ranges.

## MapDAGEncoder

`MapDAGEncoder` consumes a `TokenizedMap` and the player-aware context produced
by `EntityTransformer`. Passing `game_map=None` returns `None`, so non-map
states skip this module.

First, node type and numeric/mask features form a base embedding. Nodes are
then grouped by their longest distance to a sink and evaluated one level at a
time, so a whole level is computed in a single batched attention. Every child
sits at a strictly lower level and is therefore already final:

```text
future(node) = update(
    base(node),
    attention(future(children), base(node), player_context),
    player_context,
)
```

The child query depends on both the current node and player context. Keys and
values come from child embeddings that already contain all deeper descendants.
Within a level, nodes with different child counts are padded and masked, so the
softmax still normalizes over real children only. One set of attention and
update parameters is shared at every map depth. This is dynamic programming
over the DAG: it is neither path enumeration nor a Graph Transformer.

An additional player-conditioned attention pool reads only rows selected by
`reachable_mask`. Disconnected or expired branches cannot affect the global map
embedding. An empty reachable set uses a learned empty-map vector combined with
player context.

The output keeps both representations for testing and downstream lookup:

```python
EncodedMap(
    base_node_embeddings=Tensor[node_count, hidden_dim],
    node_embeddings=Tensor[node_count, hidden_dim],  # future-aware
    global_embedding=Tensor[hidden_dim],
)
```

A candidate embedding therefore changes when a deep descendant changes, and a
gradient from that candidate reaches the descendant's base node embedding.

## GameEncoder and actor-critic heads

`GameEncoder` owns the entity Transformer, map DAG encoder, action encoder, and
shared actor-critic heads. Encoding follows this order:

1. `EntityTransformer` produces the player-aware state context and contextual
   ordinary entity rows.
2. On map states, that context conditions `MapDAGEncoder`.
3. A learned fusion of entity context and global map embedding produces the
   final state vector. Non-map states use a learned no-map vector and do not run
   the DAG module.
4. Every action combines its action-type embedding, numeric values and masks,
   role-specific source/target projections, and learned missing-role vectors.
   A map-node reference reads the future-aware node embedding.

The public encoding result is:

```python
EncodedDecision(
    state_embedding=Tensor[hidden_dim],
    candidate_embeddings=Tensor[candidate_count, hidden_dim],
)
```

`GameEncoder.policy_value()` applies the shared candidate actor and critic:

```text
logit(action) =
    dot(state_embedding, candidate_embedding) / sqrt(hidden_dim)
    + CandidateBias(candidate_embedding)

value = ValueHead(state_embedding)
```

`PolicyValueOutput` returns the rank-one dynamic logits, scalar value, and the
underlying `EncodedDecision`. The same parameters handle combat, map, event,
shop, rest, and selection screens; no screen-specific policy head or global
discrete action ID is introduced.

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
numeric model features. The tokenizer resolves them to a semantic source or
target reference. End-turn, proceed, confirm, and skip actions legitimately
have no references.

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

For every sampled action, PPO retains the complete `TokenizedDecision`, the
tokenized next state, and the original ordered `GameAction` tuple. It also
stores the selected index, old log probability, old value, reward, and terminal
flag. Update epochs rerun `GameEncoder` on the saved token snapshots instead of
reusing detached embeddings. Consequently, loss gradients train the complete
structured encoder as well as the actor and critic heads.

Terminal transitions use a zero bootstrap value. Non-terminal rollout
boundaries, including a runner step-limit truncation, evaluate the saved next
state with `GameEncoder.value()`. The default `gamma=0.999` and
`gae_lambda=0.98` reflect the long horizon of a complete run.

Entity batches substantially reduce rollout overhead. A kind with 30 entities
uses a few matrices instead of 30 dataclasses containing three or more tiny
tensors. The tokenizer will only create rows for entities present in the
current observation; it never materializes all possible card-state
combinations.

## Responsibility summary

```text
GameObservation   raw state plus the run-level deck
GameVocabulary    stable string-to-index identity
numeric.py        finite values, scaling, and missing masks
GameTokenizer     deterministic state, action, and full-map parsing
TokenizedState    validated model-visible state snapshot
TokenizedMap      validated full map DAG
TokenizedAction   one semantic structured candidate
TokenizedDecision state plus ordered dynamic candidate set
EntityTransformer trainable relation-aware non-map entity context
MapDAGEncoder      conditioned reverse-DAG future propagation
GameEncoder        composed state/action encoder and actor-critic heads
PPO               candidate selection and learning
```
