"""The tokenizer's columns and the encoder's embedding tables must agree."""

from __future__ import annotations

import pytest

from sts2rl.encoder import (
    ENTITY_CATEGORICAL,
    ENTITY_KINDS,
    ENTITY_NUMERIC_FIELDS,
    EVENT_EFFECT_KEYS,
    GLOBAL_CATEGORICAL,
    EncoderConfig,
    EntityTransformer,
    GameTokenizer,
    GameVocabulary,
)
from sts2rl.encoder.entity_encoder import ENTITY_CATEGORICAL_VOCABS
from sts2rl.encoder.game_tokenizer import _EFFECT_RULES
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


def test_every_kind_declares_both_categorical_and_numeric_columns():
    assert set(ENTITY_KINDS) == set(ENTITY_NUMERIC_FIELDS)
    assert set(ENTITY_KINDS) == set(ENTITY_CATEGORICAL_VOCABS)
    for kind in ENTITY_KINDS:
        assert ENTITY_CATEGORICAL[kind], f"{kind} declares no categorical column"


def test_every_categorical_column_names_a_real_vocabulary(
    vocabulary: GameVocabulary,
):
    columns = list(GLOBAL_CATEGORICAL)
    for kind in ENTITY_KINDS:
        columns.extend(ENTITY_CATEGORICAL[kind])

    for field, table in columns:
        assert vocabulary.size(table) > 0, f"{field} names unknown table {table}"


def test_the_encoder_builds_one_embedding_per_tokenized_column(
    vocabulary: GameVocabulary,
):
    """A column added on one side only would otherwise fail silently at runtime."""
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16))

    assert set(model.global_embeddings) == {field for field, _ in GLOBAL_CATEGORICAL}
    for kind in ENTITY_KINDS:
        for field, table in ENTITY_CATEGORICAL[kind]:
            if field == "entity_zone":
                assert model.zone_embedding.num_embeddings == vocabulary.size(table)
                continue
            embedding = model.entity_embeddings[f"{kind}__{field}"]
            assert embedding.num_embeddings == vocabulary.size(table)


def test_tokenized_widths_match_the_declared_schema(vocabulary: GameVocabulary):
    state = GameTokenizer(vocabulary).tokenize_state(
        GameObservation({"state_type": "map", "player": {"hp": 70, "max_hp": 80}})
    )

    assert state.global_categorical.shape[0] == len(GLOBAL_CATEGORICAL)
    for kind in ENTITY_KINDS:
        batch = state.entities[kind]
        assert batch.categorical.shape[1] == len(ENTITY_CATEGORICAL[kind])
        assert batch.numeric.shape[1] == len(ENTITY_NUMERIC_FIELDS[kind])


def test_every_effect_bucket_has_a_rule_and_a_column():
    """The rule order is the column order; a rule with no column is dropped."""
    assert tuple(key for key, _ in _EFFECT_RULES) == EVENT_EFFECT_KEYS
