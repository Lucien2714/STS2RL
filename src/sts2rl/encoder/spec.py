"""A model's input definition, stored beside its weights.

The vocabulary tables and the schema's columns are what give an embedding row or an
input column of a trained model its meaning. They live in code, so after a
vocabulary refresh or a new column the old definition is gone with the old checkout.
Every checkpoint therefore carries this spec, and ``training/surgery.py`` reads it to
move the old weights to where the new model expects them.
"""

from __future__ import annotations

from typing import Any

from sts2rl.encoder.schema import (
    ACTION_NUMERIC_FIELDS,
    ENTITY_CATEGORICAL,
    ENTITY_NUMERIC_FIELDS,
    GLOBAL_CATEGORICAL,
    GLOBAL_NUMERIC_FIELDS,
    MAP_NUMERIC_FIELDS,
)
from sts2rl.encoder.vocabulary import GameVocabulary

SPEC_VERSION = 1
EVENT_OPTIONS = "event_options"


def schema_spec() -> dict[str, Any]:
    """This checkout's model-visible columns, in the order the encoder reads them."""
    return {
        "global_categorical": [list(column) for column in GLOBAL_CATEGORICAL],
        "entity_categorical": {
            kind: [list(column) for column in columns]
            for kind, columns in ENTITY_CATEGORICAL.items()
        },
        "global_numeric": list(GLOBAL_NUMERIC_FIELDS),
        "entity_numeric": {kind: list(fields) for kind, fields in ENTITY_NUMERIC_FIELDS.items()},
        "action_numeric": list(ACTION_NUMERIC_FIELDS),
        "map_numeric": list(MAP_NUMERIC_FIELDS),
    }


def model_spec(vocabulary: GameVocabulary) -> dict[str, Any]:
    """Everything that gives a checkpoint's input weights their meaning (~40 KB)."""
    return {
        "spec_version": SPEC_VERSION,
        "fingerprint": vocabulary.fingerprint(),
        "tables": {name: list(table.tokens) for name, table in vocabulary.tables.items()},
        EVENT_OPTIONS: [list(pair) for pair in vocabulary.event_options.pairs],
        "schema": schema_spec(),
    }
