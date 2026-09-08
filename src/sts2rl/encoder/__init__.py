"""Structured game tokenization and trainable encoding interfaces."""

from sts2rl.encoder.entity_encoder import (
    ENTITY_CATEGORICAL_VOCABS,
    GLOBAL_CATEGORICAL_VOCABS,
    EncodedEntities,
    EncoderConfig,
    EntityTransformer,
)
from sts2rl.encoder.game_tokenizer import GameTokenizer, TokenizationError
from sts2rl.encoder.schema import (
    ACTION_NUMERIC_FIELDS,
    ENTITY_CATEGORICAL,
    ENTITY_CATEGORICAL_FIELDS,
    ENTITY_KINDS,
    ENTITY_NUMERIC_FIELDS,
    EVENT_EFFECT_KEYS,
    GLOBAL_CATEGORICAL,
    GLOBAL_CATEGORICAL_FIELDS,
    GLOBAL_NUMERIC_FIELDS,
    MAP_CATEGORICAL_FIELDS,
    MAP_NUMERIC_FIELDS,
)
from sts2rl.encoder.game_encoder import (
    EncodedDecision,
    GameEncoder,
    PolicyValueOutput,
)
from sts2rl.encoder.numeric import (
    NumericFeature,
    linear_feature,
    pack_numeric,
    ratio_feature,
    signed_log_feature,
)
from sts2rl.encoder.map_encoder import EncodedMap, MapDAGEncoder
from sts2rl.encoder.tokens import (
    EntityReference,
    TokenizedAction,
    TokenizedDecision,
    TokenizedEntityBatch,
    TokenizedMap,
    TokenizedState,
)
from sts2rl.encoder.vocabulary import (
    PAD_INDEX,
    UNKNOWN_INDEX,
    EventOptionVocabulary,
    GameVocabulary,
    TokenVocabulary,
    normalize_token,
)

__all__ = [
    "ACTION_NUMERIC_FIELDS",
    "EntityReference",
    "EncodedEntities",
    "EncodedDecision",
    "EncodedMap",
    "EncoderConfig",
    "EntityTransformer",
    "ENTITY_CATEGORICAL_VOCABS",
    "EventOptionVocabulary",
    "GameTokenizer",
    "GameEncoder",
    "GameVocabulary",
    "MapDAGEncoder",
    "GLOBAL_CATEGORICAL",
    "GLOBAL_CATEGORICAL_FIELDS",
    "GLOBAL_CATEGORICAL_VOCABS",
    "GLOBAL_NUMERIC_FIELDS",
    "MAP_CATEGORICAL_FIELDS",
    "MAP_NUMERIC_FIELDS",
    "ENTITY_CATEGORICAL",
    "ENTITY_CATEGORICAL_FIELDS",
    "ENTITY_KINDS",
    "ENTITY_NUMERIC_FIELDS",
    "EVENT_EFFECT_KEYS",
    "NumericFeature",
    "PAD_INDEX",
    "PolicyValueOutput",
    "TokenizedAction",
    "TokenizedDecision",
    "TokenizedEntityBatch",
    "TokenizedMap",
    "TokenizedState",
    "TokenizationError",
    "TokenVocabulary",
    "UNKNOWN_INDEX",
    "linear_feature",
    "normalize_token",
    "pack_numeric",
    "ratio_feature",
    "signed_log_feature",
]
