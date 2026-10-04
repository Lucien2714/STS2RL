"""The vocabulary migration knows every embedding table the encoder has."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from sts2rl.encoder import EncoderConfig, GameEncoder, GameVocabulary

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_vocabulary.py"


def _script():
    spec = importlib.util.spec_from_file_location("migrate_vocabulary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_embedding_is_mapped_to_the_table_it_indexes():
    vocabulary = GameVocabulary.from_bundled_data()
    encoder = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4, entity_ff_dim=32))
    tables = _script().embedding_tables(encoder)
    state = encoder.state_dict()
    for name, table in tables.items():
        assert state[name].shape[0] == vocabulary.size(table), name
    embeddings = [m for m in encoder.modules() if isinstance(m, torch.nn.Embedding)]
    assert len(tables) == len(embeddings)
