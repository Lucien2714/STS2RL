"""Deterministic vocabulary behavior tests."""

from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

from sts2rl.encoder import (
    GameVocabulary,
    PAD_INDEX,
    UNKNOWN_INDEX,
    TokenVocabulary,
)


def test_bundled_vocabulary_reserves_special_indices_and_normalizes_ids():
    vocabulary = GameVocabulary.from_bundled_data()

    abrasive_index = vocabulary.lookup("cards", "ABRASIVE")

    assert vocabulary.lookup("card", None) == PAD_INDEX
    assert vocabulary.lookup("cards", "DOES_NOT_EXIST") == UNKNOWN_INDEX
    assert vocabulary.lookup("cards", "  abrasive  ") == abrasive_index
    assert abrasive_index >= 2
    assert vocabulary.table("cards").token(abrasive_index) == "ABRASIVE"
    assert vocabulary.size("cards") == 578
    assert vocabulary.lookup("power", "STRENGTH") >= 2
    assert vocabulary.lookup("monster", "ARCHITECT") >= 2
    assert vocabulary.lookup("intent", "ATTACK") >= 2


def test_vocabulary_indices_do_not_depend_on_json_record_order(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_records = [{"id": "ZETA"}, {"id": "ALPHA"}, {"id": "BETA"}]
    second_records = list(reversed(first_records))
    (first_dir / "cards.json").write_text(
        json.dumps(first_records),
        encoding="utf-8",
    )
    (second_dir / "cards.json").write_text(
        json.dumps(second_records),
        encoding="utf-8",
    )

    first = GameVocabulary.from_bundled_data(first_dir)
    second = GameVocabulary.from_bundled_data(second_dir)

    assert first.table("cards").tokens == second.table("cards").tokens
    assert first.lookup("cards", "alpha") == 2
    assert first.lookup("cards", "beta") == 3
    assert first.lookup("cards", "zeta") == 4


def test_event_options_include_top_level_and_page_options(tmp_path):
    events = [
        {
            "id": "TEST_EVENT",
            "options": [{"id": "FIRST", "title": "Take Gold"}],
            "pages": [
                {
                    "id": "NEXT",
                    "options": [
                        {"id": "SECOND", "title": "Leave"},
                        {"id": "THIRD", "title": "TAKE GOLD"},
                    ],
                }
            ],
        }
    ]
    (tmp_path / "events.json").write_text(json.dumps(events), encoding="utf-8")

    vocabulary = GameVocabulary.from_bundled_data(tmp_path)
    take_gold = vocabulary.event_option_index("TEST_EVENT", "Take Gold")

    assert take_gold >= 2
    assert vocabulary.event_option_index(" test_event ", " take gold ") == take_gold
    assert vocabulary.event_option_index("TEST_EVENT", "Leave") >= 2
    assert len(vocabulary.event_options) == 4
    assert vocabulary.event_option_index(None, "Leave") == PAD_INDEX
    assert (
        vocabulary.event_option_index("TEST_EVENT", "Unknown option") == UNKNOWN_INDEX
    )


def test_fixed_categories_are_case_insensitive_and_have_unknown_fallback():
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("state_types", " MAP ") >= 2
    assert vocabulary.lookup("action_types", "Choose_Map_Node") >= 2
    assert vocabulary.lookup("rarities", "uncommon") >= 2
    assert vocabulary.lookup("map_node_types", "RestSite") >= 2
    assert vocabulary.lookup("state_types", "future_state") == UNKNOWN_INDEX


def test_bundled_display_names_resolve_to_their_canonical_ids():
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("cards", "Uppercut") == vocabulary.lookup(
        "cards", "UPPERCUT"
    )
    assert vocabulary.lookup("monsters", "The Architect") == vocabulary.lookup(
        "monsters", "ARCHITECT"
    )
    assert vocabulary.lookup("orbs", "Lightning") == vocabulary.lookup(
        "orbs", "LIGHTNING_ORB"
    )
    assert vocabulary.lookup("cards", "Strike") == UNKNOWN_INDEX


def test_vocabulary_fingerprint_is_deterministic_and_content_sensitive():
    first = GameVocabulary.from_bundled_data()
    second = GameVocabulary.from_bundled_data()
    changed_tables = dict(first.tables)
    changed_tables["cards"] = TokenVocabulary.from_tokens(
        (*first.table("cards").tokens[2:], "FINGERPRINT_ONLY_CARD")
    )
    changed = replace(first, tables=MappingProxyType(changed_tables))

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64
    assert changed.fingerprint() != first.fingerprint()
