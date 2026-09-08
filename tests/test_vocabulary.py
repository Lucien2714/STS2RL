"""Deterministic vocabulary behavior tests."""

from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

from sts2rl.data import DEFAULT_DATA_DIR
from sts2rl.encoder import vocabulary as vocabulary_module
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
    # tracks the bundled data rather than a magic number, so refreshing the
    # tables from upstream does not require editing this assertion
    bundled_cards = json.loads(
        (DEFAULT_DATA_DIR / "cards.json").read_text(encoding="utf-8")
    )
    assert vocabulary.size("cards") == len(bundled_cards) + 2
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


def _event_table() -> list[dict[str, object]]:
    return [
        {
            "id": "TEST_EVENT",
            "options": [{"id": "FIRST", "title": "Take Gold"}],
            "pages": [
                {
                    "id": "INITIAL",
                    "options": [
                        {"id": "SECOND", "title": "Leave"},
                        {"id": "THIRD", "title": "TAKE GOLD"},
                        {"id": "RICH_LOCKED", "title": "Locked"},
                        {"id": "STRONG_LOCKED", "title": "Locked"},
                    ],
                },
                {"id": "DONE", "options": [{"id": "SECOND", "title": "Leave"}]},
            ],
        }
    ]


def _event_vocabulary(tmp_path) -> GameVocabulary:
    (tmp_path / "events.json").write_text(
        json.dumps(_event_table()),
        encoding="utf-8",
    )
    return GameVocabulary.from_bundled_data(tmp_path)


def test_event_options_are_keyed_by_text_key_across_pages(tmp_path):
    """The same option on two pages is one row; the page is its own column."""
    vocabulary = _event_vocabulary(tmp_path)

    leave = vocabulary.event_option_index(
        "TEST_EVENT", "TEST_EVENT.pages.INITIAL.options.SECOND"
    )

    assert leave >= 2
    assert vocabulary.event_options.pair(leave) == ("TEST_EVENT", "SECOND")
    assert (
        vocabulary.event_option_index(
            "TEST_EVENT", "TEST_EVENT.pages.DONE.options.SECOND"
        )
        == leave
    )
    # top-level options carry no page segment
    assert (
        vocabulary.event_option_index("TEST_EVENT", "TEST_EVENT.options.FIRST") >= 2
    )
    assert vocabulary.size("event_options") == 5 + 2
    assert vocabulary.lookup("event_pages", "INITIAL") >= 2
    assert vocabulary.lookup("event_pages", "DONE") >= 2


def test_a_text_key_beats_the_localized_title(tmp_path):
    """Two options titled "Locked" are two different options."""
    vocabulary = _event_vocabulary(tmp_path)

    rich = vocabulary.event_option_index(
        "TEST_EVENT",
        "TEST_EVENT.pages.INITIAL.options.RICH_LOCKED",
        "Locked",
    )
    strong = vocabulary.event_option_index(
        "TEST_EVENT",
        "TEST_EVENT.pages.INITIAL.options.STRONG_LOCKED",
        "Locked",
    )

    assert rich >= 2
    assert strong >= 2
    assert rich != strong


def test_a_title_resolves_only_while_it_names_one_option(tmp_path):
    """text_key can be unset, but a title that names two options names none."""
    vocabulary = _event_vocabulary(tmp_path)

    assert vocabulary.event_option_index(
        " test_event ", None, " leave "
    ) == vocabulary.event_option_index(
        "TEST_EVENT", "TEST_EVENT.pages.DONE.options.SECOND"
    )
    # two options, one title: on the bundled data this is every locked option
    assert vocabulary.event_option_index("TEST_EVENT", None, "Locked") == UNKNOWN_INDEX
    # ...and case is not what distinguishes them either
    assert vocabulary.event_option_index("TEST_EVENT", None, "Take Gold") == UNKNOWN_INDEX


def test_an_unresolvable_event_option_separates_absent_from_unrecognized(tmp_path):
    vocabulary = _event_vocabulary(tmp_path)

    assert vocabulary.event_option_index("TEST_EVENT") == PAD_INDEX
    assert vocabulary.event_option_index(None, None, None) == PAD_INDEX
    assert vocabulary.event_option_index("TEST_EVENT", None, "Nope") == UNKNOWN_INDEX
    assert (
        vocabulary.event_option_index("TEST_EVENT", "TEST_EVENT.options.NOPE")
        == UNKNOWN_INDEX
    )
    # a key of an unfamiliar shape is not sliced into nonsense
    assert vocabulary.event_option_index("TEST_EVENT", "SECOND") == UNKNOWN_INDEX


def test_parse_text_key_reads_the_parts_it_carries():
    assert vocabulary_module.parse_text_key(
        "TRASH_HEAP.pages.INITIAL.options.DIVE_IN"
    ) == ("TRASH_HEAP", "INITIAL", "DIVE_IN")
    assert vocabulary_module.parse_text_key("TRASH_HEAP.options.DIVE_IN") == (
        "TRASH_HEAP",
        None,
        "DIVE_IN",
    )
    assert vocabulary_module.parse_text_key("nonsense") == ("nonsense", None, None)
    assert vocabulary_module.parse_text_key("") == (None, None, None)


def test_the_fingerprint_covers_the_event_option_title_fallbacks(tmp_path):
    """A retitled option reroutes a live screen onto a different row."""
    before = _event_vocabulary(tmp_path).fingerprint()
    retitled = _event_table()
    retitled[0]["options"][0]["title"] = "Take The Gold"
    (tmp_path / "events.json").write_text(json.dumps(retitled), encoding="utf-8")

    after = GameVocabulary.from_bundled_data(tmp_path).fingerprint()

    assert before != after


def test_the_bundled_events_keep_their_pages_and_options_apart():
    vocabulary = GameVocabulary.from_bundled_data()

    linger = vocabulary.event_option_index(
        "ABYSSAL_BATHS", "ABYSSAL_BATHS.pages.INITIAL.options.LINGER"
    )

    assert linger >= 2
    # twelve pages offer LINGER; they share one row and differ by page
    assert (
        vocabulary.event_option_index(
            "ABYSSAL_BATHS", "ABYSSAL_BATHS.pages.LINGER_2.options.LINGER"
        )
        == linger
    )
    assert vocabulary.lookup("event_pages", "INITIAL") >= 2


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


def test_camel_case_enums_resolve_without_breaking_concatenated_tokens():
    """The API sends CamelCase where tables hold snake_case, and vice versa."""
    vocabulary = GameVocabulary.from_bundled_data()

    # tables hold DEBUFF_STRONG / CARD_DEBUFF / DEATH_BLOW
    for camel in ("DebuffStrong", "CardDebuff", "DeathBlow"):
        assert vocabulary.lookup("intents", camel) > UNKNOWN_INDEX

    # ...while map_node_types holds the concatenated "restsite"
    assert vocabulary.lookup("map_node_types", "RestSite") > UNKNOWN_INDEX
    assert vocabulary.lookup("map_node_types", "restsite") > UNKNOWN_INDEX


def test_lookup_first_falls_back_from_an_unrecognized_id_to_the_name():
    """Statuses arrive as STRENGTH_POWER against a table holding STRENGTH."""
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("powers", "STRENGTH_POWER") == UNKNOWN_INDEX
    assert vocabulary.lookup_first(
        "powers", ["STRENGTH_POWER", "Strength"]
    ) == vocabulary.lookup("powers", "STRENGTH")


def test_lookup_first_separates_absent_values_from_unrecognized_ones():
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup_first("powers", []) == PAD_INDEX
    assert vocabulary.lookup_first("powers", [None, None]) == PAD_INDEX
    assert vocabulary.lookup_first("powers", ["NOPE", "Nope"]) == UNKNOWN_INDEX


def test_no_state_type_token_the_api_cannot_produce():
    """/player carries no state_type, so the old player_detail token is gone."""
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("state_types", "player_detail") == UNKNOWN_INDEX


def test_the_games_own_spelling_of_a_status_intent_resolves():
    """The game sends StatusCard where the table holds STATUS.

    No rule bridges that: the CamelCase retry gives "status_card", and the
    display-name fallback cannot help because DEBUFF, DEBUFF_STRONG and STATUS
    all present as "Strategic", so that alias is ambiguous and dropped.
    """
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("intents", "StatusCard") == vocabulary.lookup(
        "intents", "STATUS"
    )
    assert vocabulary.lookup("intents", "StatusCard") != UNKNOWN_INDEX


def test_an_ambiguous_display_name_is_still_refused():
    """Three intents present as "Strategic"; guessing one would be worse."""
    vocabulary = GameVocabulary.from_bundled_data()

    assert vocabulary.lookup("intents", "Strategic") == UNKNOWN_INDEX


def test_every_declared_api_spelling_names_a_real_token():
    vocabulary = GameVocabulary.from_bundled_data()

    for table, spellings in vocabulary_module.API_SPELLINGS.items():
        for spelling, target in spellings.items():
            assert vocabulary.lookup(table, target) != UNKNOWN_INDEX
            assert vocabulary.lookup(table, spelling) == vocabulary.lookup(
                table, target
            )


def test_the_fingerprint_covers_the_api_spellings(monkeypatch):
    """Rerouting a token to another row changes what the model is shown.

    A checkpoint trained before the alias saw StatusCard as unknown; after it,
    the same screen lands on a different embedding row.  The fingerprint has to
    notice, or the old weights load silently against new inputs.
    """
    before = GameVocabulary.from_bundled_data().fingerprint()
    monkeypatch.setattr(
        vocabulary_module,
        "API_SPELLINGS",
        MappingProxyType({"intents": MappingProxyType({"Sleepy": "SLEEP"})}),
    )

    after = GameVocabulary.from_bundled_data().fingerprint()

    assert before != after
