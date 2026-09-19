"""Gameplay record parsing: what a record must say to be trusted."""

import json

import pytest

from sts2rl.offline.record import (
    RecordError,
    RecordedRun,
    find_records,
    load_records,
)


def state(floor: int = 1) -> dict:
    return {
        "state_type": "map",
        "run": {"act": 1, "floor": floor},
        "player": {"character": "The Ironclad", "potions": []},
        "map": {"next_options": [{"index": 0}, {"index": 1}]},
    }


def record(**overrides) -> dict:
    payload = {
        "schema_version": 2,
        "recorder_version": "0.3.0",
        "game_version": "v0.107.1",
        "run_id": "1789571004",
        "run": {
            "character": "IRONCLAD",
            "seed": "K7EAC3UM1M",
            "game_mode": "Standard",
            "ascension": 0,
            "num_reloads": 3,
        },
        "outcome": {"victory": True, "abandoned": False, "floor_reached": 48},
        "steps": [
            {
                "index": 0,
                "state": state(1),
                "player_detail": {"deck": {"cards": []}},
                "action": {
                    "action": "choose_map_node",
                    "args": {"index": 1},
                    "label": "Monster",
                },
            },
            {"index": 1, "state": state(2), "player_detail": None, "action": None},
        ],
    }
    payload.update(overrides)
    return payload


def write(path, payload, *, bom: bool = True) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(json.dumps(payload), encoding=encoding)


def test_a_record_with_a_byte_order_mark_still_reads(tmp_path):
    """The recorder writes a BOM, so utf-8-sig is not optional."""
    path = tmp_path / "run.json"
    write(path, record(), bom=True)

    run = RecordedRun.load(path)

    assert run.run_id == "1789571004"
    assert run.seed == "K7EAC3UM1M"
    assert run.victory is True
    assert run.floor_reached == 48
    assert run.num_reloads == 3
    assert len(run.steps) == 2


def test_a_step_becomes_an_observation_and_a_game_action(tmp_path):
    path = tmp_path / "run.json"
    write(path, record())

    step = RecordedRun.load(path).steps[0]
    observation = step.observation()

    assert observation.raw_state["state_type"] == "map"
    assert observation.player_detail == {"deck": {"cards": []}}
    assert step.action is not None
    assert step.action.to_game_action().to_dict() == {
        "type": "choose_map_node",
        "index": 1,
    }
    assert (step.act, step.floor) == (1, 1)


def test_the_final_step_carries_no_action(tmp_path):
    path = tmp_path / "run.json"
    write(path, record())

    assert RecordedRun.load(path).steps[-1].action is None


def test_a_resumed_step_is_marked(tmp_path):
    payload = record()
    payload["steps"][1]["resumed"] = True
    path = tmp_path / "run.json"
    write(path, payload)

    steps = RecordedRun.load(path).steps

    assert (steps[0].resumed, steps[1].resumed) == (False, True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 3}, "unsupported schema_version"),
        ({"schema_version": None}, "unsupported schema_version"),
        ({"run_id": ""}, "non-empty string run_id"),
        ({"run_id": 1789571004}, "non-empty string run_id"),
        ({"steps": []}, "non-empty steps array"),
        ({"steps": {}}, "non-empty steps array"),
    ],
)
def test_a_record_that_cannot_be_trusted_is_refused(tmp_path, overrides, message):
    path = tmp_path / "run.json"
    write(path, record(**overrides))

    with pytest.raises(RecordError, match=message):
        RecordedRun.load(path)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"index": "0", "state": state()}, "non-integer index"),
        ({"index": 0, "state": {}}, "non-empty state object"),
        ({"index": 0, "state": None}, "non-empty state object"),
        (
            {"index": 0, "state": state(), "player_detail": []},
            "player_detail must be an object or null",
        ),
        (
            {"index": 0, "state": state(), "action": {"args": {}}},
            "non-empty 'action' name",
        ),
        (
            {"index": 0, "state": state(), "action": {"action": "proceed", "args": 3}},
            "args must be an object or null",
        ),
        ({"index": 0, "state": state(), "action": 7}, "action must be an object"),
    ],
)
def test_a_step_that_cannot_be_trusted_is_refused(tmp_path, step, message):
    path = tmp_path / "run.json"
    write(path, record(steps=[step]))

    with pytest.raises(RecordError, match=message):
        RecordedRun.load(path)


def test_step_indices_must_increase(tmp_path):
    """The index locates a step in the record, so a repeat is ambiguous."""
    path = tmp_path / "run.json"
    write(
        path,
        record(
            steps=[
                {"index": 4, "state": state()},
                {"index": 4, "state": state(2)},
            ]
        ),
    )

    with pytest.raises(RecordError, match="indices must increase"):
        RecordedRun.load(path)


def test_invalid_json_is_named_rather_than_raised_raw(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RecordError, match="not valid JSON"):
        RecordedRun.load(path)


def test_records_are_found_in_a_stable_order(tmp_path):
    for name in ("b.json", "a.json", "c.json"):
        write(tmp_path / name, record(run_id=name))
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert [path.name for path in find_records(tmp_path)] == [
        "a.json",
        "b.json",
        "c.json",
    ]


def test_one_file_is_a_valid_source(tmp_path):
    path = tmp_path / "run.json"
    write(path, record())

    assert find_records(path) == (path,)


@pytest.mark.parametrize("missing", ["absent", "empty"])
def test_a_source_with_no_records_is_refused(tmp_path, missing):
    target = tmp_path / "absent" if missing == "absent" else tmp_path
    if missing == "empty":
        target.mkdir(exist_ok=True)

    with pytest.raises(RecordError, match="gameplay records"):
        find_records(target)


def test_the_same_run_cannot_be_cleaned_twice(tmp_path):
    """Two files carrying one run id would double-count its decisions."""
    write(tmp_path / "a.json", record())
    write(tmp_path / "b.json", record())

    with pytest.raises(RecordError, match="appears in both"):
        load_records(find_records(tmp_path))
