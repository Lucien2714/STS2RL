"""Cleaning contract: every recorded step is accounted for, and why."""

import copy
import json
from pathlib import Path

import pytest

from sts2rl.offline import clean, record as record_module
from sts2rl.offline.clean import (
    DATASET_FORMAT_VERSION,
    DECISIONS_FILENAME,
    MANIFEST_FILENAME,
    REJECTION_REASONS,
    Decision,
    clean_records,
    clean_run,
    read_decisions,
)
from sts2rl.offline.record import RecordedRun


def rewards_state(*, items: list[dict] | None = None, floor: int = 3) -> dict:
    """A rewards screen, which is two claims when it holds two outright items."""
    return {
        "state_type": "rewards",
        "run": {"act": 1, "floor": floor},
        "player": {"character": "The Ironclad", "potions": [], "relics": []},
        "rewards": {
            "items": (
                items
                if items is not None
                else [
                    {"index": 0, "type": "gold", "amount": 15},
                    {"index": 1, "type": "relic", "name": "Burning Blood"},
                ]
            ),
            "can_proceed": True,
        },
    }


def step(index: int, state: dict, action: dict | None, **extra) -> dict:
    return {
        "index": index,
        "state": state,
        "player_detail": {"deck": {"cards": []}},
        "action": action,
        **extra,
    }


def run_of(steps: list[dict], *, run_id: str = "run-a", path: str = "run-a.json"):
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run": {
            "character": "IRONCLAD",
            "seed": "K7EAC3UM1M",
            "game_mode": "Standard",
            "num_reloads": 0,
        },
        "outcome": {"victory": True, "abandoned": False, "floor_reached": 48},
        "steps": steps,
    }
    return RecordedRun.from_payload(payload, path=path)


def claim(index: int) -> dict:
    return {"action": "claim_reward", "args": {"index": index}}


def test_a_real_decision_is_kept_with_the_humans_candidate_index():
    run = run_of(
        [
            step(0, rewards_state(), claim(1)),
            step(1, rewards_state(floor=4), None),
        ]
    )

    decisions, report = clean_run(run)

    assert report.kept == 1
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.run_id == "run-a"
    assert decision.step_index == 0
    assert decision.state_type == "rewards"
    assert (decision.act, decision.floor) == (1, 3)
    assert [dict(c) for c in decision.candidates] == [
        {"type": "claim_reward", "index": 0},
        {"type": "claim_reward", "index": 1},
    ]
    assert decision.expert_index == 1
    assert dict(decision.expert_action) == {"type": "claim_reward", "index": 1}


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        pytest.param(
            [step(0, rewards_state(), None)],
            "terminal_no_action",
            id="the last step carries no decision",
        ),
        pytest.param(
            [
                step(0, rewards_state(), claim(0)),
                step(1, rewards_state(), None),
            ],
            "no_observable_effect",
            id="refused and unchanged is not a transition",
        ),
        pytest.param(
            [
                step(
                    0,
                    {
                        "state_type": "game_over",
                        "run": {"act": 3, "floor": 48},
                        "player": {"potions": []},
                    },
                    {"action": "menu_select", "args": {"option": "main_menu"}},
                ),
                step(1, rewards_state(), None),
            ],
            "no_candidates",
            id="a screen the provider cannot automate",
        ),
        pytest.param(
            [
                step(0, rewards_state(), {"action": "proceed", "args": {}}),
                step(1, rewards_state(floor=4), None),
            ],
            "unmatched_action",
            id="an action the provider withholds on purpose",
        ),
        pytest.param(
            [
                step(
                    0,
                    rewards_state(items=[{"index": 0, "type": "gold", "amount": 9}]),
                    claim(0),
                ),
                step(1, rewards_state(floor=4), None),
            ],
            "forced_single_candidate",
            id="one candidate carries no gradient",
        ),
    ],
)
def test_each_rejection_reason_is_reachable(steps, reason):
    decisions, report = clean_run(run_of(steps))

    assert report.rejected[reason] == 1
    assert not [
        other
        for other, count in report.rejected.items()
        if other != reason and other != "terminal_no_action" and count
    ]
    assert len(decisions) == report.kept


def test_the_ledger_accounts_for_every_step():
    """kept plus the ledger is the number of recorded steps, always."""
    steps = [
        step(0, rewards_state(), claim(1)),
        step(1, rewards_state(floor=4), {"action": "proceed", "args": {}}),
        step(2, rewards_state(floor=5), claim(0)),
        step(3, rewards_state(floor=5), claim(0)),
        step(4, rewards_state(floor=6), None),
    ]

    decisions, report = clean_run(run_of(steps))

    assert report.steps == 5
    assert report.kept + sum(report.rejected.values()) == report.steps
    assert set(report.rejected) == set(REJECTION_REASONS)
    assert len(decisions) == report.kept


def test_a_mismatch_is_named_rather_than_only_counted():
    """A pooled count cannot say whether a mismatch is accounted for."""
    run = run_of(
        [
            step(0, rewards_state(), {"action": "proceed", "args": {}}),
            step(1, rewards_state(floor=4), None),
        ]
    )

    _, report = clean_run(run)

    assert report.unmatched_examples == [
        {
            "step_index": 0,
            "state_type": "rewards",
            "action": {"type": "proceed"},
            "reason": "type_absent",
            "offered_types": ["claim_reward"],
        }
    ]


def test_named_mismatches_are_capped():
    steps = [
        step(index, rewards_state(floor=index), {"action": "proceed", "args": {}})
        for index in range(5)
    ]
    steps.append(step(5, rewards_state(floor=9), None))

    _, report = clean_run(run_of(steps), max_unmatched_examples=2)

    assert report.rejected["unmatched_action"] == 5
    assert len(report.unmatched_examples) == 2


def test_a_reload_breaks_the_transition_chain():
    """After a reload the next state is not what this action produced.

    A reward read as a difference between those two states would invent
    progress that never happened, so the successor is withheld rather than
    offered to a future critic.
    """
    run = run_of(
        [
            step(0, rewards_state(), claim(1)),
            step(1, rewards_state(floor=4), claim(1), resumed=True),
            step(2, rewards_state(floor=5), None),
        ]
    )

    decisions, _ = clean_run(run)

    assert [d.step_index for d in decisions] == [0, 1]
    assert decisions[0].next_step_index is None
    assert decisions[1].next_step_index == 2


def test_the_last_kept_decision_of_a_run_is_marked():
    run = run_of(
        [
            step(0, rewards_state(), claim(1)),
            step(1, rewards_state(floor=4), claim(1)),
            step(2, rewards_state(floor=5), None),
        ]
    )

    decisions, _ = clean_run(run)

    assert [d.is_run_final_decision for d in decisions] == [False, True]
    assert [d.steps_to_run_end for d in decisions] == [2, 1]


def records_dir(tmp_path, *runs: tuple[str, list[dict]]):
    directory = tmp_path / "records"
    directory.mkdir()
    for run_id, steps in runs:
        payload = {
            "schema_version": 2,
            "run_id": run_id,
            "run": {"character": "IRONCLAD", "seed": f"SEED-{run_id}"},
            "outcome": {"victory": False, "abandoned": False, "floor_reached": 12},
            "steps": steps,
        }
        (directory / f"{run_id}.json").write_text(
            json.dumps(payload), encoding="utf-8-sig"
        )
    return directory


def two_decision_steps() -> list[dict]:
    return [
        step(0, rewards_state(), claim(1)),
        step(1, rewards_state(floor=4), claim(0)),
        step(2, rewards_state(floor=5), None),
    ]


def test_a_dataset_round_trips_through_the_written_file(tmp_path):
    source = records_dir(tmp_path, ("run-a", two_decision_steps()))

    result = clean_records(source, tmp_path / "out")

    assert result.decisions == 2
    assert result.decisions_path.name == DECISIONS_FILENAME
    assert result.manifest_path.name == MANIFEST_FILENAME
    restored = list(read_decisions(result.decisions_path))
    assert [d.step_index for d in restored] == [0, 1]
    assert [d.expert_index for d in restored] == [1, 0]
    assert restored[0].raw_state == rewards_state()


def test_the_manifest_records_the_run_and_the_ledger(tmp_path):
    source = records_dir(
        tmp_path,
        ("run-a", two_decision_steps()),
        ("run-b", two_decision_steps()),
    )

    result = clean_records(source, tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["dataset_format_version"] == DATASET_FORMAT_VERSION
    assert manifest["decisions"] == 4
    assert manifest["steps"] == 6
    assert manifest["rejected"]["terminal_no_action"] == 2
    assert [entry["run_id"] for entry in manifest["runs"]] == ["run-a", "run-b"]
    assert manifest["runs"][0]["seed"] == "SEED-run-a"
    assert manifest["runs"][0]["kept"] == 2
    assert result.steps == 6
    assert result.decisions + sum(result.rejected.values()) == result.steps


def test_cleaning_twice_replaces_rather_than_appends(tmp_path):
    source = records_dir(tmp_path, ("run-a", two_decision_steps()))
    out = tmp_path / "out"

    clean_records(source, out)
    result = clean_records(source, out)

    assert len(list(read_decisions(result.decisions_path))) == 2


@pytest.mark.parametrize("value", [-1, True])
def test_an_invalid_example_cap_is_refused(tmp_path, value):
    source = records_dir(tmp_path, ("run-a", two_decision_steps()))

    with pytest.raises((ValueError, TypeError)):
        clean_records(source, tmp_path / "out", max_unmatched_examples=value)


def test_a_decision_line_with_an_out_of_range_label_is_refused():
    payload = {
        "run_id": "run-a",
        "step_index": 0,
        "raw_state": {"state_type": "rewards"},
        "candidates": [{"type": "proceed"}],
        "expert_index": 3,
    }

    with pytest.raises(ValueError, match="not a position in its 1 candidates"):
        Decision.from_json(payload)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"raw_state": {}}, "non-empty object"),
        ({"candidates": []}, "non-empty array"),
        ({"run_id": ""}, "non-empty string"),
        ({"step_index": "0"}, "must be an integer"),
    ],
)
def test_a_malformed_decision_line_is_refused(overrides, message):
    payload = {
        "run_id": "run-a",
        "step_index": 0,
        "raw_state": {"state_type": "rewards"},
        "candidates": [{"type": "proceed"}],
        "expert_index": 0,
    }
    payload.update(overrides)

    with pytest.raises(ValueError, match=message):
        Decision.from_json(payload)


def test_a_decision_survives_a_json_round_trip():
    original = Decision(
        run_id="run-a",
        step_index=7,
        state_type="rewards",
        act=2,
        floor=23,
        raw_state=rewards_state(),
        player_detail={"deck": {"cards": []}},
        candidates=({"type": "claim_reward", "index": 0},),
        expert_index=0,
        next_step_index=8,
        steps_to_run_end=4,
        is_run_final_decision=True,
    )

    assert Decision.from_json(json.loads(json.dumps(original.to_json()))) == original


@pytest.mark.parametrize("module", [clean, record_module])
def test_cleaning_never_reaches_into_the_encoder(module):
    """Cleaning depends on the action layer and not on the encoder.

    That boundary is what lets a dataset outlive a column added to
    encoder/schema.py, and it is why --verify is a separate opt-in pass.  It is
    a source boundary rather than a runtime one: sts2rl.agents eagerly imports
    the PPO agent, so torch is loaded either way.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "sts2rl.encoder" not in source


def test_the_source_record_is_not_mutated():
    state = rewards_state()
    original = copy.deepcopy(state)
    run = run_of([step(0, state, claim(1)), step(1, rewards_state(floor=4), None)])

    clean_run(run)

    assert state == original


def outcome_records(tmp_path, *runs):
    """Write records whose outcomes differ, to filter between."""
    directory = tmp_path / "records"
    directory.mkdir(exist_ok=True)
    for run_id, victory, floor in runs:
        payload = {
            "schema_version": 2,
            "run_id": run_id,
            "run": {"character": "IRONCLAD", "seed": f"SEED-{run_id}"},
            "outcome": {
                "victory": victory,
                "abandoned": False,
                "floor_reached": floor,
            },
            "steps": two_decision_steps(),
        }
        (directory / f"{run_id}.json").write_text(
            json.dumps(payload), encoding="utf-8-sig"
        )
    return directory


def test_only_victories_excludes_the_runs_that_died(tmp_path):
    """A run that died demonstrates losing from its fatal mistake onward."""
    source = outcome_records(
        tmp_path, ("won", True, 48), ("died", False, 12), ("unknown", None, 20)
    )

    result = clean_records(source, tmp_path / "out", only_victories=True)

    assert [report.run_id for report in result.runs] == ["won"]
    assert {run.run_id: run.reason for run in result.excluded} == {
        "died": "not a victory",
        "unknown": "not a victory",
    }
    assert result.decisions == 2


def test_min_floor_excludes_the_runs_that_stopped_early(tmp_path):
    source = outcome_records(tmp_path, ("deep", False, 30), ("shallow", False, 6))

    result = clean_records(source, tmp_path / "out", min_floor=20)

    assert [report.run_id for report in result.runs] == ["deep"]
    assert result.excluded[0].run_id == "shallow"
    assert "below --min-floor 20" in result.excluded[0].reason


def test_an_unknown_floor_cannot_clear_a_min_floor(tmp_path):
    source = outcome_records(tmp_path, ("nofloor", True, None))

    result = clean_records(source, tmp_path / "out", min_floor=5)

    assert not result.runs
    assert "floor_reached unknown" in result.excluded[0].reason


def test_an_excluded_run_contributes_nothing_to_the_ledger(tmp_path):
    """Excluding is per run, before its steps are read."""
    source = outcome_records(tmp_path, ("won", True, 48), ("died", False, 12))

    result = clean_records(source, tmp_path / "out", only_victories=True)

    assert result.steps == 3
    assert result.decisions + sum(result.rejected.values()) == result.steps


def test_the_manifest_names_what_was_excluded(tmp_path):
    """"563 decisions" means something different with eighteen runs filtered."""
    source = outcome_records(tmp_path, ("won", True, 48), ("died", False, 12))

    result = clean_records(source, tmp_path / "out", only_victories=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert [entry["run_id"] for entry in manifest["excluded_runs"]] == ["died"]
    assert manifest["excluded_runs"][0]["floor_reached"] == 12


def test_no_filter_excludes_nothing(tmp_path):
    source = outcome_records(tmp_path, ("won", True, 48), ("died", False, 12))

    result = clean_records(source, tmp_path / "out")

    assert result.excluded == ()
    assert len(result.runs) == 2


@pytest.mark.parametrize("value", [0, -3, True])
def test_an_invalid_min_floor_is_refused(tmp_path, value):
    source = outcome_records(tmp_path, ("won", True, 48))

    with pytest.raises(ValueError, match="min_floor must be a positive integer"):
        clean_records(source, tmp_path / "out", min_floor=value)
