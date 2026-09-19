"""Reading a cleaned dataset back: the labels are re-verified, not trusted."""

import gzip
import json

import pytest

from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.offline.clean import (
    DECISIONS_FILENAME,
    MANIFEST_FILENAME,
    Decision,
    clean_records,
)
from sts2rl.offline.dataset import BCDataset, DatasetError, StaleDatasetError


def rewards_state(floor: int = 3) -> dict:
    return {
        "state_type": "rewards",
        "run": {"act": 1, "floor": floor},
        "player": {"character": "The Ironclad", "potions": [], "relics": []},
        "rewards": {
            "items": [
                {"index": 0, "type": "gold", "amount": 15},
                {"index": 1, "type": "relic", "name": "Burning Blood"},
            ],
            "can_proceed": True,
        },
    }


def build(tmp_path, *run_ids: str, decisions_per_run: int = 2):
    """Write a records directory and clean it into a dataset directory."""
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    for run_id in run_ids:
        steps = [
            {
                "index": index,
                "state": rewards_state(index + 1),
                "player_detail": {"deck": {"cards": []}},
                "action": {
                    "action": "claim_reward",
                    "args": {"index": index % 2},
                },
            }
            for index in range(decisions_per_run)
        ]
        steps.append(
            {
                "index": decisions_per_run,
                "state": rewards_state(99),
                "player_detail": {"deck": {"cards": []}},
                "action": None,
            }
        )
        (records / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": run_id,
                    "run": {"character": "IRONCLAD", "seed": f"SEED{run_id}"},
                    "outcome": {"victory": True, "floor_reached": 48},
                    "steps": steps,
                }
            ),
            encoding="utf-8-sig",
        )
    out = tmp_path / f"out-{'-'.join(run_ids)}"
    clean_records(records, out)
    return out


def rewrite_first_decision(out, **changes):
    """Edit the first stored decision line in place."""
    path = out / DECISIONS_FILENAME
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    first = json.loads(lines[0])
    first.update(changes)
    lines[0] = json.dumps(first)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))


def test_a_dataset_tokenizes_every_decision_it_kept(tmp_path):
    dataset = BCDataset.load(build(tmp_path, "run-a"))

    assert len(dataset) == 2
    assert dataset.verify() == ()
    example = dataset[0]
    assert example.expert_index == dataset.decisions[0].expert_index
    assert example.run_id == "run-a"
    assert example.state_type == "rewards"
    assert len(example.decision.actions) == len(dataset.decisions[0].candidates)


def test_iterating_yields_one_example_per_decision(tmp_path):
    dataset = BCDataset.load(build(tmp_path, "run-a"))

    assert [example.step_index for example in dataset] == [0, 1]


def test_a_dataset_the_action_space_no_longer_agrees_with_is_refused(tmp_path):
    """expert_index is a position, so a shifted candidate set moves the label.

    Nothing downstream would notice -- the encoder would happily clone the
    policy onto a different action -- so the mismatch has to be fatal here.
    """
    out = build(tmp_path, "run-a")
    rewrite_first_decision(
        out,
        candidates=[
            {"type": "claim_reward", "index": 0},
            {"type": "claim_reward", "index": 1},
            {"type": "proceed"},
        ],
    )

    with pytest.raises(StaleDatasetError, match="re-run sts2rl-bc-clean"):
        BCDataset.load(out)


def test_a_reordered_candidate_set_is_also_refused(tmp_path):
    """Order is the label, so a permutation is as fatal as an addition."""
    out = build(tmp_path, "run-a")
    rewrite_first_decision(
        out,
        candidates=[
            {"type": "claim_reward", "index": 1},
            {"type": "claim_reward", "index": 0},
        ],
    )

    with pytest.raises(StaleDatasetError):
        BCDataset.load(out)


def test_an_unknown_dataset_format_version_is_refused(tmp_path):
    out = build(tmp_path, "run-a")
    manifest_path = out / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_format_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetError, match="unsupported dataset_format_version"):
        BCDataset.load(out)


@pytest.mark.parametrize("missing", [MANIFEST_FILENAME, DECISIONS_FILENAME])
def test_an_incomplete_dataset_directory_is_refused(tmp_path, missing):
    out = build(tmp_path, "run-a")
    (out / missing).unlink()

    with pytest.raises(DatasetError, match="has no"):
        BCDataset.load(out)


def test_an_empty_decisions_file_is_refused(tmp_path):
    out = build(tmp_path, "run-a")
    (out / DECISIONS_FILENAME).write_bytes(gzip.compress(b""))

    with pytest.raises(DatasetError, match="empty"):
        BCDataset.load(out)


def test_a_corrupt_decision_line_names_its_line_number(tmp_path):
    out = build(tmp_path, "run-a")
    (out / DECISIONS_FILENAME).write_bytes(gzip.compress(b'{"run_id": "run-a"}\n'))

    with pytest.raises(DatasetError, match="line 1"):
        BCDataset.load(out)


def test_the_holdout_is_split_by_run(tmp_path):
    dataset = BCDataset.load(build(tmp_path, "run-a", "run-b", "run-c", "run-d"))

    train, holdout = dataset.split_by_run(0.25)

    assert dataset.run_ids() == ("run-a", "run-b", "run-c", "run-d")
    assert train.run_ids() == ("run-a", "run-b", "run-c")
    assert holdout.run_ids() == ("run-d",)
    assert len(train) + len(holdout) == len(dataset)
    # A run is never split across the boundary: that is the whole point.
    assert not set(train.run_ids()) & set(holdout.run_ids())


def test_a_single_run_cannot_be_split_honestly_and_says_so(tmp_path):
    """One run has no holdout, and a number that looks like one is worse."""
    dataset = BCDataset.load(build(tmp_path, "run-a", decisions_per_run=10))

    with pytest.warns(UserWarning, match="NOT a measure of generalization"):
        train, holdout = dataset.split_by_run(0.2)

    assert len(train) == 8
    assert len(holdout) == 2
    assert train.run_ids() == holdout.run_ids() == ("run-a",)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_an_impossible_holdout_fraction_is_refused(tmp_path, fraction):
    dataset = BCDataset.load(build(tmp_path, "run-a"))

    with pytest.raises(ValueError, match="between 0 and 1"):
        dataset.split_by_run(fraction)


def test_a_subset_keeps_the_tokenizer_and_the_provider(tmp_path):
    provider = LegalActionProvider()
    dataset = BCDataset.load(build(tmp_path, "run-a", "run-b"), provider=provider)

    train, holdout = dataset.split_by_run(0.5)

    assert train.provider is provider
    assert holdout.tokenizer is dataset.tokenizer


def test_verify_names_a_decision_the_tokenizer_refuses(tmp_path):
    """A decision the encoder cannot accept is reported, not raised.

    ``verify`` exists to survey a whole dataset, so one bad decision must not
    stop it from reporting the rest.
    """
    dataset = BCDataset.load(build(tmp_path, "run-a"))
    unresolvable = Decision(
        run_id="run-a",
        step_index=41,
        state_type="rewards",
        act=1,
        floor=3,
        # No reward carries index 99, so the action cannot resolve to an entity.
        raw_state=rewards_state(),
        player_detail={"deck": {"cards": []}},
        candidates=({"type": "claim_reward", "index": 99},),
        expert_index=0,
        next_step_index=None,
        steps_to_run_end=0,
        is_run_final_decision=True,
    )
    surveyed = BCDataset(
        (*dataset.decisions, unresolvable),
        dataset.manifest,
        tokenizer=dataset.tokenizer,
        provider=dataset.provider,
    )

    failures = surveyed.verify()

    assert len(failures) == 1
    assert (failures[0].run_id, failures[0].step_index) == ("run-a", 41)
    assert "index" in failures[0].error
