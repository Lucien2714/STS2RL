"""Behavior cloning: the actor learns the label, and nothing else travels."""

import json

import pytest
import torch

from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.offline.clean import DECISIONS_FILENAME, clean_records
from sts2rl.offline.dataset import BCDataset
from sts2rl.training.bc import (
    BC_FORMAT_VERSION,
    BEST_CHECKPOINT_FILENAME,
    METRICS_FILENAME,
    BCConfig,
    BCTrainer,
    EpochMetrics,
    baseline_scores,
    load_bc_encoder_state,
    save_bc_checkpoint,
    split_dataset,
)


def rewards_state(floor: int = 3, items: int = 3) -> dict:
    kinds = ["gold", "relic", "potion"]
    return {
        "state_type": "rewards",
        "run": {"act": 1, "floor": floor},
        "player": {"character": "The Ironclad", "potions": [], "relics": []},
        "rewards": {
            "items": [
                {"index": i, "type": kinds[i % len(kinds)]} for i in range(items)
            ],
            "can_proceed": True,
        },
    }


def build_dataset(tmp_path, *run_ids: str, decisions_per_run: int = 4):
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    for offset, run_id in enumerate(run_ids):
        steps = [
            {
                "index": index,
                "state": rewards_state(floor=index + 1),
                "player_detail": {"deck": {"cards": []}},
                # A label that depends only on the floor, so a working trainer
                # can reach it and a broken one cannot.
                "action": {
                    "action": "claim_reward",
                    "args": {"index": (index + offset) % 3},
                },
            }
            for index in range(decisions_per_run)
        ]
        steps.append(
            {
                "index": decisions_per_run,
                "state": rewards_state(floor=99),
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
    out = tmp_path / f"dataset-{'-'.join(run_ids)}"
    clean_records(records, out)
    return BCDataset.load(out)


def make_trainer(tmp_path, dataset, **overrides):
    config = BCConfig(**{"epochs": 3, "minibatch_size": 4, "patience": 3, **overrides})
    train, holdout = split_dataset(dataset, config)
    vocabulary = dataset.tokenizer.vocabulary
    return (
        BCTrainer(
            GameEncoder(vocabulary, EncoderConfig()),
            train,
            holdout,
            config=config,
            out_dir=tmp_path / "bc-out",
        ),
        train,
        holdout,
    )


def test_cloning_drives_the_training_loss_down(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=6)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=8, patience=8)

    result = trainer.run()

    assert result.epochs_run == 8
    assert result.history[-1].train_loss < result.history[0].train_loss


def test_a_trainer_with_no_training_decisions_is_refused(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset)
    trainer.train_set = trainer.train_set._subset([])

    with pytest.raises(ValueError, match="at least one training decision"):
        trainer.run()


def test_every_epoch_is_written_to_metrics_jsonl(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=3, patience=3)

    trainer.run()

    lines = (
        (tmp_path / "bc-out" / METRICS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = [json.loads(line) for line in lines]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    assert all(record["type"] == "bc_epoch" for record in records)
    assert all("holdout_chance" in record for record in records)


def test_rerunning_replaces_the_metrics_rather_than_appending(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=2, patience=2)

    trainer.run()
    trainer.run()

    lines = (
        (tmp_path / "bc-out" / METRICS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 2


def test_training_stops_when_the_holdout_stops_improving(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=50, patience=1)

    result = trainer.run()

    # Patience 1 stops at the first epoch that does not beat the best, so a
    # 50-epoch budget must not be spent.
    assert result.epochs_run < 50
    assert result.best_epoch <= result.epochs_run


def test_the_baselines_are_the_bar_a_top_one_number_is_read_against(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", decisions_per_run=6)

    chance, first = baseline_scores(dataset)

    # Three candidates per decision, and the label cycles 0, 1, 2.
    assert chance == pytest.approx(1 / 3)
    assert first == pytest.approx(
        sum(1 for d in dataset.decisions if d.expert_index == 0) / len(dataset)
    )


def test_an_empty_split_has_no_baseline_rather_than_a_divide_by_zero(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", decisions_per_run=4)

    assert baseline_scores(dataset._subset([])) == (0.0, 0.0)


def test_the_artifact_carries_weights_and_nothing_else(tmp_path):
    """Adam's state belongs to a different objective; the counters to no run."""
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=2, patience=2)

    trainer.run()
    payload = torch.load(
        tmp_path / "bc-out" / BEST_CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=True,
    )

    assert payload["format_version"] == BC_FORMAT_VERSION
    assert set(payload["encoder"]) == set(trainer.encoder.state_dict())
    assert "optimizer" not in payload
    assert "training_state" not in payload
    assert payload["dataset"]["run_ids"] == ["run-a", "run-b"]


def test_cloned_weights_load_back_into_a_matching_encoder(tmp_path):
    """One epoch, so the best weights and the final weights are the same ones."""
    dataset = build_dataset(tmp_path, "run-a", "run-b", decisions_per_run=4)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=1, patience=1)
    trainer.run()
    vocabulary = dataset.tokenizer.vocabulary

    state = load_bc_encoder_state(
        tmp_path / "bc-out" / BEST_CHECKPOINT_FILENAME,
        vocabulary=vocabulary,
        encoder_config=EncoderConfig(),
    )
    fresh = GameEncoder(vocabulary, EncoderConfig())
    fresh.load_state_dict(state)

    for name, tensor in trainer.encoder.state_dict().items():
        assert torch.allclose(fresh.state_dict()[name], tensor.cpu())


def test_the_artifact_holds_the_best_epoch_not_the_last(tmp_path):
    """Early stopping only means something if the artifact is the best one.

    Training accuracy keeps climbing after the holdout has turned, so saving
    the final weights would hand PPO the most overfitted encoder of the run.
    """
    dataset = build_dataset(tmp_path, "run-a", "run-b", "run-c", decisions_per_run=6)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=6, patience=6)

    result = trainer.run()
    payload = torch.load(
        tmp_path / "bc-out" / BEST_CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=True,
    )

    assert payload["metrics"]["epoch"] == result.best_epoch
    assert payload["metrics"]["holdout_accuracy"] == pytest.approx(
        result.best_holdout_accuracy
    )


def epoch_metrics() -> EpochMetrics:
    return EpochMetrics(
        epoch=1,
        train_loss=1.0,
        train_accuracy=0.5,
        holdout_loss=1.0,
        holdout_accuracy=0.5,
        holdout_chance=0.3,
        holdout_first_candidate=0.3,
    )


def write_artifact(tmp_path, *, encoder_config: EncoderConfig, fingerprint: str):
    vocabulary = GameVocabulary.from_bundled_data()
    return save_bc_checkpoint(
        tmp_path / "artifact.pt",
        encoder=GameEncoder(vocabulary, encoder_config),
        vocabulary_fingerprint=fingerprint,
        encoder_config=encoder_config,
        bc_config=BCConfig(),
        metrics=epoch_metrics(),
    )


def test_an_artifact_from_a_different_vocabulary_is_refused(tmp_path):
    """Rerouted embedding rows make the same tensors mean something else."""
    path = write_artifact(
        tmp_path, encoder_config=EncoderConfig(), fingerprint="not-the-bundled-one"
    )

    with pytest.raises(ValueError, match="fingerprint does not match"):
        load_bc_encoder_state(
            path,
            vocabulary=GameVocabulary.from_bundled_data(),
            encoder_config=EncoderConfig(),
        )


def test_an_artifact_from_a_different_encoder_shape_is_refused(tmp_path):
    vocabulary = GameVocabulary.from_bundled_data()
    path = write_artifact(
        tmp_path,
        encoder_config=EncoderConfig(hidden_dim=64),
        fingerprint=vocabulary.fingerprint(),
    )

    with pytest.raises(ValueError, match="encoder config"):
        load_bc_encoder_state(
            path, vocabulary=vocabulary, encoder_config=EncoderConfig()
        )


def test_an_artifact_from_a_future_format_is_refused(tmp_path):
    vocabulary = GameVocabulary.from_bundled_data()
    path = write_artifact(
        tmp_path, encoder_config=EncoderConfig(), fingerprint=vocabulary.fingerprint()
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = BC_FORMAT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported BC format_version"):
        load_bc_encoder_state(
            path, vocabulary=vocabulary, encoder_config=EncoderConfig()
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"epochs": 0}, "epochs must be a positive integer"),
        ({"minibatch_size": 0}, "minibatch_size must be a positive integer"),
        ({"patience": 0}, "patience must be a positive integer"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"max_grad_norm": 0.0}, "max_grad_norm must be positive"),
        ({"weight_decay": -1.0}, "weight_decay must not be negative"),
        ({"holdout_fraction": 1.0}, "holdout_fraction must be between 0 and 1"),
        ({"holdout_fraction": 0.0}, "holdout_fraction must be between 0 and 1"),
    ],
)
def test_an_invalid_config_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        BCConfig(**overrides)


def test_the_split_is_by_run(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", "run-b", "run-c", "run-d")

    train, holdout = split_dataset(dataset, BCConfig(holdout_fraction=0.25))

    assert not set(train.run_ids()) & set(holdout.run_ids())
    assert len(train) + len(holdout) == len(dataset)


def test_a_single_run_split_warns_that_the_holdout_measures_nothing(tmp_path):
    dataset = build_dataset(tmp_path, "run-a", decisions_per_run=10)

    with pytest.warns(UserWarning, match="NOT a measure of generalization"):
        split_dataset(dataset, BCConfig())


def test_a_stale_dataset_never_reaches_the_trainer(tmp_path):
    """The labels are positions; a moved candidate set is a silent relabel."""
    import gzip

    build_dataset(tmp_path, "run-a", decisions_per_run=4)
    path = tmp_path / "dataset-run-a" / DECISIONS_FILENAME
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    first = json.loads(lines[0])
    first["candidates"] = list(reversed(first["candidates"]))
    lines[0] = json.dumps(first)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))

    from sts2rl.offline.dataset import StaleDatasetError

    with pytest.raises(StaleDatasetError):
        BCDataset.load(
            tmp_path / "dataset-run-a",
            tokenizer=GameTokenizer(GameVocabulary.from_bundled_data()),
        )


def test_the_summary_describes_the_saved_epoch_not_the_last(tmp_path):
    from sts2rl.training.bc import BCResult, summarize

    def epoch(number, accuracy, screen):
        return EpochMetrics(
            epoch=number,
            train_loss=1.0,
            train_accuracy=0.1 * number,
            holdout_loss=1.0,
            holdout_accuracy=accuracy,
            holdout_chance=0.2,
            holdout_first_candidate=0.3,
            accuracy_by_state_type={"monster": screen},
        )

    history = (epoch(1, 0.6, 0.11), epoch(2, 0.5, 0.22))
    result = BCResult(
        best_epoch=1,
        best_holdout_accuracy=0.6,
        final_train_accuracy=0.2,
        epochs_run=2,
        checkpoint_path=None,
        history=history,
    )

    text = summarize(result)

    assert "monster        11.0%" in text
    assert "22.0%" not in text
    assert "train accuracy       10.0%" in text


def test_the_saved_epoch_is_the_one_with_the_lowest_holdout_loss(tmp_path):
    """Accuracy wanders in noise; the loss says when the policy got overconfident.

    PPO samples from this policy, so calibration is what transfers.
    """
    dataset = build_dataset(tmp_path, "run-a", "run-b", "run-c", decisions_per_run=6)
    trainer, _, _ = make_trainer(tmp_path, dataset, epochs=6, patience=6)

    result = trainer.run()

    losses = [epoch.holdout_loss for epoch in result.history]
    assert result.history[result.best_epoch - 1].holdout_loss == min(losses)
    assert result.best_holdout_accuracy == (
        result.history[result.best_epoch - 1].holdout_accuracy
    )
