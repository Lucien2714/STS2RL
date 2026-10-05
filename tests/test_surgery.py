"""Carrying trained weights across a vocabulary or schema change (training/surgery.py).

The claim worth testing is behavioural: after a migration the new model scores a
state exactly as the old model did. The old model cannot be built here -- its code
is gone, which is the whole problem -- but its behaviour can: a model that never
had a column computes what this model computes with that column's inputs zeroed.
So each test builds a model, removes something from it to make the "old" one, and
checks the migrated model against the original fed the stripped input.
"""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sts2rl.actions import GameAction
from sts2rl.agents.ppo import CandidatePPOAgent, PPOConfig
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.encoder.schema import ENTITY_CATEGORICAL_FIELDS, GLOBAL_NUMERIC_FIELDS
from sts2rl.encoder.spec import model_spec, schema_spec
from sts2rl.env import GameObservation
from sts2rl.training import cli
from sts2rl.training.checkpoint import CheckpointManager
from sts2rl.training.config import TrainingConfig, TrainingPlan, TrainingState
from sts2rl.training.surgery import (
    SurgeryError,
    compare_outputs,
    decision_outputs,
    embedding_tables,
    load_initial_weights,
    migrate_checkpoint,
    migrate_encoder_state,
    numeric_inputs,
)

CONFIG = EncoderConfig(hidden_dim=16, entity_heads=4, entity_ff_dim=32)


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


@pytest.fixture(scope="module")
def tokenizer(vocabulary: GameVocabulary) -> GameTokenizer:
    return GameTokenizer(vocabulary)


def _card(index: int, card_id: str) -> dict[str, object]:
    return {
        "index": index,
        "id": card_id,
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "target_type": "AnyEnemy",
        "can_play": True,
    }


def _combat_state() -> dict[str, object]:
    return {
        "state_type": "monster",
        "player": {
            "character": "The Ironclad",
            "hp": 61,
            "max_hp": 80,
            "block": 3,
            "gold": 137,
            "energy": 3,
            "max_energy": 3,
            "relics": [],
            "potions": [],
            "hand": [_card(0, "UPPERCUT"), _card(1, "ABRASIVE")],
            "status": [{"id": "STRENGTH_POWER", "amount": 2}],
        },
        "battle": {
            "enemies": [
                {
                    "entity_id": "JAW_WORM_0",
                    "name": "Jaw Worm",
                    "hp": 30,
                    "max_hp": 44,
                    "block": 5,
                    "status": [],
                    "intents": [],
                }
            ]
        },
    }


def _actions() -> list[GameAction]:
    return [
        GameAction("play_card", card_index=0, target="JAW_WORM_0"),
        GameAction("play_card", card_index=1, target="JAW_WORM_0"),
        GameAction("end_turn"),
    ]


def _decision(tokenizer: GameTokenizer):
    return tokenizer.tokenize_decision(GameObservation(_combat_state()), _actions())


def _encoder(vocabulary: GameVocabulary, seed: int = 7) -> GameEncoder:
    torch.manual_seed(seed)
    return GameEncoder(vocabulary, CONFIG).eval()


def _outputs(encoder: GameEncoder, decision) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = encoder.policy_value(decision)
    return output.logits, output.value


def _assert_same_outputs(a: GameEncoder, b: GameEncoder, decision_a, decision_b=None) -> None:
    logits_a, value_a = _outputs(a, decision_a)
    logits_b, value_b = _outputs(b, decision_a if decision_b is None else decision_b)
    assert torch.allclose(logits_a, logits_b, atol=1e-6), (logits_a, logits_b)
    assert torch.allclose(value_a, value_b, atol=1e-6), (value_a, value_b)


def _with_state(decision, **changes):
    return replace(decision, state=replace(decision.state, **changes))


# ------------------------------------------------------------------ the maps


def test_every_embedding_is_mapped_to_the_table_it_indexes(vocabulary):
    encoder = GameEncoder(vocabulary, CONFIG)
    tables = embedding_tables(schema_spec())
    state = encoder.state_dict()
    for name, table in tables.items():
        assert state[name].shape[0] == vocabulary.size(table), name
    embeddings = [m for m in encoder.modules() if isinstance(m, torch.nn.Embedding)]
    assert len(tables) == len(embeddings)


def test_every_numeric_projection_is_described_column_by_column(vocabulary):
    state = GameEncoder(vocabulary, CONFIG).state_dict()
    for name, columns in numeric_inputs(schema_spec()).items():
        assert state[name].shape[1] == len(columns), name
        assert len(set(columns)) == len(columns), name


# ------------------------------------------------------------------ migration


def test_an_unchanged_model_migrates_to_itself(vocabulary, tokenizer):
    old = _encoder(vocabulary)
    new = GameEncoder(vocabulary, CONFIG).eval()

    migration = migrate_encoder_state(old.state_dict(), model_spec(vocabulary), new, vocabulary)

    assert migration.output_preserving
    for name, tensor in old.state_dict().items():
        assert torch.equal(new.state_dict()[name], tensor), name
    assert all(not counts["moved"] and not counts["new"] for counts in migration.report["tables"].values())
    _assert_same_outputs(old, new, _decision(tokenizer))


def test_a_new_token_keeps_every_row_under_its_own_token(vocabulary, tokenizer):
    """The old vocabulary lacked UPPERCUT: every row after it sat one place earlier,
    and the old model read UPPERCUT as <unknown>, so that is what it keeps reading."""
    original = _encoder(vocabulary)
    tokens = list(vocabulary.table("cards").tokens)
    row = tokens.index("UPPERCUT")
    spec = copy.deepcopy(model_spec(vocabulary))
    spec["tables"]["cards"].remove("UPPERCUT")
    card_tables = [name for name, table in embedding_tables(schema_spec()).items() if table == "cards"]
    old_state = {
        name: (torch.cat([tensor[:row], tensor[row + 1 :]]) if name in card_tables else tensor)
        for name, tensor in original.state_dict().items()
    }
    migrated = GameEncoder(vocabulary, CONFIG).eval()

    migration = migrate_encoder_state(old_state, spec, migrated, vocabulary)

    assert migration.output_preserving
    for name in card_tables:
        weights = migrated.state_dict()[name]
        assert torch.equal(weights[row], old_state[name][1])  # <unknown>
        assert torch.equal(weights[row + 1 :], original.state_dict()[name][row + 1 :])
        assert migration.report["tables"][name]["new"] == 1
    # The old model on this state is the original with UPPERCUT read as <unknown>.
    decision = _decision(tokenizer)
    card = decision.state.entities["card"]
    column = ENTITY_CATEGORICAL_FIELDS["card"].index("card_id")
    categorical = card.categorical.clone()
    categorical[categorical[:, column] == row, column] = 1
    entities = dict(decision.state.entities, card=replace(card, categorical=categorical))
    _assert_same_outputs(migrated, original, decision, _with_state(decision, entities=entities))


def test_a_new_numeric_column_starts_at_zero_and_changes_nothing(vocabulary, tokenizer):
    original = _encoder(vocabulary)
    field = GLOBAL_NUMERIC_FIELDS.index("gold")
    count = len(GLOBAL_NUMERIC_FIELDS)
    spec = copy.deepcopy(model_spec(vocabulary))
    spec["schema"]["global_numeric"].remove("gold")
    name = "entity_encoder.global_numeric_projection.weight"
    keep = [i for i in range(2 * count) if i not in (field, count + field)]
    old_state = dict(original.state_dict())
    old_state[name] = old_state[name][:, keep]
    migrated = GameEncoder(vocabulary, CONFIG).eval()

    migration = migrate_encoder_state(old_state, spec, migrated, vocabulary)

    assert migration.output_preserving
    assert migration.report["columns"][name] == {"added": ["gold"], "removed": []}
    # The old model never saw gold: the original with gold's value and mask zeroed.
    decision = _decision(tokenizer)
    assert decision.state.global_numeric[field] != 0
    values = decision.state.global_numeric.clone()
    mask = decision.state.global_numeric_mask.clone()
    values[field], mask[field] = 0, False
    blind = _with_state(decision, global_numeric=values, global_numeric_mask=mask)
    _assert_same_outputs(migrated, original, decision, blind)
    logits, _ = _outputs(original, decision)
    assert not torch.allclose(logits, _outputs(original, blind)[0])  # gold did matter


def test_a_new_categorical_column_starts_as_a_zero_table(vocabulary, tokenizer):
    original = _encoder(vocabulary)
    spec = copy.deepcopy(model_spec(vocabulary))
    spec["schema"]["entity_categorical"]["card"] = [
        column for column in spec["schema"]["entity_categorical"]["card"] if column[0] != "rarity"
    ]
    name = next(
        key for key in embedding_tables(schema_spec()) if key.endswith(".card__rarity.weight")
    )
    old_state = {key: value for key, value in original.state_dict().items() if key != name}
    migrated = GameEncoder(vocabulary, CONFIG).eval()

    migration = migrate_encoder_state(old_state, spec, migrated, vocabulary)

    assert migration.output_preserving
    assert migration.report["zeroed"] == [name]
    decision = _decision(tokenizer)
    card = decision.state.entities["card"]
    column = ENTITY_CATEGORICAL_FIELDS["card"].index("rarity")
    categorical = card.categorical.clone()
    assert categorical[:, column].ne(0).all()
    categorical[:, column] = 0  # the padding row, zero in the original
    entities = dict(decision.state.entities, card=replace(card, categorical=categorical))
    _assert_same_outputs(migrated, original, decision, _with_state(decision, entities=entities))


def test_a_new_entity_kind_is_reported_as_changing_the_outputs(vocabulary):
    original = _encoder(vocabulary)
    spec = copy.deepcopy(model_spec(vocabulary))
    del spec["schema"]["entity_categorical"]["orb"]
    del spec["schema"]["entity_numeric"]["orb"]
    old_state = {
        key: value
        for key, value in original.state_dict().items()
        if ".orb__" not in key and "projections.orb." not in key
    }

    migration = migrate_encoder_state(old_state, spec, GameEncoder(vocabulary, CONFIG), vocabulary)

    assert not migration.output_preserving
    assert migration.report["new_entity_kinds"] == ["orb"]
    assert "entity_encoder.entity_numeric_projections.orb.weight" in migration.report["fresh"]


def test_a_shape_change_outside_the_inputs_is_refused(vocabulary):
    old_state = dict(_encoder(vocabulary).state_dict())
    old_state["value_head.0.weight"] = torch.zeros(3, 3)

    with pytest.raises(SurgeryError, match="value_head.0.weight changed shape"):
        migrate_encoder_state(old_state, model_spec(vocabulary), GameEncoder(vocabulary, CONFIG), vocabulary)


def test_a_vocabulary_only_spec_cannot_describe_a_schema_change(vocabulary):
    """The dumps scripts/migrate_vocabulary.py wrote carry no schema."""
    old_state = dict(_encoder(vocabulary).state_dict())
    name = "entity_encoder.global_numeric_projection.weight"
    old_state[name] = old_state[name][:, 2:]
    spec = {key: value for key, value in model_spec(vocabulary).items() if key != "schema"}

    with pytest.raises(SurgeryError, match="sts2rl-surgery spec"):
        migrate_encoder_state(old_state, spec, GameEncoder(vocabulary, CONFIG), vocabulary)


# ---------------------------------------------------------------- checkpoints


def _plan(run_dir: Path) -> TrainingPlan:
    return TrainingPlan(
        training=TrainingConfig(
            total_episodes=4, checkpoint_every=2, run_dir=run_dir, tensorboard_enabled=False
        ),
        encoder=CONFIG,
        ppo=PPOConfig(update_epochs=1),
    )


def _save_run(run_dir: Path, vocabulary: GameVocabulary) -> tuple[Path, CandidatePPOAgent]:
    plan = _plan(run_dir)
    agent = CandidatePPOAgent(GameTokenizer(vocabulary), _encoder(vocabulary), config=plan.ppo)
    agent._return_scale.load({"var": 9.0, "count": 50.0})
    manager = CheckpointManager(run_dir, vocabulary)
    manager.initialize_run(plan, resume=False)
    path = manager.save(
        "update_000004.pt",
        agent,
        plan,
        TrainingState(completed_episodes=2, environment_steps=25, optimizer_updates=4),
        "tensorboard",
    )
    return path, agent


def test_a_checkpoint_carries_its_own_spec(tmp_path, vocabulary):
    path, _ = _save_run(tmp_path / "run", vocabulary)

    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["model_spec"] == model_spec(vocabulary)


def _age(path: Path, vocabulary: GameVocabulary) -> int:
    """Rewrite a checkpoint as if trained before UPPERCUT was in the vocabulary."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    row = list(vocabulary.table("cards").tokens).index("UPPERCUT")
    spec = payload["model_spec"]
    spec["tables"]["cards"].remove("UPPERCUT")
    spec["fingerprint"] = payload["vocabulary_fingerprint"] = "old"
    encoder = payload["agent_state"]["encoder"]
    for name, table in embedding_tables(schema_spec()).items():
        if table == "cards":
            encoder[name] = torch.cat([encoder[name][:row], encoder[name][row + 1 :]])
    torch.save(payload, path)
    return row


def test_migrate_checkpoint_writes_a_loadable_run(tmp_path, vocabulary):
    path, agent = _save_run(tmp_path / "old", vocabulary)
    row = _age(path, vocabulary)

    saved, migration = migrate_checkpoint(tmp_path / "old", tmp_path / "new")

    loaded = CheckpointManager(tmp_path / "new", vocabulary).load("latest")
    assert saved.name == "update_000004.pt"
    assert loaded.training_state.optimizer_updates == 4  # the weights did take 4 updates
    assert loaded.plan.training.run_dir == tmp_path / "new"
    assert loaded.agent_state["return_scale"]["var"] == 9.0
    assert loaded.agent_state["optimizer"]["state"] == {}
    name = "entity_encoder.entity_embeddings.card__card_id.weight"
    weights = loaded.agent_state["encoder"][name]
    assert torch.equal(weights[row], agent.game_encoder.state_dict()[name][1])  # <unknown>
    report = json.loads((tmp_path / "new" / "migration.json").read_text(encoding="utf-8"))
    assert report["old_fingerprint"] == "old"
    assert migration.output_preserving


def test_a_checkpoint_without_a_spec_needs_one_given(tmp_path, vocabulary):
    path, _ = _save_run(tmp_path / "old", vocabulary)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["model_spec"]
    torch.save(payload, path)

    with pytest.raises(SurgeryError, match="--old-spec"):
        migrate_checkpoint(path, tmp_path / "new")
    saved, _ = migrate_checkpoint(path, tmp_path / "new", old_spec=model_spec(vocabulary))
    assert saved.is_file()


# ------------------------------------------------------------- --init-from


def test_a_run_directory_given_as_a_relative_path_resolves(tmp_path, vocabulary, monkeypatch):
    """``--init-from runs/x`` and ``outputs --from runs/x``: the latest checkpoint's path
    was relative, and loading it prefixed the checkpoint directory a second time."""
    path, _ = _save_run(tmp_path / "old", vocabulary)
    _dataset(tmp_path / "data")
    monkeypatch.chdir(tmp_path)

    weights = load_initial_weights("old", vocabulary=vocabulary, encoder_config=CONFIG)
    outputs = decision_outputs("old", "data")

    assert weights.source == path.resolve()
    assert len(outputs["rows"]) == 2


def test_initial_weights_are_the_model_without_its_optimizer(tmp_path, vocabulary):
    _, trained = _save_run(tmp_path / "old", vocabulary)

    weights = load_initial_weights(tmp_path / "old", vocabulary=vocabulary, encoder_config=CONFIG)
    agent = CandidatePPOAgent(
        GameTokenizer(vocabulary), GameEncoder(vocabulary, CONFIG), config=PPOConfig(learning_rate=1e-5)
    )
    agent.initialize_from(weights.encoder, weights.return_scale)

    assert weights.source.name == "update_000004.pt"
    for name, tensor in trained.game_encoder.state_dict().items():
        assert torch.equal(agent.game_encoder.state_dict()[name], tensor), name
    assert agent._return_scale.state() == trained._return_scale.state()
    assert agent.optimizer.state_dict()["state"] == {}
    assert agent.optimizer.param_groups[0]["lr"] == 1e-5


def test_initial_weights_refuse_another_width_or_vocabulary(tmp_path, vocabulary):
    path, _ = _save_run(tmp_path / "old", vocabulary)

    with pytest.raises(SurgeryError, match="encoder config"):
        load_initial_weights(path, vocabulary=vocabulary, encoder_config=EncoderConfig())
    _age(path, vocabulary)
    with pytest.raises(SurgeryError, match="sts2rl-surgery migrate"):
        load_initial_weights(path, vocabulary=vocabulary, encoder_config=CONFIG)


def test_init_from_and_resume_are_mutually_exclusive(tmp_path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--init-from", str(tmp_path / "old"), "--resume"]
    )
    with pytest.raises(ValueError, match="--init-from"):
        cli.run_training(args)


def test_a_run_starts_from_init_encoder_or_init_from_not_both(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        TrainingConfig(run_dir=tmp_path, init_encoder="a.pt", init_from="b.pt")


def test_a_bad_init_from_leaves_no_run_directory(tmp_path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--init-from", str(tmp_path / "missing"), "--no-tensorboard"]
    )
    with pytest.raises(SurgeryError):
        cli.run_training(args)
    assert not (tmp_path / "run").exists()


def test_a_run_started_from_another_records_the_checkpoint_file(tmp_path, vocabulary, monkeypatch):
    """The file, not the directory: the directory's latest checkpoint moves on."""
    path, _ = _save_run(tmp_path / "old", vocabulary)

    class Stop(RuntimeError):
        pass

    def no_client(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(cli, "GameEnv", no_client)
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--init-from", str(tmp_path / "old"),
            "--hidden-dim", "16", "--entity-heads", "4", "--entity-ff-dim", "32",
            "--no-tensorboard",
        ]
    )
    with pytest.raises(Stop):
        cli.run_training(args)

    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert Path(config["training"]["init_from"]) == path.resolve()


# --------------------------------------------------------------- output check


def _dataset(directory: Path) -> Path:
    directory.mkdir()
    lines = [
        {
            "run_id": "r1",
            "step_index": 0,
            "state_type": "monster",
            "raw_state": _combat_state(),
            "player_detail": None,
            "candidates": [action.to_dict() for action in _actions()],
        },
        {
            "run_id": "r1",
            "step_index": 1,
            "state_type": "monster",
            "raw_state": _combat_state(),
            "player_detail": None,
            "candidates": [GameAction("play_card", card_index=7, target="JAW_WORM_0").to_dict()],
        },
    ]
    with gzip.open(directory / "decisions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return directory


def test_a_model_scores_recorded_decisions_the_same_as_itself(tmp_path, vocabulary):
    _save_run(tmp_path / "old", vocabulary)
    dataset = _dataset(tmp_path / "data")

    outputs = decision_outputs(tmp_path / "old", dataset)

    assert [row["key"] for row in outputs["rows"]] == ["r1:0", "r1:1"]
    assert len(outputs["rows"][0]["logits"]) == 3
    assert "error" in outputs["rows"][1]  # a card the hand does not hold
    report = compare_outputs(outputs, outputs)
    assert report["total"] == {"decisions": 2, "same_argmax": 1, "identical": 1, "unscored": 1}


def test_compare_counts_what_moved(vocabulary):
    before = {"rows": [
        {"key": "a", "state_type": "map", "logits": [1.0, 0.0], "value": 0.5},
        {"key": "b", "state_type": "monster", "logits": [1.0, 0.0], "value": 0.5},
        {"key": "c", "state_type": "monster", "logits": [1.0, 0.0], "value": 0.5},
    ]}
    after = {"rows": [
        {"key": "a", "state_type": "map", "logits": [1.0, 0.0], "value": 0.5},
        {"key": "b", "state_type": "monster", "logits": [0.0, 1.0], "value": 0.5},
        {"key": "c", "state_type": "monster", "logits": [1.0, 0.0, 2.0], "value": 0.5},
    ]}

    report = compare_outputs(before, after)

    assert report["by_state_type"]["map"] == {"decisions": 1, "same_argmax": 1, "identical": 1}
    assert report["by_state_type"]["monster"] == {
        "decisions": 2, "same_argmax": 0, "candidates_differ": 1,
    }
    assert report["largest_differences"][0]["key"] == "b"
    assert report["max_logit_difference"] == 1.0
