"""Training CLI parsing and resume compatibility tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sts2rl.agents import PPOConfig
from sts2rl.encoder import EncoderConfig, GameEncoder, GameVocabulary
from sts2rl.env import GameObservation, ResetSpec
from sts2rl.training import TrainingConfig, TrainingPlan
from sts2rl.training.bc import BCConfig, EpochMetrics, save_bc_checkpoint
from sts2rl.training.config import DEFAULT_HOLDOUT_SEEDS, DEFAULT_SEED_POOL
from sts2rl.training import cli
from sts2rl.training import eval_cli


def test_help_exits_without_starting_training(monkeypatch: pytest.MonkeyPatch):
    called = False

    def fail_if_called(args: argparse.Namespace) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli, "run_training", fail_if_called)

    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    assert not called


def test_new_plan_maps_cli_options_and_can_disable_tensorboard(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--total-episodes",
            "12",
            "--checkpoint-every",
            "3",
            "--hidden-dim",
            "32",
            "--entity-heads",
            "4",
            "--gamma",
            "0.9",
            "--no-tensorboard",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.total_episodes == 12
    assert plan.training.checkpoint_every == 3
    assert not plan.training.tensorboard_enabled
    assert plan.encoder.hidden_dim == 32
    assert plan.ppo.gamma == 0.9


def test_resume_uses_checkpoint_plan_and_allows_runtime_overrides(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(
            total_episodes=10,
            checkpoint_every=2,
            run_dir=tmp_path / "run",
        )
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "latest",
            "--total-episodes",
            "25",
            "--checkpoint-every",
            "5",
            "--timeout",
            "40",
        ]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.total_episodes == 25
    assert resumed.training.checkpoint_every == 5
    assert resumed.training.timeout == 40
    assert resumed.encoder == saved.encoder
    assert resumed.ppo == saved.ppo


def test_resume_rejects_model_defining_override(tmp_path: Path):
    saved = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "latest",
            "--gamma",
            "0.5",
        ]
    )

    with pytest.raises(ValueError, match="--gamma cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_main_returns_training_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(cli, "run_training", lambda args: 7)

    result = cli.main(["--run-dir", str(tmp_path / "run")])

    assert result == 7


def test_main_maps_keyboard_interrupt_to_exit_130(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def interrupt(args: argparse.Namespace) -> int:
        del args
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_training", interrupt)

    assert cli.main(["--run-dir", str(tmp_path / "run")]) == 130


def test_action_delay_is_configurable_on_a_new_plan(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--action-delay", "0.25"]
    )

    plan = cli._new_plan(args)

    assert plan.training.action_delay_seconds == 0.25


def test_action_delay_can_be_disabled_on_resume(tmp_path: Path):
    """Zero is a real setting, so it must survive _or_default."""
    saved = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--resume", "latest", "--action-delay", "0"]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.action_delay_seconds == 0.0


def test_seed_pool_and_modifiers_come_from_the_command_line(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--game-mode", "custom",
            "--seed-pool", "AAA,BBB",
            "--holdout-seeds", "ZZZ",
            "--modifiers", "MIDAS",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.training_seeds == ("AAA", "BBB")
    assert plan.training.holdout_seeds == ("ZZZ",)
    assert plan.reset.modifiers == ("MIDAS",)


def test_default_installs_the_bundled_pools(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--game-mode", "custom",
            "--seed-pool", "default",
            "--holdout-seeds", "default",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.training_seeds == DEFAULT_SEED_POOL
    assert plan.training.holdout_seeds == DEFAULT_HOLDOUT_SEEDS
    assert not set(plan.training.training_seeds) & set(plan.training.holdout_seeds)
    # Size is the dial between memorizing a map and spending the variance
    # budget on draw luck, so a pool that silently shrank back to a dozen is
    # worth failing over.
    assert len(plan.training.training_seeds) >= 100
    assert len(plan.training.holdout_seeds) >= 20


def test_resume_refuses_to_reshuffle_the_seed_pool(tmp_path: Path):
    """Changing the pool would remap every seed onto a different episode."""
    saved = TrainingPlan(
        training=TrainingConfig(
            run_dir=tmp_path / "run", training_seeds=("AAA", "BBB")
        ),
        reset=ResetSpec(game_mode="custom"),
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--seed-pool", "AAA,CCC",
        ]
    )

    with pytest.raises(ValueError, match="--seed-pool cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_resume_accepts_the_pool_it_was_saved_with(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(
            run_dir=tmp_path / "run", training_seeds=("AAA", "BBB")
        ),
        reset=ResetSpec(game_mode="custom"),
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--seed-pool", "AAA,BBB",
        ]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.training_seeds == ("AAA", "BBB")


def test_ports_come_from_the_command_line(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--ports", "15526, 15527"]
    )

    plan = cli._new_plan(args)

    assert plan.training.ports == (15526, 15527)


def test_resume_refuses_to_change_the_client_set(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(run_dir=tmp_path / "run", ports=(15526, 15527))
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--ports", "15526",
        ]
    )

    with pytest.raises(ValueError, match="--ports cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_the_evaluation_cli_defaults_to_both_pools(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert args.pools == "both"
    assert args.checkpoint == "latest"


def test_the_evaluation_cli_takes_ports_and_a_checkpoint(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--checkpoint", "episode_000500.pt",
            "--ports", "15526,15527",
            "--episodes-per-seed", "5",
        ]
    )

    assert args.checkpoint == "episode_000500.pt"
    assert args.episodes_per_seed == 5
    assert args.ports == "15526,15527"


def test_the_evaluation_cli_builds_one_client_per_port(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--ports", "15526,15527,15528"]
    )
    plan = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))

    urls = eval_cli._client_base_urls(args, plan)

    assert urls == (
        "http://localhost:15526/api/v1",
        "http://localhost:15527/api/v1",
        "http://localhost:15528/api/v1",
    )


def test_init_encoder_and_resume_are_mutually_exclusive(tmp_path: Path):
    """One starts from cloned weights, the other continues its own."""
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
        ]
    )

    with pytest.raises(ValueError, match="Pass one or the other"):
        cli.run_training(args)


def test_init_encoder_defaults_to_absent(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert args.init_encoder is None


def test_init_encoder_loads_only_the_encoder_weights(tmp_path: Path):
    """The artifact carries no optimizer state and no counters to restore."""
    from sts2rl.encoder import EncoderConfig, GameEncoder, GameVocabulary
    from sts2rl.training.bc import BCConfig, EpochMetrics, save_bc_checkpoint

    vocabulary = GameVocabulary.from_bundled_data()
    cloned = GameEncoder(vocabulary, EncoderConfig())
    with torch.no_grad():
        for parameter in cloned.parameters():
            parameter.add_(0.25)
    artifact = save_bc_checkpoint(
        tmp_path / "bc_best.pt",
        encoder=cloned,
        vocabulary_fingerprint=vocabulary.fingerprint(),
        encoder_config=EncoderConfig(),
        bc_config=BCConfig(),
        metrics=EpochMetrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.9,
            holdout_loss=0.6,
            holdout_accuracy=0.4,
            holdout_chance=0.2,
            holdout_first_candidate=0.3,
        ),
    )

    fresh = GameEncoder(vocabulary, EncoderConfig())
    fresh.load_state_dict(
        cli.load_bc_encoder_state(
            artifact, vocabulary=vocabulary, encoder_config=EncoderConfig()
        )
    )

    for name, tensor in cloned.state_dict().items():
        assert torch.allclose(fresh.state_dict()[name], tensor)


def test_init_encoder_is_applied_before_any_client_is_contacted(tmp_path: Path):
    """The call site is wired, and it runs before the network.

    Loading weights after a client is contacted would leave a half-built run
    behind when the artifact turns out to be incompatible.
    """
    from sts2rl.encoder import EncoderConfig

    seen: dict[str, object] = {}

    class Stop(RuntimeError):
        pass

    def capture(path, *, vocabulary, encoder_config):
        seen["path"] = path
        seen["fingerprint"] = vocabulary.fingerprint()
        seen["encoder_config"] = encoder_config
        raise Stop

    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
            "--hidden-dim",
            "32",
            "--entity-heads",
            "2",
        ]
    )
    original = cli.load_bc_encoder_state
    cli.load_bc_encoder_state = capture
    try:
        with pytest.raises(Stop):
            cli.run_training(args)
    finally:
        cli.load_bc_encoder_state = original

    assert seen["path"] == tmp_path / "bc_best.pt"
    assert seen["encoder_config"] == EncoderConfig(hidden_dim=32, entity_heads=2)


def test_a_run_started_from_cloned_weights_says_so_in_its_config(tmp_path: Path):
    """Two runs identical in every other field are different experiments."""
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.init_encoder == str(tmp_path / "bc_best.pt")
    assert TrainingPlan.from_dict(plan.to_dict()).training.init_encoder == str(
        tmp_path / "bc_best.pt"
    )


def test_a_run_from_random_weights_records_none(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert cli._new_plan(args).training.init_encoder is None


def test_a_config_saved_before_the_field_existed_still_loads(tmp_path: Path):
    """Every run directory written so far lacks init_encoder."""
    values = TrainingConfig(run_dir=tmp_path).to_dict()
    del values["init_encoder"]

    assert TrainingConfig.from_dict(values).init_encoder is None


def test_an_empty_init_encoder_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="init_encoder"):
        TrainingConfig(run_dir=tmp_path, init_encoder="")


def test_search_options_reach_a_new_plan(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--backend", "sim", "--seed-pool", "AAAA", "--search-combat",
         "--search-rooms", "elite,boss", "--search-simulations", "80", "--search-depth", "1"]
    )
    training = cli._new_plan(args).training
    assert training.search_combat is True
    assert training.search_rooms == ("elite", "boss")
    assert (training.search_simulations, training.search_turn_depth) == (80, 1)


def test_room_budgets_reach_a_new_plan(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--backend", "sim", "--seed-pool", "AAAA", "--search-combat",
         "--search-elite-simulations", "200", "--search-boss-simulations", "300"]
    )
    training = cli._new_plan(args).training
    assert training.search_room_simulations() == {"elite": 200, "boss": 300}
    assert training.search_simulations == 50


def test_search_is_off_unless_asked(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])
    assert cli._new_plan(args).training.search_combat is False


def test_search_episode_labels_differ_after_a_resume(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--backend", "sim", "--seed-pool", "AAAA", "--search-combat"]
    )
    plan = cli._new_plan(args)

    class Agent:
        def lane_view(self, lane):
            return object()

    first = cli._lane_agent(Agent(), 2, env=object(), plan=plan, recorder=None, start=0)
    resumed = cli._lane_agent(Agent(), 2, env=object(), plan=plan, recorder=None, start=1000)
    assert first.run_label != resumed.run_label



def test_the_kl_limit_reaches_a_new_plan_and_is_off_by_default(tmp_path: Path):
    parser = cli.create_parser()
    assert cli._new_plan(parser.parse_args(["--run-dir", str(tmp_path / "a")])).ppo.target_kl is None
    plan = cli._new_plan(parser.parse_args(["--run-dir", str(tmp_path / "b"), "--target-kl", "0.02"]))
    assert plan.ppo.target_kl == 0.02


def test_searched_fights_can_be_kept_out_of_the_rollout(tmp_path: Path):
    parser = cli.create_parser()
    base = ["--backend", "sim", "--seed-pool", "AAAA", "--search-combat"]
    assert cli._new_plan(parser.parse_args(["--run-dir", str(tmp_path / "a"), *base])).training.search_fights_in_rollout
    out = cli._new_plan(parser.parse_args(["--run-dir", str(tmp_path / "b"), *base, "--search-fights-out-of-rollout"]))
    assert out.training.search_fights_in_rollout is False


def test_exploration_rates_come_from_the_command_line(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--explore", "rest_site=0.3, card_reward=0.3,map=0.15"]
    )

    plan = cli._new_plan(args)

    assert plan.ppo.exploration == (("card_reward", 0.3), ("map", 0.15), ("rest_site", 0.3))


def test_exploration_is_off_unless_asked(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])
    assert cli._new_plan(args).ppo.exploration == ()


@pytest.mark.parametrize("value", ["rest_site", "rest_site=lots", "rest_site:0.3"])
def test_a_malformed_exploration_rate_names_the_flag(tmp_path: Path, value: str):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--explore", value])
    with pytest.raises(ValueError, match="--explore"):
        cli._new_plan(args)


def test_an_unknown_screen_in_the_exploration_rates_is_refused(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--explore", "lobby=0.3"])
    with pytest.raises(ValueError, match="unknown state type"):
        cli._new_plan(args)


def test_resume_inherits_the_exploration_rates(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(run_dir=tmp_path / "run"),
        ppo=PPOConfig(exploration={"map": 0.15}),
    )
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--resume", "latest"])

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.ppo.exploration == (("map", 0.15),)


def test_resume_refuses_a_changed_exploration_rate(tmp_path: Path):
    """A different rate is a different experiment, like a different seed pool."""
    saved = TrainingPlan(
        training=TrainingConfig(run_dir=tmp_path / "run"),
        ppo=PPOConfig(exploration={"map": 0.15}),
    )
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--resume", "latest", "--explore", "map=0.3"]
    )

    with pytest.raises(ValueError, match="--explore cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_resume_accepts_the_exploration_rates_it_was_saved_with_in_any_order(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(run_dir=tmp_path / "run"),
        ppo=PPOConfig(exploration={"map": 0.15, "rest_site": 0.3}),
    )
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--resume", "latest", "--explore", "rest_site=0.3,map=0.15"]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.ppo == saved.ppo


# --------------------------------------------------------------- reference KL

SMALL_ENCODER = EncoderConfig(hidden_dim=16, entity_heads=4, entity_ff_dim=32)
SMALL_ENCODER_FLAGS = ["--hidden-dim", "16", "--entity-heads", "4", "--entity-ff-dim", "32"]


def _artifact(tmp_path: Path, config: EncoderConfig = SMALL_ENCODER, name: str = "bc_best.pt") -> Path:
    """A BC artifact with its own random weights, of the given encoder shape."""
    vocabulary = GameVocabulary.from_bundled_data()
    return save_bc_checkpoint(
        tmp_path / name,
        encoder=GameEncoder(vocabulary, config),
        vocabulary_fingerprint=vocabulary.fingerprint(),
        encoder_config=config,
        bc_config=BCConfig(),
        metrics=EpochMetrics(
            epoch=1, train_loss=0.5, train_accuracy=0.9, holdout_loss=0.6,
            holdout_accuracy=0.4, holdout_chance=0.2, holdout_first_candidate=0.3,
        ),
    )


def _capture_trainer(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace ``Trainer`` with one that records what run_training built and
    saves a single checkpoint, so a resume has something to read."""
    captured: list = []

    class CapturingTrainer:
        def __init__(self, runners, agent, manager, metrics_writer, plan, state, **kwargs):
            self.agent, self.manager, self.plan, self.state = agent, manager, plan, state
            captured.append(self)

        def train(self):
            self.manager.save_progress(self.agent, self.plan, self.state, "tensorboard")
            return self.state

    monkeypatch.setattr(cli, "Trainer", CapturingTrainer)
    return captured


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rest_site() -> dict:
    return {
        "state_type": "rest_site",
        "run": {"floor": 6},
        "player": {
            "character": "The Ironclad", "hp": 40, "max_hp": 80, "gold": 50,
            "relics": [], "potions": [], "status": [],
        },
        "rest_site": {
            "options": [{"index": 0, "id": "REST"}, {"index": 1, "id": "SMITH"}],
            "can_proceed": False,
        },
    }


def test_a_reference_needs_both_the_path_and_the_coefficient(tmp_path: Path):
    parse = cli.create_parser().parse_args
    with pytest.raises(ValueError, match="go together"):
        cli._new_plan(parse(["--run-dir", str(tmp_path / "a"), "--reference-kl", "0.1"]))
    with pytest.raises(ValueError, match="go together"):
        cli._new_plan(parse(["--run-dir", str(tmp_path / "b"), "--reference-policy", "bc_best.pt"]))

    plan = cli._new_plan(
        parse(["--run-dir", str(tmp_path / "c"), "--reference-policy", "bc_best.pt", "--reference-kl", "0.1"])
    )

    assert plan.ppo.reference_kl_coefficient == 0.1
    assert plan.training.reference_policy == "bc_best.pt"
    assert plan.training.reference_policy_sha256 is None  # Recorded once the copy exists.
    plain = cli._new_plan(parse(["--run-dir", str(tmp_path / "d")]))
    assert plain.ppo.reference_kl_coefficient == 0.0 and plain.training.reference_policy is None


def test_a_new_run_copies_its_reference_and_records_the_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The source may move or be retrained; the run reads its own copy."""
    artifact = _artifact(tmp_path)
    captured = _capture_trainer(monkeypatch)
    run = tmp_path / "run"
    args = cli.create_parser().parse_args(
        ["--run-dir", str(run), "--no-tensorboard", *SMALL_ENCODER_FLAGS,
         "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    )

    assert cli.run_training(args) == 0

    copy = run / "reference_policy.pt"
    assert copy.read_bytes() == artifact.read_bytes()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["training"]["reference_policy"] == str(artifact)
    assert config["training"]["reference_policy_sha256"] == _sha256(artifact)
    assert config["ppo"]["reference_kl_coefficient"] == 0.1
    agent = captured[0].agent
    assert agent.config.reference_kl_coefficient == 0.1
    expected = torch.load(artifact, map_location="cpu", weights_only=True)["encoder"]
    for name, tensor in agent.reference_encoder.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    assert not any(p.requires_grad for p in agent.reference_encoder.parameters())
    assert not agent.reference_encoder.training
    # The actor started from random weights: the reference is not --init-encoder.
    assert not torch.equal(
        next(iter(agent.game_encoder.state_dict().values())), next(iter(expected.values()))
    )


def test_init_encoder_and_the_reference_may_be_the_same_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact = _artifact(tmp_path)
    captured = _capture_trainer(monkeypatch)
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--no-tensorboard", *SMALL_ENCODER_FLAGS,
         "--init-encoder", str(artifact), "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    )

    assert cli.run_training(args) == 0

    agent = captured[0].agent
    assert agent.game_encoder is not agent.reference_encoder
    for name, tensor in agent.game_encoder.state_dict().items():
        assert torch.equal(tensor, agent.reference_encoder.state_dict()[name]), name
    assert all(p.requires_grad for p in agent.game_encoder.parameters())


def test_without_a_coefficient_no_reference_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = _capture_trainer(monkeypatch)
    monkeypatch.setattr(
        cli, "load_reference_encoder", lambda *args, **kwargs: pytest.fail("a reference was loaded")
    )
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--no-tensorboard", *SMALL_ENCODER_FLAGS]
    )

    assert cli.run_training(args) == 0

    assert captured[0].agent.reference_encoder is None
    assert not (tmp_path / "run" / "reference_policy.pt").exists()


def test_an_incompatible_reference_leaves_no_run_directory_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _capture_trainer(monkeypatch)
    artifact = _artifact(tmp_path, EncoderConfig(hidden_dim=64), name="wide.pt")
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--no-tensorboard", *SMALL_ENCODER_FLAGS,
         "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    )

    with pytest.raises(ValueError, match="encoder config"):
        cli.run_training(args)

    assert not (tmp_path / "run").exists()


def test_resume_reads_the_copy_and_checks_its_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact = _artifact(tmp_path)
    captured = _capture_trainer(monkeypatch)
    run = tmp_path / "run"
    parse = cli.create_parser().parse_args
    cli.run_training(parse(
        ["--run-dir", str(run), "--no-tensorboard", *SMALL_ENCODER_FLAGS,
         "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    ))
    # The source is gone; the run does not need it any more.
    artifact.unlink()

    assert cli.run_training(parse(["--run-dir", str(run), "--resume"])) == 0
    resumed = captured[1]
    assert resumed.plan.training.reference_policy == str(artifact)
    assert resumed.agent.reference_encoder is not None
    for name, tensor in resumed.agent.reference_encoder.state_dict().items():
        assert torch.equal(tensor, captured[0].agent.reference_encoder.state_dict()[name]), name
    # The same values are accepted; different ones are a different experiment.
    assert cli.run_training(parse(
        ["--run-dir", str(run), "--resume", "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    )) == 0
    with pytest.raises(ValueError, match="--reference-kl cannot change"):
        cli.run_training(parse(["--run-dir", str(run), "--resume", "--reference-kl", "0.2"]))
    with pytest.raises(ValueError, match="--reference-policy cannot change"):
        cli.run_training(parse(["--run-dir", str(run), "--resume", "--reference-policy", str(tmp_path / "other.pt")]))

    copy = run / "reference_policy.pt"
    copy.write_bytes(copy.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        cli.run_training(parse(["--run-dir", str(run), "--resume"]))
    copy.unlink()
    with pytest.raises(ValueError, match="missing"):
        cli.run_training(parse(["--run-dir", str(run), "--resume"]))


def test_resume_refuses_a_changed_reference_before_touching_anything(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(
            run_dir=tmp_path / "run", reference_policy="bc_best.pt", reference_policy_sha256="0" * 64
        ),
        ppo=PPOConfig(reference_kl_coefficient=0.1),
    )
    parse = cli.create_parser().parse_args
    base = ["--run-dir", str(tmp_path / "run"), "--resume", "latest"]

    with pytest.raises(ValueError, match="--reference-kl cannot change"):
        cli._resumed_plan(parse([*base, "--reference-kl", "0.05"]), SimpleNamespace(plan=saved))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="--reference-policy cannot change"):
        cli._resumed_plan(parse([*base, "--reference-policy", "other.pt"]), SimpleNamespace(plan=saved))  # type: ignore[arg-type]
    resumed = cli._resumed_plan(
        parse([*base, "--reference-policy", "bc_best.pt", "--reference-kl", "0.1"]), SimpleNamespace(plan=saved)  # type: ignore[arg-type]
    )
    assert resumed.ppo == saved.ppo
    assert resumed.training.reference_policy_sha256 == "0" * 64


def test_evaluation_builds_the_agent_without_the_reference_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """sts2rl-eval reads the plan of a run trained with a reference, whose copy
    may be gone; evaluation never samples, so it never needs it."""
    artifact = _artifact(tmp_path)
    _capture_trainer(monkeypatch)
    run = tmp_path / "run"
    cli.run_training(cli.create_parser().parse_args(
        ["--run-dir", str(run), "--no-tensorboard", *SMALL_ENCODER_FLAGS,
         "--reference-policy", str(artifact), "--reference-kl", "0.1"]
    ))
    (run / "reference_policy.pt").unlink()
    chosen: list[dict] = []

    class FakeEvaluator:
        def __init__(self, runners, agent, reset, **kwargs):
            self.agent = agent

        def run(self, schedule):
            self.agent.eval()
            chosen.append(self.agent.choose_action(GameObservation(_rest_site())).to_dict())
            assert self.agent.config.reference_kl_coefficient == 0.1
            assert self.agent.reference_encoder is None
            return []

    monkeypatch.setattr(eval_cli, "Evaluator", FakeEvaluator)
    args = eval_cli.create_parser().parse_args(["--run-dir", str(run), "--holdout-seeds", "X"])

    assert eval_cli.run_evaluation(args) == 0

    assert chosen and chosen[0]["type"] == "choose_rest_option"
