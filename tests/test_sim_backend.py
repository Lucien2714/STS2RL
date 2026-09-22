"""The simulator backend: reset, environment wiring, configuration, and resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from sts2rl.env import ResetSpec, SimResetController
from sts2rl.env.game_env import GameEnv
from sts2rl.training import TrainingConfig, TrainingPlan
from sts2rl.training import cli


class FakeSimClient:
    """Answers ``sim/reset`` the way STS2Simulator does, and nothing else."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.state = {
            "state_type": "map",
            "run": {"act": 1, "floor": 0, "ascension": 0},
            "player": {"character": "The Ironclad", "hp": 80},
        }

    def sim_reset(self, **body) -> dict:
        self.requests.append(body)
        return {"status": "ok", "seed": body["seed"], "state": self.state}

    def menu_select(self, option, seed=None):  # pragma: no cover - must never run
        raise AssertionError("the simulator has no menus to navigate")


def test_reset_starts_the_run_in_one_request():
    client = FakeSimClient()

    state = SimResetController(client).reset(
        ResetSpec(character=1, run_seed="ABC1234567", ascension=3, sim_mode="gauntlet")
    )

    assert state == client.state
    assert client.requests == [
        {
            "character": "SILENT",
            "seed": "ABC1234567",
            "ascension": 3,
            "mode": "gauntlet",
            "max_fights": 12,
            "start_act": 1,
            "capture": False,
            "start_boss": False,
        }
    ]


def test_reset_can_start_from_a_later_act_and_capture():
    """The seed still names the run; the act says where in it to begin."""
    client = FakeSimClient()

    SimResetController(client).reset(
        ResetSpec(run_seed="ABC1234567", sim_start_act=2, sim_capture=True)
    )

    assert client.requests[0]["seed"] == "ABC1234567"
    assert client.requests[0]["start_act"] == 2
    assert client.requests[0]["capture"] is True


def test_reset_can_start_at_an_acts_boss():
    client = FakeSimClient()

    SimResetController(client).reset(ResetSpec(run_seed="ABC1234567", sim_start_boss=True))

    assert client.requests[0]["start_act"] == 1
    assert client.requests[0]["start_boss"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sim_start_act": 0},
        {"sim_start_act": True},
        {"sim_start_act": 2, "sim_mode": "gauntlet"},
        {"sim_capture": True, "sim_mode": "gauntlet"},
        {"sim_start_boss": True, "sim_mode": "gauntlet"},
    ],
)
def test_reset_spec_rejects_an_impossible_start(kwargs):
    # A gauntlet has no map, so there is no act to begin or capture.
    with pytest.raises(ValueError):
        ResetSpec(**kwargs)


def test_a_later_start_act_needs_the_simulator(tmp_path: Path):
    # A real client starts where its menu starts; only the simulator restores.
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--seed-pool", "ABC1234567",
            "--game-mode", "custom",
            "--sim-start-act", "2",
        ]
    )
    with pytest.raises(ValueError, match="need --backend sim"):
        cli._new_plan(args)


def test_a_boss_start_needs_the_simulator(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--seed-pool", "ABC1234567",
            "--game-mode", "custom",
            "--sim-start-boss",
        ]
    )
    with pytest.raises(ValueError, match="need --backend sim"):
        cli._new_plan(args)


def test_new_plan_carries_the_start_act(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--backend", "sim",
            "--seed-pool", "ABC1234567",
            "--sim-start-act", "2",
        ]
    )
    assert cli._new_plan(args).reset.sim_start_act == 2


def test_new_plan_carries_the_boss_start(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--backend", "sim",
            "--seed-pool", "ABC1234567",
            "--sim-start-boss",
        ]
    )
    reset = cli._new_plan(args).reset
    assert (reset.sim_start_act, reset.sim_start_boss) == (1, True)


def test_reset_never_reuses_a_run():
    """A reset always starts the seed it was asked for, so nothing is reused.

    The real controller can join a run already in progress, which silently
    measures a different seed.  The simulator cannot, and the flag says so.
    """
    controller = SimResetController(FakeSimClient())
    controller.reset(ResetSpec(run_seed="ABC1234567"))

    assert controller.reused_active_run is False


def test_reset_refuses_without_a_seed():
    with pytest.raises(ValueError, match="explicit seed"):
        SimResetController(FakeSimClient()).reset(ResetSpec())


def test_env_selects_the_controller_from_the_backend():
    client = FakeSimClient()

    sim = GameEnv(client=client, backend="sim")
    game = GameEnv(client=client)

    assert isinstance(sim.reset_controller, SimResetController)
    assert not isinstance(game.reset_controller, SimResetController)


def test_env_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        GameEnv(client=FakeSimClient(), backend="emulator")


def test_config_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        TrainingConfig(backend="emulator")


def test_sim_backend_accepts_a_seed_pool_without_the_custom_run_screen():
    """The custom-run screen is a menu, and the simulator has none."""
    plan = TrainingPlan(
        training=TrainingConfig(backend="sim", training_seeds=("ABC1234567",)),
        reset=ResetSpec(game_mode="standard"),
    )

    assert plan.training.backend == "sim"


def test_sim_backend_requires_seeds():
    with pytest.raises(ValueError, match="needs seeds"):
        TrainingPlan(training=TrainingConfig(backend="sim"))


def test_new_plan_defaults_the_action_delay_away_for_the_simulator(tmp_path: Path):
    """The delay paces a real client's next request; the simulator has settled."""
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--backend", "sim", "--seed-pool", "default"]
    )

    plan = cli._new_plan(args)

    assert plan.training.backend == "sim"
    assert plan.training.action_delay_seconds == 0.0


def test_new_plan_keeps_an_explicit_action_delay_for_the_simulator(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sim",
            "--seed-pool",
            "default",
            "--action-delay",
            "0.05",
        ]
    )

    assert cli._new_plan(args).training.action_delay_seconds == 0.05


def test_new_plan_carries_the_simulator_mode(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--backend",
            "sim",
            "--seed-pool",
            "default",
            "--sim-mode",
            "gauntlet",
            "--sim-max-fights",
            "5",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.reset.sim_mode == "gauntlet"
    assert plan.reset.sim_max_fights == 5


@pytest.mark.parametrize(
    "option,value",
    [("--backend", "game"), ("--sim-mode", "run"), ("--sim-max-fights", "9")],
)
def test_backend_cannot_change_when_resuming(
    tmp_path: Path, option: str, value: str
) -> None:
    """Switching environment mid-run makes the metrics describe two experiments."""
    saved = TrainingPlan(
        training=TrainingConfig(
            backend="sim", run_dir=tmp_path / "run", training_seeds=("ABC1234567",)
        ),
        reset=ResetSpec(sim_mode="gauntlet", sim_max_fights=5),
    )
    loaded = type("Loaded", (), {"plan": saved})()
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--resume", option, value]
    )

    with pytest.raises(ValueError, match="cannot change when resuming"):
        cli._resumed_plan(args, loaded)  # type: ignore[arg-type]
