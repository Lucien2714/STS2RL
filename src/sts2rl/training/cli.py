"""Command-line construction and resume rules for structured PPO training."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import torch

from sts2rl.agents import CandidatePPOAgent, EpisodeRunner, PPOConfig
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.env import GameEnv, ResetSpec
from sts2rl.env.game_env import BACKENDS
from sts2rl.search import CombatSearch, LeafEvaluator, MctsConfig, SearchCombatAgent, SearchDecisionRecorder
from sts2rl.training.bc import load_bc_encoder_state
from sts2rl.training.checkpoint import CheckpointManager, LoadedCheckpoint
from sts2rl.training.config import (
    DEFAULT_HOLDOUT_SEEDS,
    DEFAULT_SEED_POOL,
    TrainingConfig,
    TrainingPlan,
    TrainingState,
)
from sts2rl.training.metrics import EpisodeMetrics, TrainingMetricsWriter
from sts2rl.training.surgery import InitialWeights, load_initial_weights
from sts2rl.training.trainer import Trainer

DEFAULT_SEARCH_WEIGHTS = Path(__file__).resolve().parents[1] / "search" / "weights" / "act1_boss_step1b.json"


T = TypeVar("T")


def create_parser() -> argparse.ArgumentParser:
    """Build the training CLI without touching the environment or filesystem."""
    parser = argparse.ArgumentParser(
        description="Train structured candidate PPO through a local STS2MCP server."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Optimizer updates between checkpoints (not episodes).",
    )
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument(
        "--init-encoder",
        type=Path,
        help=(
            "start a NEW run from a behavior-cloning artifact's encoder "
            "weights (sts2rl-bc-train). Only the weights transfer: the "
            "optimizer state belongs to a different objective and the counters "
            "to a run that never happened. Cannot be combined with --resume."
        ),
    )
    parser.add_argument(
        "--init-from",
        help=(
            "start a NEW run from another run's whole model: actor, critic and "
            "return scale, from a run directory (its latest checkpoint) or a "
            "checkpoint file. The optimizer and counters start fresh, so the PPO "
            "settings may differ. A checkpoint from an older vocabulary or schema "
            "goes through `sts2rl-surgery migrate` first. Cannot be combined with "
            "--resume or --init-encoder."
        ),
    )
    parser.add_argument(
        "--init-optimizer",
        action="store_true",
        default=None,
        help=(
            "with --init-from: also carry the source's optimizer moments (same model "
            "only), so the restart does not jolt the policy the way a fresh Adam does"
        ),
    )
    parser.add_argument(
        "--search-combat",
        action="store_true",
        default=None,
        help=(
            "step 2 of docs/mcts: the combat search plays fights on each lane's "
            "simulator, PPO learns everything else (needs --backend sim)"
        ),
    )
    parser.add_argument(
        "--search-rooms",
        help="comma-separated rooms whose fights are searched (default monster,elite,boss)",
    )
    parser.add_argument("--search-simulations", type=int, help="simulations per searched decision (default 50)")
    parser.add_argument("--search-elite-simulations", type=int, help="simulations per decision in elite fights (default: --search-simulations)")
    parser.add_argument("--search-boss-simulations", type=int, help="simulations per decision in boss fights (default: --search-simulations)")
    parser.add_argument("--search-depth", type=int, help="turns the search tree reaches (default 2)")
    parser.add_argument("--search-weights", help="leaf evaluator weights (JSON); default: the step 1-b fit")
    parser.add_argument(
        "--search-fights-out-of-rollout",
        action="store_true",
        default=None,
        help=(
            "with --search-combat: fold each searched fight into the transition between "
            "the macro decisions around it, so the rollout holds only PPO's own choices"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help=(
            "game (default) plays through real STS2MCP clients; sim plays "
            "through STS2Simulator, which starts every run from a seed."
        ),
    )
    parser.add_argument(
        "--sim-mode",
        choices=("run", "gauntlet"),
        help=(
            "simulator only: run follows the seed's map act by act, gauntlet "
            "plays hallway fights with no map."
        ),
    )
    parser.add_argument(
        "--sim-max-fights",
        type=int,
        help="simulator gauntlet only: fights won before the episode truncates.",
    )
    parser.add_argument(
        "--sim-start-act",
        type=int,
        help=(
            "simulator run mode only: start each episode at this act, restored "
            "from the simulator's snapshot of the seed's run (sts2sim --snapshots). "
            "The seed pools must name seeds the library holds that act for."
        ),
    )
    parser.add_argument(
        "--sim-start-boss",
        action="store_const",
        const=True,
        help=(
            "simulator run mode only: start each episode at the boss fight of "
            "--sim-start-act (default act 1), restored from the save the game "
            "takes on entering the boss room."
        ),
    )
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--ports",
        help=(
            "Comma-separated STS2MCP ports, one game client each, played in "
            "parallel. Omitted uses the single client at --base-url."
        ),
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        help="Seconds to pause after each accepted action (0 disables).",
    )
    parser.add_argument("--device")
    parser.add_argument("--torch-seed", type=int)
    parser.add_argument("--character", type=int)
    parser.add_argument("--game-mode", choices=("standard", "custom", "daily"))
    parser.add_argument("--run-seed")
    parser.add_argument(
        "--seed-pool",
        help=(
            "Comma-separated run seeds cycled one per episode, or 'default' "
            "for the bundled pool. Requires --game-mode custom."
        ),
    )
    parser.add_argument(
        "--holdout-seeds",
        help=(
            "Comma-separated seeds reserved for evaluation and never trained "
            "on, or 'default' for the bundled set."
        ),
    )
    parser.add_argument(
        "--modifiers",
        help=(
            "Comma-separated custom-run modifier keys to tick; empty string "
            "means every modifier off. Custom runs reconcile either way."
        ),
    )
    parser.add_argument("--start-run-option", choices=("confirm", "embark"))
    parser.add_argument("--allow-active-run", action="store_true", default=None)
    parser.add_argument(
        "--ascension",
        type=int,
        help="Ascension level to start runs at; omitted leaves the menu as-is.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-state-refreshes", type=int)
    parser.add_argument(
        "--max-episode-failures",
        type=int,
        help="Consecutive failed episodes before a client is given up on.",
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--entity-layers", type=int)
    parser.add_argument("--entity-heads", type=int)
    parser.add_argument("--entity-ff-dim", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-ratio", type=float)
    parser.add_argument("--value-coefficient", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--rollout-size", type=int)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument(
        "--target-kl",
        type=float,
        help="stop an update's epochs once the policy's approximate KL from the rollout policy passes 1.5x this (default: no limit)",
    )
    parser.add_argument("--no-tensorboard", action="store_true", default=None)
    parser.add_argument("--tensorboard-flush-secs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run training, and return a process exit status."""
    args = create_parser().parse_args(argv)
    try:
        return run_training(args)
    except KeyboardInterrupt:
        return 130


def run_training(args: argparse.Namespace) -> int:
    """Build either a new or resumed runtime and train to its target."""
    init_encoder = getattr(args, "init_encoder", None)
    init_from = getattr(args, "init_from", None)
    if init_encoder is not None and args.resume is not None:
        raise ValueError(
            "--init-encoder starts a new run from cloned weights; --resume "
            "continues one that already has its own. Pass one or the other."
        )
    if init_from is not None and args.resume is not None:
        raise ValueError(
            "--init-from starts a new run from another run's weights; --resume "
            "continues one that already has its own. Pass one or the other."
        )
    initial_weights: InitialWeights | None = None

    vocabulary = GameVocabulary.from_bundled_data()
    manager = CheckpointManager(args.run_dir, vocabulary)
    loaded: LoadedCheckpoint | None = None
    if args.resume is None:
        plan = _new_plan(args)
        plan.training.validate_runtime_device()
        if init_from is not None:
            # Before the run directory exists, so an incompatible source leaves
            # nothing half-built behind.
            initial_weights = load_initial_weights(
                init_from, vocabulary=vocabulary, encoder_config=plan.encoder
            )
            # The file, not the directory: a run's latest checkpoint moves on.
            plan = replace(
                plan,
                training=replace(plan.training, init_from=str(initial_weights.source)),
            )
        torch.manual_seed(plan.training.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(plan.training.torch_seed)
        manager.initialize_run(plan, resume=False)
        state = TrainingState()
        resume_step = None
        tensorboard_log_dir = "tensorboard"
    else:
        loaded = manager.load(args.resume, map_location="cpu")
        plan = _resumed_plan(args, loaded)
        plan.training.validate_runtime_device()
        manager.initialize_run(plan, resume=True)
        state = loaded.training_state
        resume_step = state.environment_steps
        tensorboard_log_dir = loaded.tensorboard_log_dir

    if plan.training.total_episodes < state.completed_episodes:
        raise ValueError(
            "total_episodes is below the checkpoint completed episode count"
        )

    tokenizer = GameTokenizer(vocabulary)
    encoder = GameEncoder(vocabulary, plan.encoder)
    agent = CandidatePPOAgent(
        tokenizer=tokenizer,
        game_encoder=encoder,
        config=plan.ppo,
        device=plan.training.device,
        # Fights folded out of the rollout: an update must not train the decision
        # before a fight that is still running. Set here, not only on the first
        # unrecorded step, so each lane's first fight is covered too.
        hold_open_steps=(
            plan.training.search_combat and not plan.training.search_fights_in_rollout
        ),
    )
    if loaded is not None:
        manager.restore_agent(loaded, agent, plan)
    elif init_encoder is not None:
        # Weights only, and into a fresh agent: the critic head comes along
        # untrained because behavior cloning never touched it, which is
        # deliberate -- an expert's returns sit at a scale an early policy
        # never reaches, and _ReturnScale would relearn the divisor underneath
        # a critic calibrated to the old one.
        encoder.load_state_dict(
            load_bc_encoder_state(
                init_encoder,
                vocabulary=vocabulary,
                encoder_config=plan.encoder,
            )
        )
        encoder.to(agent.device)
    elif initial_weights is not None:
        agent.initialize_from(
            initial_weights.encoder,
            initial_weights.return_scale,
            initial_weights.optimizer if plan.training.init_optimizer else None,
        )

    with TrainingMetricsWriter(
        plan.training.run_dir,
        tensorboard_enabled=plan.training.tensorboard_enabled,
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_flush_secs=plan.training.tensorboard_flush_secs,
        resume_step=resume_step,
    ) as metrics_writer:
        base_urls = plan.training.client_base_urls()
        with ExitStack() as clients:
            envs = [
                clients.enter_context(
                    GameEnv(
                        base_url=base_url,
                        timeout=plan.training.timeout,
                        action_delay_seconds=plan.training.action_delay_seconds,
                        backend=plan.training.backend,
                    )
                )
                for base_url in base_urls
            ]
            recorder = (
                SearchDecisionRecorder(plan.training.run_dir / "search" / "decisions.jsonl.gz")
                if plan.training.search_combat
                else None
            )
            runners = [
                EpisodeRunner(
                    env,
                    # Each client gets its own lane so the agent keeps their
                    # trajectories -- and therefore their advantages -- apart.
                    _lane_agent(agent, lane, env, plan, recorder, state.completed_episodes),
                    max_steps=plan.training.max_steps_per_episode,
                    max_state_refreshes=plan.training.max_state_refreshes,
                )
                for lane, env in enumerate(envs)
            ]
            trainer = Trainer(
                runners,
                agent,
                manager,
                metrics_writer,
                plan,
                state,
                reporter=_report_episode,
                tensorboard_log_dir=tensorboard_log_dir,
                max_episode_failures=plan.training.max_episode_failures,
            )
            trainer.train()
    return 0


def _lane_agent(agent, lane: int, env: GameEnv, plan: TrainingPlan, recorder, start: int = 0):
    """The lane's agent: PPO alone, or PPO with the combat search playing fights."""
    view = agent.lane_view(lane)
    if not plan.training.search_combat:
        return view
    weights_path = (
        Path(plan.training.search_weights)
        if plan.training.search_weights
        else DEFAULT_SEARCH_WEIGHTS
    )
    search = CombatSearch(
        MctsConfig(
            simulations=plan.training.search_simulations,
            turn_depth=plan.training.search_turn_depth,
        ),
        evaluator=LeafEvaluator(json.loads(weights_path.read_text(encoding="utf-8"))),
        rng=random.Random(plan.training.torch_seed * 1000 + lane),
    )
    return SearchCombatAgent(
        view,
        env,
        search,
        rooms=frozenset(plan.training.search_rooms),
        recorder=recorder,
        # Episode labels restart in every process; the run's completed count at
        # start keeps a resumed run's labels apart from the ones already recorded.
        run_label=f"s{start}-lane{lane}",
        record_fights=plan.training.search_fights_in_rollout,
        room_simulations=plan.training.search_room_simulations(),
    )


def _new_plan(args: argparse.Namespace) -> TrainingPlan:
    training_defaults = TrainingConfig()
    encoder_defaults = EncoderConfig()
    ppo_defaults = PPOConfig()
    reset_defaults = ResetSpec()
    backend = _or_default(args.backend, training_defaults.backend)
    # The delay paces the *next* request while a real client is still resolving
    # the last one.  The simulator answers only once the game has settled, so
    # there is nothing to pace and the default would be pure cost.
    action_delay = _or_default(
        args.action_delay,
        0.0 if backend == "sim" else training_defaults.action_delay_seconds,
    )
    return TrainingPlan(
        training=TrainingConfig(
            total_episodes=_or_default(
                args.total_episodes, training_defaults.total_episodes
            ),
            checkpoint_every=_or_default(
                args.checkpoint_every, training_defaults.checkpoint_every
            ),
            max_steps_per_episode=_or_default(
                args.max_steps, training_defaults.max_steps_per_episode
            ),
            max_state_refreshes=_or_default(
                args.max_state_refreshes, training_defaults.max_state_refreshes
            ),
            max_episode_failures=_or_default(
                args.max_episode_failures, training_defaults.max_episode_failures
            ),
            backend=backend,
            base_url=_or_default(args.base_url, training_defaults.base_url),
            timeout=_or_default(args.timeout, training_defaults.timeout),
            action_delay_seconds=action_delay,
            ports=_port_list(args.ports),
            training_seeds=_seed_list(args.seed_pool, DEFAULT_SEED_POOL),
            holdout_seeds=_seed_list(args.holdout_seeds, DEFAULT_HOLDOUT_SEEDS),
            device=_or_default(args.device, training_defaults.device),
            torch_seed=_or_default(args.torch_seed, training_defaults.torch_seed),
            run_dir=args.run_dir,
            init_encoder=(
                None
                if getattr(args, "init_encoder", None) is None
                else str(args.init_encoder)
            ),
            init_from=getattr(args, "init_from", None),
            init_optimizer=bool(getattr(args, "init_optimizer", None)),
            search_combat=bool(args.search_combat),
            search_rooms=(
                tuple(room.strip() for room in args.search_rooms.split(",") if room.strip())
                if args.search_rooms
                else training_defaults.search_rooms
            ),
            search_simulations=_or_default(args.search_simulations, training_defaults.search_simulations),
            search_turn_depth=_or_default(args.search_depth, training_defaults.search_turn_depth),
            search_elite_simulations=args.search_elite_simulations,
            search_boss_simulations=args.search_boss_simulations,
            search_weights=args.search_weights,
            search_fights_in_rollout=not args.search_fights_out_of_rollout,
            tensorboard_enabled=(
                training_defaults.tensorboard_enabled
                if args.no_tensorboard is None
                else not args.no_tensorboard
            ),
            tensorboard_flush_secs=_or_default(
                args.tensorboard_flush_secs,
                training_defaults.tensorboard_flush_secs,
            ),
        ),
        encoder=EncoderConfig(
            hidden_dim=_or_default(args.hidden_dim, encoder_defaults.hidden_dim),
            entity_layers=_or_default(
                args.entity_layers, encoder_defaults.entity_layers
            ),
            entity_heads=_or_default(args.entity_heads, encoder_defaults.entity_heads),
            entity_ff_dim=_or_default(
                args.entity_ff_dim, encoder_defaults.entity_ff_dim
            ),
        ),
        ppo=PPOConfig(
            learning_rate=_or_default(args.learning_rate, ppo_defaults.learning_rate),
            gamma=_or_default(args.gamma, ppo_defaults.gamma),
            gae_lambda=_or_default(args.gae_lambda, ppo_defaults.gae_lambda),
            clip_ratio=_or_default(args.clip_ratio, ppo_defaults.clip_ratio),
            value_coefficient=_or_default(
                args.value_coefficient, ppo_defaults.value_coefficient
            ),
            entropy_coefficient=_or_default(
                args.entropy_coefficient, ppo_defaults.entropy_coefficient
            ),
            max_grad_norm=_or_default(args.max_grad_norm, ppo_defaults.max_grad_norm),
            rollout_size=_or_default(args.rollout_size, ppo_defaults.rollout_size),
            update_epochs=_or_default(args.update_epochs, ppo_defaults.update_epochs),
            target_kl=_or_default(args.target_kl, ppo_defaults.target_kl),
        ),
        reset=ResetSpec(
            character=_or_default(args.character, reset_defaults.character),
            game_mode=_or_default(args.game_mode, reset_defaults.game_mode),
            run_seed=args.run_seed,
            start_run_option=_or_default(
                args.start_run_option, reset_defaults.start_run_option
            ),
            allow_active_run=(
                reset_defaults.allow_active_run
                if args.allow_active_run is None
                else args.allow_active_run
            ),
            ascension=_or_default(args.ascension, reset_defaults.ascension),
            modifiers=_seed_list(args.modifiers, ()),
            sim_mode=_or_default(args.sim_mode, reset_defaults.sim_mode),
            sim_max_fights=_or_default(
                args.sim_max_fights, reset_defaults.sim_max_fights
            ),
            sim_start_act=_or_default(
                args.sim_start_act, reset_defaults.sim_start_act
            ),
            sim_start_boss=_or_default(
                args.sim_start_boss, reset_defaults.sim_start_boss
            ),
        ),
    )


def _resumed_plan(
    args: argparse.Namespace,
    loaded: LoadedCheckpoint,
) -> TrainingPlan:
    saved = loaded.plan
    _require_equal_overrides(
        args,
        {
            "torch_seed": saved.training.torch_seed,
            "max_steps": saved.training.max_steps_per_episode,
            "max_state_refreshes": saved.training.max_state_refreshes,
            "hidden_dim": saved.encoder.hidden_dim,
            "entity_layers": saved.encoder.entity_layers,
            "entity_heads": saved.encoder.entity_heads,
            "entity_ff_dim": saved.encoder.entity_ff_dim,
            "learning_rate": saved.ppo.learning_rate,
            "gamma": saved.ppo.gamma,
            "gae_lambda": saved.ppo.gae_lambda,
            "clip_ratio": saved.ppo.clip_ratio,
            "value_coefficient": saved.ppo.value_coefficient,
            "entropy_coefficient": saved.ppo.entropy_coefficient,
            "max_grad_norm": saved.ppo.max_grad_norm,
            "rollout_size": saved.ppo.rollout_size,
            "update_epochs": saved.ppo.update_epochs,
            "target_kl": saved.ppo.target_kl,
            "backend": saved.training.backend,
            "search_combat": saved.training.search_combat,
            "search_simulations": saved.training.search_simulations,
            "search_depth": saved.training.search_turn_depth,
            "search_elite_simulations": saved.training.search_elite_simulations,
            "search_boss_simulations": saved.training.search_boss_simulations,
            "search_weights": saved.training.search_weights,
            "search_fights_out_of_rollout": (
                None if saved.training.search_fights_in_rollout else True
            ),
            "sim_mode": saved.reset.sim_mode,
            "sim_max_fights": saved.reset.sim_max_fights,
            "sim_start_act": saved.reset.sim_start_act,
            "sim_start_boss": saved.reset.sim_start_boss,
            "character": saved.reset.character,
            "game_mode": saved.reset.game_mode,
            "run_seed": saved.reset.run_seed,
            "start_run_option": saved.reset.start_run_option,
            "allow_active_run": saved.reset.allow_active_run,
            "ascension": saved.reset.ascension,
        },
    )
    _require_equal_overrides(
        args,
        {
            "seed_pool": ",".join(saved.training.training_seeds),
            "ports": ",".join(str(port) for port in saved.training.ports),
            "holdout_seeds": ",".join(saved.training.holdout_seeds),
            "modifiers": ",".join(saved.reset.modifiers),
            "search_rooms": ",".join(saved.training.search_rooms),
        },
        normalize=lambda value: ",".join(_seed_list(value, ()) or ()),
    )
    if (
        args.no_tensorboard is not None
        and (not args.no_tensorboard) != saved.training.tensorboard_enabled
    ):
        raise ValueError("--no-tensorboard cannot change when resuming")
    training = replace(
        saved.training,
        total_episodes=_or_default(args.total_episodes, saved.training.total_episodes),
        checkpoint_every=_or_default(
            args.checkpoint_every, saved.training.checkpoint_every
        ),
        base_url=_or_default(args.base_url, saved.training.base_url),
        timeout=_or_default(args.timeout, saved.training.timeout),
        action_delay_seconds=_or_default(
            args.action_delay, saved.training.action_delay_seconds
        ),
        device=_or_default(args.device, saved.training.device),
        run_dir=args.run_dir,
        tensorboard_flush_secs=_or_default(
            args.tensorboard_flush_secs,
            saved.training.tensorboard_flush_secs,
        ),
    )
    return replace(saved, training=training)


def _require_equal_overrides(
    args: argparse.Namespace,
    expected: dict[str, object],
    normalize: Callable[[object], object] | None = None,
) -> None:
    for name, expected_value in expected.items():
        value = getattr(args, name)
        if value is not None and normalize is not None:
            value = normalize(value)
        if value is not None and value != expected_value:
            option = "--" + name.replace("_", "-")
            raise ValueError(
                f"{option} cannot change when resuming: "
                f"checkpoint={expected_value!r}, requested={value!r}"
            )


def _report_episode(metrics: EpisodeMetrics) -> None:
    print(
        f"episode={metrics.episode} steps={metrics.environment_steps} "
        f"reward={metrics.reward:.3f} floor={metrics.floor} "
        f"length={metrics.steps} action_errors={metrics.action_errors}"
    )


def _or_default(value: T | None, default: T) -> T:
    return default if value is None else value


def _port_list(value: object) -> tuple[int, ...]:
    """Parse the comma-separated client ports."""
    if value is None:
        return ()
    ports = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ports.append(int(part))
        except ValueError:
            raise ValueError(f"--ports expects numbers, got {part!r}") from None
    return tuple(ports)


def _seed_list(value: object, bundled: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated CLI list, with 'default' naming the bundled set.

    An empty string is a real answer -- "none of them" -- and is kept distinct
    from the flag being absent, which leaves the default in place.
    """
    if value is None:
        return ()
    text = str(value).strip()
    if text == "default":
        return bundled
    return tuple(part.strip() for part in text.split(",") if part.strip())
