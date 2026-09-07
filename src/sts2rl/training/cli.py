"""Command-line construction and resume rules for structured PPO training."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import torch

from sts2rl.agents import CandidatePPOAgent, EpisodeRunner, PPOConfig
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.env import GameEnv, ResetSpec
from sts2rl.training.checkpoint import CheckpointManager, LoadedCheckpoint
from sts2rl.training.config import (
    DEFAULT_HOLDOUT_SEEDS,
    DEFAULT_SEED_POOL,
    TrainingConfig,
    TrainingPlan,
    TrainingState,
)
from sts2rl.training.metrics import EpisodeMetrics, TrainingMetricsWriter
from sts2rl.training.trainer import Trainer


T = TypeVar("T")


def create_parser() -> argparse.ArgumentParser:
    """Build the training CLI without touching the environment or filesystem."""
    parser = argparse.ArgumentParser(
        description="Train structured candidate PPO through a local STS2MCP server."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume", nargs="?", const="latest")
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
    vocabulary = GameVocabulary.from_bundled_data()
    manager = CheckpointManager(args.run_dir, vocabulary)
    loaded: LoadedCheckpoint | None = None
    if args.resume is None:
        plan = _new_plan(args)
        plan.training.validate_runtime_device()
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
    )
    if loaded is not None:
        manager.restore_agent(loaded, agent, plan)

    with TrainingMetricsWriter(
        plan.training.run_dir,
        tensorboard_enabled=plan.training.tensorboard_enabled,
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_flush_secs=plan.training.tensorboard_flush_secs,
        resume_step=resume_step,
    ) as metrics_writer:
        base_urls = plan.training.client_base_urls()
        with ExitStack() as clients:
            runners = [
                EpisodeRunner(
                    clients.enter_context(
                        GameEnv(
                            base_url=base_url,
                            timeout=plan.training.timeout,
                            action_delay_seconds=(
                                plan.training.action_delay_seconds
                            ),
                        )
                    ),
                    # Each client gets its own lane so the agent keeps their
                    # trajectories -- and therefore their advantages -- apart.
                    agent.lane_view(lane),
                    max_steps=plan.training.max_steps_per_episode,
                    max_state_refreshes=plan.training.max_state_refreshes,
                )
                for lane, base_url in enumerate(base_urls)
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


def _new_plan(args: argparse.Namespace) -> TrainingPlan:
    training_defaults = TrainingConfig()
    encoder_defaults = EncoderConfig()
    ppo_defaults = PPOConfig()
    reset_defaults = ResetSpec()
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
            base_url=_or_default(args.base_url, training_defaults.base_url),
            timeout=_or_default(args.timeout, training_defaults.timeout),
            action_delay_seconds=_or_default(
                args.action_delay, training_defaults.action_delay_seconds
            ),
            ports=_port_list(args.ports),
            training_seeds=_seed_list(args.seed_pool, DEFAULT_SEED_POOL),
            holdout_seeds=_seed_list(args.holdout_seeds, DEFAULT_HOLDOUT_SEEDS),
            device=_or_default(args.device, training_defaults.device),
            torch_seed=_or_default(args.torch_seed, training_defaults.torch_seed),
            run_dir=args.run_dir,
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
