"""Behavioral-cloning pretraining for candidate-scoring agents.

Records of human play (state, action) pairs are turned into a supervised signal:
for each decision state, the agent enumerates the legal action candidates, scores
them, and is trained with softmax cross-entropy to give the candidate the human
actually chose the highest probability.

This trainer is intentionally agent-agnostic. It relies only on the shared
candidate interface (``encode_state``, ``encode_action``, ``valid_action_candidates``,
``action_key``, ``bc_score``, ``save``), so it pretrains the battle agent or any of
the non-battle screen agents (map, reward, shop, rest, event) unchanged. Pass
``--screen <name>`` to pretrain a screen agent instead of the battle agent.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from sts2rl.agents.orchestrator import (
    SCREEN_NAMES,
    create_battle_agent,
    create_screen_agent,
    is_battle_policy_state,
    normalize_battle_agent_type,
    normalize_screen_agent_type,
    screen_name_for_state,
)
from sts2rl.checkpoints.manager import agent_latest_path, battle_latest_path
from sts2rl.data.recordings import iter_recordings

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """One precomputed supervised sample.

    ``state_action`` has shape ``(num_candidates, model_input_size)`` and is
    constant across epochs; only the model changes, so we encode once.
    """

    state_action: torch.Tensor
    target_index: int


DIAG_SAMPLE_CAP = 5  # how many mismatch examples to retain for debug logging


@dataclass
class BuildStats:
    """Counters describing how recordings mapped onto candidate sets."""

    total: int = 0
    matched: int = 0
    no_candidates: int = 0
    no_match: int = 0
    single_candidate: int = 0  # only one legal action (e.g. forced end_turn): nothing to learn
    # Up to DIAG_SAMPLE_CAP diagnostics each, for DEBUG-level inspection.
    no_candidates_samples: list = field(default_factory=list)
    no_match_samples: list = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.matched / self.total if self.total else 0.0


def build_examples(agent, samples) -> tuple[list[Example], BuildStats]:
    """Encode recordings into supervised examples for the given agent.

    Each recorded action must correspond to exactly one legal candidate (matched
    by ``action_key``). Samples with no candidates or no matching candidate are
    skipped and counted so the caller can report data quality.
    """
    examples: list[Example] = []
    stats = BuildStats()

    for raw_state, action in samples:
        stats.total += 1
        candidates = agent.valid_action_candidates(raw_state)
        if not candidates:
            stats.no_candidates += 1
            if len(stats.no_candidates_samples) < DIAG_SAMPLE_CAP:
                stats.no_candidates_samples.append(
                    {"state_type": raw_state.get("state_type"), "action": action.get("type")}
                )
            continue

        # A single legal candidate is not a decision: the softmax over one option is
        # always 1.0 (zero loss, zero gradient), so it teaches nothing and only
        # inflates accuracy/match-rate. Skip it, mirroring how the live battle flow
        # auto-advances forced end_turn states (flow.battle_flow.is_forced_end_turn_state).
        if len(candidates) <= 1:
            stats.single_candidate += 1
            continue

        candidate_keys = [candidate["action_key"] for candidate in candidates]
        try:
            target_key = agent.action_key(action, raw_state)
        except (KeyError, ValueError):
            stats.no_match += 1
            if len(stats.no_match_samples) < DIAG_SAMPLE_CAP:
                stats.no_match_samples.append(
                    {
                        "state_type": raw_state.get("state_type"),
                        "action": action.get("type"),
                        "recorded_key": None,  # action could not be keyed
                        "candidate_keys": candidate_keys,
                    }
                )
            continue

        target_index = next(
            (index for index, key in enumerate(candidate_keys) if key == target_key),
            None,
        )
        if target_index is None:
            stats.no_match += 1
            if len(stats.no_match_samples) < DIAG_SAMPLE_CAP:
                stats.no_match_samples.append(
                    {
                        "state_type": raw_state.get("state_type"),
                        "action": action.get("type"),
                        "recorded_key": target_key,
                        "candidate_keys": candidate_keys,
                    }
                )
            continue

        state_vector = agent.encode_state(raw_state)
        rows = [
            state_vector + agent.encode_action(raw_state, candidate["action"])
            for candidate in candidates
        ]
        examples.append(
            Example(
                state_action=torch.tensor(rows, dtype=torch.float32),
                target_index=target_index,
            )
        )
        stats.matched += 1

    return examples, stats


def _run_epoch(
    agent, examples: list[Example], *, train: bool, batch_size: int
) -> tuple[float, float]:
    """Run one pass over examples, returning (mean_loss, top1_accuracy)."""
    device = agent.device
    order = list(range(len(examples)))
    if train:
        random.shuffle(order)

    total_loss = 0.0
    correct = 0
    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            losses = []
            for index in batch:
                example = examples[index]
                logits = agent.bc_score(example.state_action.to(device))
                target = torch.tensor(example.target_index, device=device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
                if int(torch.argmax(logits).item()) == example.target_index:
                    correct += 1

            batch_loss = torch.stack(losses).mean()
            if train:
                agent.optimizer.zero_grad()
                batch_loss.backward()
                agent.optimizer.step()
            total_loss += float(batch_loss.item()) * len(batch)

    mean_loss = total_loss / len(examples) if examples else 0.0
    accuracy = correct / len(examples) if examples else 0.0
    return mean_loss, accuracy


def _current_lr(agent) -> float | None:
    """Best-effort read of the agent optimizer's current learning rate."""
    try:
        return float(agent.optimizer.param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def pretrain(
    agent,
    examples: list[Example],
    *,
    epochs: int,
    batch_size: int,
    val_split: float,
) -> None:
    """Train ``agent`` in place with behavioral cloning over ``examples``."""
    split = int(len(examples) * (1.0 - val_split))
    train_examples = examples[:split]
    val_examples = examples[split:]
    steps_per_epoch = math.ceil(len(train_examples) / batch_size) if train_examples else 0
    logger.info(
        "Behavioral cloning on %d train / %d val examples for %d epochs",
        len(train_examples),
        len(val_examples),
        epochs,
    )
    logger.debug(
        "BC hyperparams: lr=%s batch_size=%d steps/epoch=%d total_steps=%d device=%s",
        _current_lr(agent),
        batch_size,
        steps_per_epoch,
        steps_per_epoch * epochs,
        agent.device,
    )

    best_val_acc: float | None = None
    for epoch in range(1, epochs + 1):
        agent.model.train()
        train_loss, train_acc = _run_epoch(agent, train_examples, train=True, batch_size=batch_size)
        if val_examples:
            agent.model.eval()
            val_loss, val_acc = _run_epoch(agent, val_examples, train=False, batch_size=batch_size)
            best_val_acc = val_acc if best_val_acc is None else max(best_val_acc, val_acc)
            logger.info(
                "epoch %d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
        else:
            logger.info(
                "epoch %d/%d  train_loss=%.4f train_acc=%.3f",
                epoch,
                epochs,
                train_loss,
                train_acc,
            )

    if best_val_acc is not None:
        logger.info("Best val_acc over %d epochs: %.3f", epochs, best_val_acc)


def screen_samples(paths, screen: str):
    """Yield ``(state, action)`` recordings whose state the screen controls.

    Filtering by ``screen_name_for_state`` (not just action type) is required: a
    screen encoder fed a foreign state can emit a spurious ``proceed`` candidate
    that would otherwise match unrelated ``proceed`` actions from other screens.
    """
    for state, action in iter_recordings(paths):
        if screen_name_for_state(state) == screen:
            yield state, action


@dataclass
class PretrainTarget:
    """One agent to pretrain plus how to route recordings and where to save it."""

    name: str
    agent: Any
    default_out: Path
    controls: Callable[[dict], bool]
    samples: list = field(default_factory=list)


def build_target(name: str, args: argparse.Namespace) -> PretrainTarget:
    """Create a pretrain target (agent + routing predicate + output path) by name."""
    if name == "battle":
        battle_type = normalize_battle_agent_type(args.battle_agent)
        agent = create_battle_agent(battle_type)
        return PretrainTarget("battle", agent, battle_latest_path(battle_type), is_battle_policy_state)

    screen_type = normalize_screen_agent_type(args.screen_agent)
    agent = create_screen_agent(name, screen_type)
    return PretrainTarget(
        name,
        agent,
        agent_latest_path(name, screen_type),
        lambda state, screen=name: screen_name_for_state(state) == screen,
    )


def resolve_target_names(args: argparse.Namespace) -> list[str]:
    """Resolve the CLI selection into an ordered, de-duplicated list of target names.

    No ``--screen`` means battle only (backward compatible). ``all`` expands to
    battle plus every screen; ``--screens-only`` drops battle from the result.
    """
    if not args.screen:
        names = ["battle"]
    elif "all" in args.screen:
        names = ["battle", *SCREEN_NAMES]
    else:
        names = list(dict.fromkeys(args.screen))  # de-dupe, preserve order

    if args.screens_only:
        names = [name for name in names if name != "battle"]
    if not names:
        raise SystemExit("No agents selected to pretrain.")
    return names


def bucket_samples(paths, targets: list[PretrainTarget], limit: int | None) -> None:
    """Read recordings once and route each sample into its controlling target.

    Battle and screen predicates are mutually exclusive (a state is battle, a
    screen, or neither), so the first matching target claims the sample.
    """
    total = 0
    dropped: Counter = Counter()  # state_type -> count for samples no target claimed
    for index, (state, action) in enumerate(iter_recordings(paths)):
        if limit is not None and index >= limit:
            break
        total += 1
        for target in targets:
            if target.controls(state):
                target.samples.append((state, action))
                break
        else:
            dropped[state.get("state_type")] += 1

    routed = ", ".join(f"{target.name}={len(target.samples)}" for target in targets)
    logger.info(
        "Read %d recordings; routed: %s; dropped=%d (no selected agent controls them)",
        total,
        routed,
        sum(dropped.values()),
    )
    if dropped:
        logger.debug("Dropped recordings by state_type: %s", dict(dropped))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse behavioral-cloning CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain an STS2RL battle or screen agent on human-play recordings.",
    )
    parser.add_argument(
        "--recordings",
        action="append",
        required=True,
        help="JSONL recording file, directory, or glob. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--screen",
        action="append",
        choices=[*sorted(SCREEN_NAMES), "all"],
        default=None,
        help="Pretrain non-battle screen agent(s) instead of the battle agent. Repeatable; "
        "'all' trains the battle agent plus every screen. No --screen trains the battle agent.",
    )
    parser.add_argument(
        "--screens-only",
        action="store_true",
        help="With --screen all, exclude the battle agent and train the screen agents only.",
    )
    parser.add_argument("--battle-agent", default="DQN", help="Battle agent type (DQN or PPO).")
    parser.add_argument(
        "--screen-agent",
        default="PPO",
        help="Screen agent type (DQN or PPO) applied to every selected screen.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr", type=float, default=None, help="Override the agent's learning rate."
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Exploration rate written into the checkpoint so RL exploits the "
        "pretrained policy on resume. Defaults to the agent's epsilon_min.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Checkpoint output path for a single target. Each agent otherwise saves to "
        "its own standard latest path so `sts2rl-train` resumes from it. Cannot be used "
        "when training multiple agents.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap on samples (for quick runs).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (per-target dims, routing breakdown, action-key mismatches).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``sts2rl-pretrain``."""
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("sts2rl").setLevel(level)  # take effect even if logging was pre-configured

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.debug("Seeded random and torch with %d", args.seed)

    target_names = resolve_target_names(args)
    if args.out is not None and len(target_names) > 1:
        raise SystemExit("--out cannot be used when training multiple agents; each saves to its own path.")

    logger.info(
        "Pretrain config: targets=%s recordings=%s battle_agent=%s screen_agent=%s "
        "epochs=%d batch_size=%d lr=%s val_split=%.2f device=%s seed=%s limit=%s",
        target_names,
        args.recordings,
        args.battle_agent,
        args.screen_agent,
        args.epochs,
        args.batch_size,
        args.lr,
        args.val_split,
        args.device or "auto",
        args.seed,
        args.limit,
    )

    targets = [build_target(name, args) for name in target_names]
    for target in targets:
        if args.device is not None:
            target.agent.device = torch.device(args.device)
            target.agent.model.to(target.agent.device)
        if args.lr is not None:
            target.agent.optimizer = torch.optim.Adam(target.agent.model.parameters(), lr=args.lr)

    bucket_samples(args.recordings, targets, args.limit)

    trained_names: list[str] = []
    skipped_names: list[str] = []
    for target in targets:
        out_path = Path(args.out) if args.out else target.default_out
        if pretrain_target(target, args, out_path):
            trained_names.append(target.name)
        else:
            skipped_names.append(target.name)

    logger.info(
        "Pretraining complete: %d/%d trained=%s skipped=%s",
        len(trained_names),
        len(targets),
        trained_names,
        skipped_names or "[]",
    )
    if not trained_names:
        raise SystemExit("No usable training examples for any target; check recordings and schema.")


def pretrain_target(target: PretrainTarget, args: argparse.Namespace, out_path: Path) -> bool:
    """Pretrain one target on its routed samples and save it. Returns whether it trained."""
    agent = target.agent
    logger.debug(
        "[%s] agent=%s schema=%s state_size=%d action_feature_size=%d model_input_size=%d "
        "device=%s routed_samples=%d",
        target.name,
        type(agent).__name__,
        getattr(agent, "ACTION_SCHEMA", "?"),
        agent.state_size,
        agent.action_feature_size,
        agent.model_input_size,
        agent.device,
        len(target.samples),
    )

    examples, stats = build_examples(agent, target.samples)
    logger.info(
        "[%s] Recordings: %d total, %d matched (%.1f%%), %d single_candidate, "
        "%d no_candidates, %d no_match",
        target.name,
        stats.total,
        stats.matched,
        stats.match_rate * 100,
        stats.single_candidate,
        stats.no_candidates,
        stats.no_match,
    )
    if stats.no_candidates_samples:
        logger.debug(
            "[%s] no_candidates examples (up to %d): %s",
            target.name,
            DIAG_SAMPLE_CAP,
            stats.no_candidates_samples,
        )
    for sample in stats.no_match_samples:
        logger.debug(
            "[%s] no_match: state_type=%s action=%s recorded_key=%s candidates=%s",
            target.name,
            sample["state_type"],
            sample["action"],
            sample["recorded_key"],
            sample["candidate_keys"],
        )
    if not examples:
        logger.warning("[%s] No usable training examples; skipping.", target.name)
        return False

    random.shuffle(examples)
    pretrain(
        agent,
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    # Set a low exploration rate (if the agent uses one) so RL fine-tuning
    # exploits the cloned policy instead of overwriting it with random play.
    if hasattr(agent, "epsilon"):
        agent.epsilon = (
            args.epsilon
            if args.epsilon is not None
            else getattr(agent, "epsilon_min", agent.epsilon)
        )
    agent.learn_steps = args.epochs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_path))
    logger.info(
        "[%s] Saved pretrained checkpoint to %s (epsilon=%s learn_steps=%d)",
        target.name,
        out_path,
        getattr(agent, "epsilon", None),
        agent.learn_steps,
    )
    return True


if __name__ == "__main__":
    main()
