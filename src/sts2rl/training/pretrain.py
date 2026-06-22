"""Behavioral-cloning pretraining for candidate-scoring battle agents.

Records of human play (state, action) pairs are turned into a supervised signal:
for each decision state, the agent enumerates the legal action candidates, scores
them, and is trained with softmax cross-entropy to give the candidate the human
actually chose the highest probability.

This trainer is intentionally agent-agnostic. It relies only on the shared
candidate interface (``encode_state``, ``encode_action``, ``valid_action_candidates``,
``action_key``, ``bc_score``, ``save``). Adding a new algorithm = registering it in
``orchestrator.BATTLE_AGENT_TYPES``; nothing here changes.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from sts2rl.agents.orchestrator import create_battle_agent, normalize_battle_agent_type
from sts2rl.checkpoints.manager import battle_latest_path
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


@dataclass
class BuildStats:
    """Counters describing how recordings mapped onto candidate sets."""

    total: int = 0
    matched: int = 0
    no_candidates: int = 0
    no_match: int = 0

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
            continue

        try:
            target_key = agent.action_key(action, raw_state)
        except (KeyError, ValueError):
            stats.no_match += 1
            continue

        target_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate["action_key"] == target_key
            ),
            None,
        )
        if target_index is None:
            stats.no_match += 1
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
    logger.info(
        "Behavioral cloning on %d train / %d val examples for %d epochs",
        len(train_examples),
        len(val_examples),
        epochs,
    )

    for epoch in range(1, epochs + 1):
        agent.model.train()
        train_loss, train_acc = _run_epoch(agent, train_examples, train=True, batch_size=batch_size)
        if val_examples:
            agent.model.eval()
            val_loss, val_acc = _run_epoch(agent, val_examples, train=False, batch_size=batch_size)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse behavioral-cloning CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain an STS2RL battle agent on human-play recordings.",
    )
    parser.add_argument(
        "--recordings",
        action="append",
        required=True,
        help="JSONL recording file, directory, or glob. Repeat for multiple sources.",
    )
    parser.add_argument("--battle-agent", default="DQN", help="Battle agent type (DQN or PPO).")
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
        help="Checkpoint output path. Defaults to the standard battle latest path "
        "for the agent type so `sts2rl-train` resumes from it.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap on samples (for quick runs).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``sts2rl-pretrain``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    agent_type = normalize_battle_agent_type(args.battle_agent)
    agent = create_battle_agent(agent_type)
    if args.device is not None:
        agent.device = torch.device(args.device)
        agent.model.to(agent.device)
    if args.lr is not None:
        agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    samples = iter_recordings(args.recordings, action_types=agent.ACTION_TYPES)
    if args.limit is not None:
        samples = (sample for index, sample in enumerate(samples) if index < args.limit)

    examples, stats = build_examples(agent, samples)
    logger.info(
        "Recordings: %d total, %d matched (%.1f%%), %d no_candidates, %d no_match",
        stats.total,
        stats.matched,
        stats.match_rate * 100,
        stats.no_candidates,
        stats.no_match,
    )
    if not examples:
        raise SystemExit("No usable training examples; check recordings and action schema.")

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

    out_path = Path(args.out) if args.out else battle_latest_path(agent_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_path))
    logger.info("Saved pretrained %s checkpoint to %s", agent_type, out_path)


if __name__ == "__main__":
    main()
