"""Command line entry point for ``sts2rl-bc-train``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import torch

from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.offline.dataset import BCDataset, DatasetError
from sts2rl.training.bc import BCConfig, BCTrainer, split_dataset, summarize


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts2rl-bc-train",
        description=(
            "Clone demonstrated choices into the actor head, producing an "
            "encoder-weights artifact for sts2rl-train --init-encoder."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="dataset directory written by sts2rl-bc-clean",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="directory to write bc_best.pt and metrics.jsonl into",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=EncoderConfig().hidden_dim)
    parser.add_argument(
        "--entity-layers", type=int, default=EncoderConfig().entity_layers
    )
    parser.add_argument(
        "--entity-heads", type=int, default=EncoderConfig().entity_heads
    )
    parser.add_argument(
        "--entity-ff-dim", type=int, default=EncoderConfig().entity_ff_dim
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        return run_bc(args)
    except KeyboardInterrupt:
        return 130


def run_bc(args: argparse.Namespace) -> int:
    try:
        config = BCConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            holdout_fraction=args.holdout_fraction,
            patience=args.patience,
            seed=args.seed,
        )
        encoder_config = EncoderConfig(
            hidden_dim=args.hidden_dim,
            entity_layers=args.entity_layers,
            entity_heads=args.entity_heads,
            entity_ff_dim=args.entity_ff_dim,
        )
    except (TypeError, ValueError) as exc:
        print(f"sts2rl-bc-train: {exc}", file=sys.stderr)
        return 2

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    vocabulary = GameVocabulary.from_bundled_data()
    try:
        dataset = BCDataset.load(
            args.dataset, tokenizer=GameTokenizer(vocabulary)
        )
    except DatasetError as exc:
        print(f"sts2rl-bc-train: {exc}", file=sys.stderr)
        return 2

    # The warning says the holdout shares its run with training, which makes
    # every holdout number a sanity check rather than a measure.  It has to
    # reach the operator, not a log nobody reads.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train, holdout = split_dataset(dataset, config)
    for warning in caught:
        print(f"sts2rl-bc-train: WARNING: {warning.message}", file=sys.stderr)

    print(f"decisions: {len(dataset)} ({len(train)} train / {len(holdout)} holdout)")
    print(f"runs:      {len(dataset.run_ids())}")
    print()

    trainer = BCTrainer(
        GameEncoder(vocabulary, encoder_config),
        train,
        holdout,
        config=config,
        device=args.device,
        out_dir=args.out,
    )
    result = trainer.run()

    print(summarize(result))
    if result.checkpoint_path is not None:
        print(f"\nwrote {result.checkpoint_path}")
        print(f"wrote {Path(args.out) / 'metrics.jsonl'}")
        print(
            "\nInitialize a PPO run from it with:\n"
            f"  uv run sts2rl-train --run-dir runs/NAME "
            f"--init-encoder {result.checkpoint_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
