"""Parse commands for embedding extraction and saved-embedding evaluation.

Both commands report errors to stderr and require explicit output paths.
"""

import argparse
import sys
from enum import StrEnum
from pathlib import Path

from .configuration import (
    DEFAULT_BATCH_SIZE,
    DeviceName,
    DistanceMetric,
    EmbeddingEvaluationOptions,
    ExtractionOptions,
    ModelArchitecture,
)
from .pipeline import evaluate_extraction, extract


class Command(StrEnum):
    """Identify a top-level CLI action.

    Each value is the exact command name accepted by argparse.
    """

    EXTRACT = "extract"
    EVALUATE = "evaluate"


def _parser() -> argparse.ArgumentParser:
    """Build the extraction and evaluation parsers.

    Returns:
        A parser requiring a command and that command's paths. Batch-size
        validation happens in main after argparse converts the input.
    """
    parser = argparse.ArgumentParser(
        description="Extract embeddings and evaluate OSNet-x1.0 person retrieval on Market-1501."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        Command.EXTRACT, help="Save query and gallery embeddings for one checkpoint."
    )
    extract.set_defaults(command=Command.EXTRACT)

    # Input paths and output location.
    extract.add_argument("--dataset-root", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument(
        "--architecture",
        type=ModelArchitecture,
        choices=list(ModelArchitecture),
        default=ModelArchitecture.OSNET_X1_0,
    )
    extract.add_argument("--output", type=Path, required=True, help="New output directory.")

    # Runtime choices apply to both splits.
    extract.add_argument(
        "--device", type=DeviceName, choices=list(DeviceName), default=DeviceName.AUTO
    )
    extract.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    evaluation = commands.add_parser(
        Command.EVALUATE, help="Score embeddings from a saved extraction."
    )
    evaluation.set_defaults(command=Command.EVALUATE)
    evaluation.add_argument(
        "--extraction", type=Path, required=True, help="Directory written by extract."
    )
    evaluation.add_argument(
        "--output", type=Path, required=True, help="New result directory; must not already exist."
    )
    evaluation.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluation.add_argument(
        "--distance",
        type=DistanceMetric,
        choices=list(DistanceMetric),
        default=DistanceMetric.SQUARED_EUCLIDEAN,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse a command, run it, and report errors to stderr.

    Args:
        argv: Arguments without the program name; None reads sys.argv.

    Returns:
        Zero on success, 2 for input/runtime errors, and 130 when interrupted.

    Raises:
        SystemExit: argparse handles help and invalid command arguments.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command in {Command.EXTRACT, Command.EVALUATE} and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    try:
        if args.command is Command.EXTRACT:
            extract(
                ExtractionOptions(
                    dataset_root=args.dataset_root,
                    checkpoint=args.checkpoint,
                    output=args.output,
                    architecture=args.architecture,
                    device=args.device,
                    batch_size=args.batch_size,
                )
            )
        elif args.command is Command.EVALUATE:
            evaluate_extraction(
                EmbeddingEvaluationOptions(
                    extraction=args.extraction,
                    output=args.output,
                    batch_size=args.batch_size,
                    distance=args.distance,
                )
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted. Partial output may remain; retry with a new output directory.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
