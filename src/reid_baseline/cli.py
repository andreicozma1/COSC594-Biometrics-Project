"""Run embedding extraction from the command line.

Workflow:
    - Read the local dataset and checkpoint paths.
    - Extract query and gallery embeddings with the same model.
    - Save the arrays, image labels, and extraction settings together.

The command reports progress to stdout and errors to stderr.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

from .configuration import (
    DEFAULT_BATCH_SIZE,
    DeviceName,
    ExtractionOptions,
    ExtractionSettings,
)
from .data import load_market1501
from .model import extract_embeddings, load_model, select_device
from .results import save_embeddings


def _parser() -> argparse.ArgumentParser:
    """Build the parser for the extract command.

    Returns:
        A parser with required input/output paths and optional device and batch
        settings. The positive batch-size check happens in main after parsing.
    """
    parser = argparse.ArgumentParser(description="Extract OSNet-x1.0 embeddings from Market-1501.")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract", help="Save query and gallery embeddings for one checkpoint."
    )

    # Input paths and output location.
    extract.add_argument("--dataset-root", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True, help="New output directory.")

    # Runtime choices apply to both splits.
    extract.add_argument(
        "--device", type=DeviceName, choices=list(DeviceName), default=DeviceName.AUTO
    )
    extract.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def _extract(options: ExtractionOptions) -> None:
    """Extract both Market-1501 splits with one local checkpoint.

    Args:
        options: Input paths and inference settings. The output directory must
            not already exist; paths may contain a leading tilde.

    Side Effects:
        Creates the output directory and writes embeddings plus JSON metadata.
        Prints the selected device, split sizes, and output location.

    Raises:
        OSError: An input is missing, the output exists, or a file cannot be written.
        ValueError: The dataset, checkpoint, or model output is invalid.
        RuntimeError: PyTorch cannot complete inference on the selected device.

    A failed or interrupted extraction may leave partial output. Retry with a new
    output directory.
    """
    # Load the inputs before creating a directory that could be left incomplete.
    started = perf_counter()
    output = options.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory to keep the existing files."
        )

    dataset = load_market1501(options.dataset_root)
    device = select_device(options.device)
    checkpoint = options.checkpoint.expanduser().resolve()
    model = load_model(checkpoint, device)

    # The filename alone does not identify which weights produced the vectors.
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()

    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}\nDevice: {device}", flush=True)

    # Use one model and the same batching settings for both splits.
    print(f"Extracting {len(dataset.queries)} query images", flush=True)
    query = extract_embeddings(model, dataset.root, dataset.queries, device, options.batch_size)

    print(f"Extracting {len(dataset.gallery)} gallery images", flush=True)
    gallery = extract_embeddings(model, dataset.root, dataset.gallery, device, options.batch_size)

    # Record the actual device and stop the extraction timer before saving.
    settings = ExtractionSettings(
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        device=str(device),
        batch_size=options.batch_size,
        extraction_seconds=perf_counter() - started,
    )
    save_embeddings(output, dataset, query, gallery, settings)
    print(f"Saved embeddings and metadata: {output}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one extraction.

    Args:
        argv: Command arguments without the program name. None reads sys.argv.

    Returns:
        0 on success, 2 for a reported input or runtime error, or 130 when the
        user interrupts extraction.

    Raises:
        SystemExit: argparse handles --help and invalid command arguments before
            extraction starts.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    # Keep argparse-specific objects out of the extraction code.
    try:
        _extract(
            ExtractionOptions(
                dataset_root=args.dataset_root,
                checkpoint=args.checkpoint,
                output=args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Extraction interrupted. Partial output may remain; retry with a new output directory.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
