"""Run extraction, saved-vector evaluation, and checkpoint benchmarks.

Each command requires a new output directory. Benchmarking applies one dataset
and runtime configuration to every supplied checkpoint.
"""

import hashlib
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .configuration import (
    BenchmarkOptions,
    CheckpointSpec,
    EmbeddingEvaluationOptions,
    ExtractionOptions,
    ExtractionSettings,
)
from .data import Market1501, load_market1501
from .evaluation import evaluate
from .results import (
    BenchmarkResult,
    load_extraction,
    save_comparison,
    save_embeddings,
    save_metrics,
)

if TYPE_CHECKING:
    import torch


def _extract_and_save(
    options: ExtractionOptions,
    dataset: Market1501,
    output: Path,
    device: "torch.device",
    started_at: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extract both splits and save files understood by ``evaluate``.

    Args:
        options: Checkpoint, architecture, output, and batching settings.
        dataset: Ordered query and gallery image records.
        output: Existing directory that receives the extraction files.
        device: Resolved backend that runs the model.
        started_at: Timer value used to record extraction duration.

    Returns:
        Query and gallery embeddings in the dataset's saved record order.

    Raises:
        OSError: An input cannot be read or an output cannot be written.
        ValueError: The checkpoint, image, or embedding output is invalid.
        RuntimeError: PyTorch cannot complete inference.

    The caller creates the output directory and resolves the dataset and device.
    Keeping this sequence shared ensures standalone and benchmark extractions
    produce the same files.
    """
    from .model import extract_embeddings, load_model

    checkpoint = options.checkpoint.expanduser().resolve()
    model = load_model(checkpoint, device, options.architecture)

    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()

    print(f"Extracting {len(dataset.queries)} query images", flush=True)
    query = extract_embeddings(
        model, dataset.root, dataset.queries, device, options.batch_size
    )

    print(f"Extracting {len(dataset.gallery)} gallery images", flush=True)
    gallery = extract_embeddings(
        model, dataset.root, dataset.gallery, device, options.batch_size
    )

    settings = ExtractionSettings(
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        device=str(device),
        batch_size=options.batch_size,
        extraction_seconds=perf_counter() - started_at,
        model=options.architecture,
    )
    save_embeddings(output, dataset, query, gallery, settings)
    return query, gallery


def extract(options: ExtractionOptions) -> None:
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
    # Saved-vector evaluation does not need PyTorch, so import device selection
    # only for commands that actually run inference.
    from .model import select_device

    # Load the inputs before creating a directory that could be left incomplete.
    started = perf_counter()
    output = options.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory to keep the existing files."
        )

    dataset = load_market1501(options.dataset_root)
    device = select_device(options.device)

    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}\nDevice: {device}", flush=True)
    _extract_and_save(options, dataset, output, device, started)
    print(f"Saved embeddings and metadata: {output}", flush=True)


def _resolve_checkpoints(
    checkpoints: tuple[CheckpointSpec, ...],
) -> list[CheckpointSpec]:
    """Resolve checkpoint paths and reject missing files or duplicate names.

    Args:
        checkpoints: Models in command-line order.

    Returns:
        Equivalent settings with absolute checkpoint paths.

    Raises:
        FileNotFoundError: A checkpoint does not exist.
        ValueError: No checkpoint was supplied or filename stems collide.

    Result directory names come from filename stems. Validation happens before
    creating the benchmark directory so a naming error leaves no output behind.
    """
    if not checkpoints:
        raise ValueError("Provide at least one --checkpoint.")

    resolved: list[CheckpointSpec] = []
    names: set[str] = set()
    for checkpoint in checkpoints:
        path = checkpoint.path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

        name = path.stem.casefold()
        if name in names:
            raise ValueError(
                f"Checkpoint filenames must have distinct stems: {path.name}"
            )
        names.add(name)
        resolved.append(
            CheckpointSpec(path, checkpoint.architecture, checkpoint.distance)
        )

    return resolved


def _print_comparison(results: list[BenchmarkResult]) -> None:
    """Print the aggregate metrics stored in ``comparison.json``.

    Args:
        results: Completed checkpoints in the order supplied to the command.

    Side Effects:
        Writes one heading and one metric row per checkpoint to standard output.
    """
    print(
        f"{'Checkpoint':<24} {'mAP':>8} {'Rank-1':>8} "
        f"{'Rank-5':>8} {'Rank-10':>8} {'Seconds':>9}"
    )
    for result in results:
        scores = " ".join(
            f"{score:>7.2%}"
            for score in (
                result.mAP,
                result.rank1,
                result.rank5,
                result.rank10,
            )
        )
        print(f"{result.checkpoint:<24} {scores} {result.elapsed_seconds:>9.2f}")


def benchmark(options: BenchmarkOptions) -> None:
    """Extract and evaluate checkpoints against the same Market-1501 split.

    Each checkpoint receives a subdirectory containing the standard extraction
    and evaluation files. ``comparison.json`` preserves command-line order and
    is written after every checkpoint completes.

    Args:
        options: Dataset, checkpoint, output, device, and batching settings.

    Side Effects:
        Creates the benchmark directory, saves each checkpoint's embeddings and
        metrics, writes ``comparison.json``, and prints aggregate results.

    Raises:
        OSError: An input is missing or an output cannot be created.
        ValueError: Checkpoint names collide or model/evaluation data is invalid.
        RuntimeError: Model loading or inference fails.
    """
    from .model import select_device

    checkpoints = _resolve_checkpoints(options.checkpoints)
    output = options.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory."
        )

    dataset = load_market1501(options.dataset_root)
    device = select_device(options.device)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}\nDevice: {device}", flush=True)

    results: list[BenchmarkResult] = []
    for checkpoint in checkpoints:
        started = perf_counter()
        directory = output / checkpoint.path.stem
        directory.mkdir()
        print(f"\n{checkpoint.path.name}", flush=True)

        extraction = ExtractionOptions(
            dataset_root=dataset.root,
            checkpoint=checkpoint.path,
            output=directory,
            architecture=checkpoint.architecture,
            device=options.device,
            batch_size=options.batch_size,
        )
        query, gallery = _extract_and_save(
            extraction,
            dataset,
            directory,
            device,
            started,
        )
        evaluation = evaluate(
            dataset.queries,
            dataset.gallery,
            query,
            gallery,
            options.batch_size,
            checkpoint.distance,
        )
        save_metrics(directory, evaluation)
        results.append(
            BenchmarkResult.from_evaluation(
                checkpoint,
                evaluation,
                perf_counter() - started,
            )
        )

    save_comparison(output, results)
    _print_comparison(results)
    print(f"Saved benchmark: {output}")


def evaluate_extraction(options: EmbeddingEvaluationOptions) -> None:
    """Score a saved extraction and write JSON results to a new directory.

    Args:
        options: Extraction directory, output directory, and query batch size.

    Side Effects:
        Writes metrics.json and per-query.json, then prints the aggregate metrics.
        Neither the extraction files nor original images are modified.

    Raises:
        ValueError: Saved inputs are invalid or no query has a usable positive match.
        OSError: The output already exists or cannot be written.
    """
    output = options.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory."
        )

    extraction = load_extraction(options.extraction)
    evaluation = evaluate(
        extraction.queries,
        extraction.gallery,
        extraction.query_features,
        extraction.gallery_features,
        options.batch_size,
        options.distance,
    )
    # Invalid inputs and unscorable queries should not leave an empty output directory.
    output.mkdir(parents=True, exist_ok=False)
    save_metrics(output, evaluation)

    metrics = evaluation.metrics
    print(
        f"Evaluated {metrics.valid_queries}/{metrics.total_queries} queries; "
        f"skipped {metrics.skipped_queries}."
    )
    print(
        f"Rank-1: {metrics.rank1:.2%} | Rank-5: {metrics.rank5:.2%} | "
        f"Rank-10: {metrics.rank10:.2%} | mAP: {metrics.mAP:.2%}"
    )
    print(f"Saved evaluation: {output}")
