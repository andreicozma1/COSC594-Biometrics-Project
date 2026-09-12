"""Run extraction and evaluate saved embeddings from typed command options.

Extraction loads a local model; evaluation works directly from saved vectors
and image labels. Each command requires a new output directory.
"""

import hashlib
from time import perf_counter

from .configuration import EmbeddingEvaluationOptions, ExtractionOptions, ExtractionSettings
from .data import load_market1501
from .evaluation import evaluate
from .results import load_extraction, save_embeddings, save_metrics


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
    # Saved-vector evaluation does not need the image or model stack, so load it
    # only for commands that actually run inference.
    from .model import extract_embeddings, load_model, select_device

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
    model = load_model(checkpoint, device, options.architecture)

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
        model=options.architecture,
    )
    save_embeddings(output, dataset, query, gallery, settings)
    print(f"Saved embeddings and metadata: {output}", flush=True)


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
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")

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
