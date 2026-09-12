"""Save extracted vectors and evaluate them using aligned image records.

Extraction writes arrays, image metadata, and settings. Evaluation reads those
files and writes aggregate and per-query results as JSON.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .configuration import (
    EMBEDDING_DIM,
    EMBEDDING_DTYPE,
    MODEL_NAME,
    ExtractionSettings,
)
from .data import ImageRecord, Market1501, MarketSplit, parse_filename
from .evaluation import Evaluation


@dataclass(frozen=True)
class ImageMetadata:
    """Pair the saved vectors with their source images.

    Attributes:
        dataset_root: Absolute location of the dataset on the extraction machine.
        query: Query records in embedding order, with paths relative to dataset_root.
        gallery: Gallery records in embedding order, using the same path convention.
    """

    dataset_root: str
    query: list[ImageRecord]
    gallery: list[ImageRecord]


def _serialize_metadata(record: ImageMetadata | ExtractionSettings) -> str:
    """Prepare a metadata record for writing, without touching the filesystem.

    Args:
        record: Image labels or extraction settings to serialize.

    Returns:
        Indented JSON with a trailing newline.

    Raises:
        ValueError: The record contains a nonfinite number.
    """
    return json.dumps(asdict(record), indent=2, allow_nan=False) + "\n"


def save_embeddings(
    directory: Path,
    dataset: Market1501,
    query: NDArray[np.float32],
    gallery: NDArray[np.float32],
    settings: ExtractionSettings,
) -> None:
    """Save both embedding arrays and the metadata that identifies their rows.

    Args:
        directory: Existing output directory. The three output files must be new.
        dataset: Image records in the order used during extraction.
        query: Finite float32 query array, one row per record and 512 columns.
        gallery: Gallery array with the same dtype and width, in gallery record order.
        settings: Checkpoint and inference details to save alongside the arrays.

    Raises:
        ValueError: Array shape, dtype, or values are invalid, or metadata contains
            a nonfinite number.
        OSError: An output file exists or cannot be written.

    Side Effects:
        Writes images.json, embeddings.npz, then settings.json. A failed write
        can leave earlier files in place; retry in a new directory.
    """
    # Validate both splits before writing either one, naming the failing split.
    _validate_saved_embeddings(query, len(dataset.queries), "Query")
    _validate_saved_embeddings(gallery, len(dataset.gallery), "Gallery")

    # Serialize both records first so invalid settings cannot leave earlier files behind.
    metadata = ImageMetadata(str(dataset.root), dataset.queries, dataset.gallery)
    images_json = _serialize_metadata(metadata)
    settings_json = _serialize_metadata(settings)

    # Image records and vectors use the same order; no sorting happens at the saving step.
    with (directory / "images.json").open("x") as handle:
        handle.write(images_json)
    with (directory / "embeddings.npz").open("xb") as handle:
        np.savez(handle, query=query, gallery=gallery)

    # An array-write failure must not leave a settings file suggesting the run finished.
    with (directory / "settings.json").open("x") as handle:
        handle.write(settings_json)


def _read_records(rows: list[dict[str, Any]], split: MarketSplit) -> list[ImageRecord]:
    """Decode image records and check their saved path and label conventions.

    Returns:
        Nonempty, unique records sorted by filename. Paths must name an image
        directly inside the expected split folder.

    Raises:
        ValueError: Paths, filename labels, identities, or ordering disagree.
    """
    records = [ImageRecord(**row) for row in rows]
    for item in records:
        path = Path(item.path)
        if path.parts != (split.value, path.name):
            raise ValueError(f"Invalid saved image path: {item.path}")
        if parse_filename(item.filename) != (item.pid, item.camera):
            raise ValueError(f"Saved image labels disagree with filename: {item.filename}")
        if item.pid < 0 or (split is MarketSplit.QUERY and item.pid == 0):
            raise ValueError(f"Invalid saved identity: {item.filename}")
    # Metadata order is the only link between an image and its saved vector row.
    names = [item.filename for item in records]
    if not records or names != sorted(set(names)):
        raise ValueError(f"Saved {split.value} records must be nonempty, unique, and sorted.")
    return records


def _validate_saved_embeddings(features: np.ndarray, count: int, split: str) -> None:
    """Check the saved OSNet array format before writing or reusing vectors.

    Args:
        features: Array to validate; its contents are never changed.
        count: Number of image records the rows must match.
        split: Query or Gallery, included in errors to identify the affected input.

    Raises:
        ValueError: Shape, dtype, or values violate the finite float32 [N, 512] format.
    """
    expected_shape = (count, EMBEDDING_DIM)
    if features.shape != expected_shape:
        raise ValueError(
            f"{split} embeddings have shape {features.shape}; expected {expected_shape}."
        )
    if features.dtype != np.float32:
        raise ValueError(f"{split} embeddings have dtype {features.dtype}; expected float32.")
    if not np.isfinite(features).all():
        raise ValueError(f"{split} embeddings contain NaN or infinity.")


def _load_embeddings(
    directory: Path,
    queries: list[ImageRecord],
    gallery: list[ImageRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Load saved vectors and check alignment with the image records.

    Returns:
        Query and gallery float32 matrices, in that order. Invalid shape,
        dtype, or values raise ValueError before the arrays are returned.
    """
    with np.load(directory / "embeddings.npz", allow_pickle=False) as data:
        query_features, gallery_features = data["query"], data["gallery"]
    _validate_saved_embeddings(query_features, len(queries), "Query")
    _validate_saved_embeddings(gallery_features, len(gallery), "Gallery")

    return query_features, gallery_features


@dataclass(frozen=True)
class SavedExtraction:
    """Pair saved embeddings with the image labels used during extraction.

    Each array row corresponds to the record at the same position. Reading an
    extraction requires no checkpoint or original image files.
    """

    queries: list[ImageRecord]
    gallery: list[ImageRecord]
    query_features: NDArray[np.float32]
    gallery_features: NDArray[np.float32]


def load_extraction(directory: Path) -> SavedExtraction:
    """Load the three files written by the extract command.

    Args:
        directory: Extraction directory, with a leading tilde expanded.

    Returns:
        Image labels and finite float32 embedding matrices in saved row order.

    Raises:
        ValueError: Files are missing, malformed, or inconsistent with the
            extraction format. Original image paths are not opened.
    """
    directory = directory.expanduser().resolve()
    try:
        # Settings are written last by extract; require them before using the arrays.
        settings_data = json.loads((directory / "settings.json").read_text())
        if not isinstance(settings_data, dict):
            raise ValueError("Extraction settings must be a JSON object.")
        settings = ExtractionSettings.from_mapping(settings_data)
        if (
            settings.model != MODEL_NAME
            or settings.dtype != EMBEDDING_DTYPE
            or settings.embedding_dimensions != EMBEDDING_DIM
            or settings.feature_normalization is not False
        ):
            raise ValueError("Expected unnormalized float32 OSNet-x1.0 extraction settings.")

        metadata = json.loads((directory / "images.json").read_text())
        queries = _read_records(metadata["query"], MarketSplit.QUERY)
        gallery = _read_records(metadata["gallery"], MarketSplit.GALLERY)
        query_features, gallery_features = _load_embeddings(directory, queries, gallery)
        return SavedExtraction(queries, gallery, query_features, gallery_features)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read extraction {directory}: {error}") from error


def save_metrics(directory: Path, evaluation: Evaluation) -> None:
    """Write aggregate and per-query results as JSON.

    Args:
        directory: Existing directory; metrics.json and per-query.json must be new.
        evaluation: Metric fractions and query results in input order.

    Skipped queries have null AP and first_match_rank values. Serialization is
    completed before either file is opened; I/O errors may leave partial output.
    """
    metrics_json = json.dumps(asdict(evaluation.metrics), indent=2, allow_nan=False) + "\n"
    queries_json = json.dumps(
        [asdict(result) for result in evaluation.queries], indent=2, allow_nan=False
    ) + "\n"
    with (directory / "metrics.json").open("x") as handle:
        handle.write(metrics_json)
    with (directory / "per-query.json").open("x") as handle:
        handle.write(queries_json)
