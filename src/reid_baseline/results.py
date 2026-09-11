"""Write the files needed to reuse an extraction.

Outputs:
    embeddings.npz: Query and gallery arrays in their original record order.
    images.json: Dataset root, relative image paths, and identity/camera labels.
    settings.json: Checkpoint and inference settings; written after the arrays.

Files are created exclusively. A failed write may leave partial output.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .configuration import EMBEDDING_DIM, ExtractionSettings
from .data import ImageRecord, Market1501


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
    # Validate both splits before writing either one, with the failing split named.
    for split, features, records in (
        ("Query", query, dataset.queries),
        ("Gallery", gallery, dataset.gallery),
    ):
        expected_shape = (len(records), EMBEDDING_DIM)
        if features.shape != expected_shape:
            raise ValueError(
                f"{split} embeddings have shape {features.shape}; expected {expected_shape}."
            )
        if features.dtype != np.float32:
            raise ValueError(f"{split} embeddings have dtype {features.dtype}; expected float32.")
        if not np.isfinite(features).all():
            raise ValueError(f"{split} embeddings contain NaN or infinity.")

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
