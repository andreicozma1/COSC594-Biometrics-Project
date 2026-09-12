"""Define command inputs and the settings saved with extracted embeddings.

Extraction and evaluation share a batch-size default. Preprocessing settings
describe the image transform used before the vectors are saved.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

DEFAULT_BATCH_SIZE = 64
EMBEDDING_DIM = 512
EMBEDDING_DTYPE = "float32"
MODEL_NAME = "osnet_x1_0"


class DeviceName(StrEnum):
    """Select where inference runs.

    Values:
        AUTO: Use an available accelerator, otherwise CPU.
        CUDA: Require CUDA; fail if it is unavailable.
        MPS: Require Apple's Metal backend; fail if it is unavailable.
        CPU: Run on the CPU even when an accelerator is available.
    """

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class ExtractionOptions:
    """Collect the inputs for a single extraction.

    Attributes:
        dataset_root: Folder containing query/ and bounding_box_test/.
        checkpoint: Local OSNet-x1.0 weights; no download is performed.
        output: New directory for the arrays and metadata.
        device: Requested backend, resolved when extraction starts.
        batch_size: Maximum images per inference batch; the CLI requires at least 1.
    """

    dataset_root: Path
    checkpoint: Path
    output: Path
    device: DeviceName = DeviceName.AUTO
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass(frozen=True)
class EmbeddingEvaluationOptions:
    """Choose an existing extraction to evaluate.

    Attributes:
        extraction: Directory containing embeddings, image records, and settings.
        output: New directory for aggregate and per-query JSON results.
        batch_size: Maximum query rows per distance block.
    """

    extraction: Path
    output: Path
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass(frozen=True)
class PreprocessingSettings:
    """Describe the reference resize and channel normalization.

    Attributes:
        height: Image height after resizing, in pixels.
        width: Image width after resizing, in pixels.
        color: Channel order expected by the transform.
        mean: ImageNet RGB means subtracted from pixels scaled to [0, 1].
        std: ImageNet RGB standard deviations used to scale the centered values.
        interpolation: Resize method recorded in the saved settings.
    """

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    color: str = "RGB"
    height: int = 256
    width: int = 128
    interpolation: str = "bilinear"


PREPROCESSING = PreprocessingSettings()


@dataclass(frozen=True)
class ExtractionSettings:
    """Record how a pair of embedding arrays was produced.

    Attributes:
        checkpoint: Resolved path to the weight file.
        checkpoint_sha256: SHA-256 hash of that file.
        device: Backend used for inference.
        batch_size: Maximum images processed in one batch.
        extraction_seconds: Elapsed time through input loading and extraction;
            excludes writing the output files.
        preprocessing: Resize and normalization settings shared with the transform.
        model: Architecture used to load the checkpoint.
        dtype: Numeric type of the saved vectors.
        embedding_dimensions: Number of values in each vector.
        feature_normalization: Whether vectors were normalized after extraction.
    """

    checkpoint: str
    checkpoint_sha256: str
    device: str
    batch_size: int
    extraction_seconds: float
    preprocessing: PreprocessingSettings = PREPROCESSING
    model: str = MODEL_NAME
    dtype: str = EMBEDDING_DTYPE
    embedding_dimensions: int = EMBEDDING_DIM
    feature_normalization: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Self:
        """Convert decoded JSON into typed extraction settings.

        The JSON boundary is handled here so consumers can use named attributes
        instead of repeatedly looking up string keys. Missing, extra, or malformed
        preprocessing fields raise ``TypeError``.
        """
        settings: dict[str, Any] = dict(values)
        preprocessing = settings.pop("preprocessing")
        if not isinstance(preprocessing, Mapping):
            raise TypeError("preprocessing must be a JSON object")

        settings["preprocessing"] = PreprocessingSettings(**dict(preprocessing))
        return cls(**settings)
