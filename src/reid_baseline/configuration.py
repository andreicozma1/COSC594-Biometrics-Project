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


class ModelArchitecture(StrEnum):
    """Name the OSNet architecture that owns a checkpoint's parameters.

    The value is passed directly to Torchreid's model registry and saved with
    the extracted vectors so the checkpoint can be identified later.
    """

    OSNET_X1_0 = "osnet_x1_0"
    OSNET_IBN_X1_0 = "osnet_ibn_x1_0"
    OSNET_AIN_X1_0 = "osnet_ain_x1_0"


class DistanceMetric(StrEnum):
    """Select how query and gallery embeddings are compared.

    Both measures treat smaller values as better matches. Cosine distance
    normalizes vectors during evaluation; it does not change saved embeddings.
    """

    SQUARED_EUCLIDEAN = "squared_euclidean"
    COSINE = "cosine"


DEFAULT_ARCHITECTURE = ModelArchitecture.OSNET_X1_0
DEFAULT_DISTANCE = DistanceMetric.SQUARED_EUCLIDEAN


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
        checkpoint: Local OSNet weights; no download is performed.
        output: New directory for the arrays and metadata.
        architecture: Model definition used to interpret the checkpoint.
        device: Requested backend, resolved when extraction starts.
        batch_size: Maximum images per inference batch; the CLI requires at least 1.
    """

    dataset_root: Path
    checkpoint: Path
    output: Path
    architecture: ModelArchitecture = DEFAULT_ARCHITECTURE
    device: DeviceName = DeviceName.AUTO
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass(frozen=True)
class EmbeddingEvaluationOptions:
    """Choose an existing extraction to evaluate.

    Attributes:
        extraction: Directory containing embeddings, image records, and settings.
        output: New directory for aggregate and per-query JSON results.
        batch_size: Maximum query rows per distance block.
        distance: Measure used to rank gallery embeddings for each query.
    """

    extraction: Path
    output: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    distance: DistanceMetric = DEFAULT_DISTANCE


@dataclass(frozen=True)
class CheckpointSpec:
    """Describe one checkpoint included in a benchmark.

    Attributes:
        path: Local checkpoint file.
        architecture: OSNet model definition used to load its parameters.
        distance: Measure used to rank its saved embeddings.
    """

    path: Path
    architecture: ModelArchitecture = DEFAULT_ARCHITECTURE
    distance: DistanceMetric = DEFAULT_DISTANCE


@dataclass(frozen=True)
class BenchmarkOptions:
    """Collect inputs shared by a multi-checkpoint benchmark.

    Attributes:
        dataset_root: Market-1501 directory used for every checkpoint.
        checkpoints: Models evaluated in command-line order.
        output: New directory containing one subdirectory per checkpoint.
        device: Inference backend shared by all checkpoints.
        batch_size: Maximum images or queries processed in one batch.
    """

    dataset_root: Path
    checkpoints: tuple[CheckpointSpec, ...]
    output: Path
    device: DeviceName = DeviceName.AUTO
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
    model: ModelArchitecture = DEFAULT_ARCHITECTURE
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
        settings["model"] = ModelArchitecture(settings["model"])
        return cls(**settings)
