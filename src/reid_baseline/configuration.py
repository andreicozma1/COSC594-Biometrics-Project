"""Define the arguments and settings used by embedding extraction.

Configuration:
    ExtractionOptions holds command inputs. ExtractionSettings records the
    checkpoint and inference settings written to settings.json.

Preprocessing:
    PREPROCESSING supplies the resize and normalization values to both the
    image transform and saved metadata.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Batch size is a runtime default; embedding width is fixed by OSNet-x1.0.
DEFAULT_BATCH_SIZE = 64
EMBEDDING_DIM = 512


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


# Share one definition so recorded settings agree with the image transform.
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
    model: str = "osnet_x1_0"
    dtype: str = "float32"
    embedding_dimensions: int = EMBEDDING_DIM
    feature_normalization: bool = False
