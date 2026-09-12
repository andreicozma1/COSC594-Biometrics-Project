"""Load local OSNet weights and turn person crops into embeddings.

Model loading:
    Read checkpoints on CPU, discard the training classifier, and require the
    full embedding network before moving the model to the selected device.

Extraction:
    Decode images as RGB, apply the reference preprocessing, and process them
    in batches. Return float32 vectors in the supplied record order.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torchreid.models import build_model
from torchvision import transforms

from .configuration import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    PREPROCESSING,
    DeviceName,
    ModelArchitecture,
)
from .data import ImageRecord


def select_device(requested: str) -> torch.device:
    """Select an available accelerator or use the requested backend.

    Args:
        requested: auto, cuda, mps, or cpu. Auto uses PyTorch's accelerator
            detection and falls back to CPU when no accelerator is available.

    Returns:
        The device used for the model and input batches.

    Raises:
        ValueError: The name is unsupported or the requested backend is unavailable.
    """
    if requested not in set(DeviceName):
        raise ValueError(f"Unsupported device: {requested}. Choose auto, cuda, mps, or cpu.")

    if requested == DeviceName.AUTO:
        return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    # An explicit choice must not silently switch to another backend.
    device = torch.device(requested)
    if device.type != "cpu" and not torch.get_device_module(device).is_available():
        raise ValueError(f"{requested.upper()} is unavailable; use --device auto or --device cpu.")
    return device


def build_transform() -> transforms.Compose:
    """Build the resize and normalization pipeline for RGB crops.

    Returns:
        A transform that resizes PIL RGB images to 256 high by 128 wide, scales
        pixels to [0, 1], then applies ImageNet channel normalization. Output
        tensors have channel-first layout: [3, 256, 128].
    """
    return transforms.Compose(
        [
            transforms.Resize((PREPROCESSING.height, PREPROCESSING.width)),
            transforms.ToTensor(),
            transforms.Normalize(PREPROCESSING.mean, PREPROCESSING.std),
        ]
    )


def _read_checkpoint(checkpoint: Path) -> Mapping[object, object]:
    """Read weight tensors on CPU before allocating accelerator memory.

    Args:
        checkpoint: Local PyTorch checkpoint file.

    Returns:
        A state dictionary, either stored directly or under the state_dict key.

    Raises:
        FileNotFoundError: The checkpoint file is missing.
        ValueError: PyTorch's weights-only loader cannot read the file, or the
            payload is not a state dictionary.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint {checkpoint}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a state dictionary: {checkpoint}")

    # Upstream files may wrap the tensors together with training metadata.
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"Invalid state_dict in {checkpoint}")

    return state


def _embedding_weights(
    state: Mapping[object, object], model: nn.Module, checkpoint: Path
) -> dict[str, object]:
    """Adapt training weights for strict loading into the embedding model.

    Args:
        state: Raw checkpoint entries, possibly prefixed with ``module.``.
        model: OSNet with its unused classifier replaced by nn.Identity.
        checkpoint: File path included in loading errors.

    Returns:
        A new dictionary with the prefix and classifier entries removed.
        PyTorch checks keys, tensor values, and shapes when this is loaded.

    Raises:
        ValueError: Prefix removal creates a duplicate key, a key is not a
            string, or a tensor contains nonfinite values or an incompatible
            numeric type.
    """
    weights: dict[str, object] = {}
    expected = model.state_dict()

    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f"Checkpoint keys must be strings: {checkpoint}")

        # DataParallel adds this prefix; the training classifier is unused here.
        key = key.removeprefix("module.")
        if key.startswith("classifier."):
            continue
        if key in weights:
            raise ValueError(f"Duplicate checkpoint key after removing module prefix: {key}")

        # Strict loading checks compatibility, but permits casts and NaN/Inf values.
        if isinstance(value, torch.Tensor):
            target = expected.get(key)
            if value.is_complex() or (
                target is not None and value.is_floating_point() != target.is_floating_point()
            ):
                raise ValueError(f"Incompatible checkpoint dtype for {key}: {value.dtype}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {key}")

        weights[key] = value

    return weights


def load_model(
    checkpoint: Path,
    device: torch.device,
    architecture: ModelArchitecture = DEFAULT_ARCHITECTURE,
) -> nn.Module:
    """Load a local OSNet checkpoint for embedding extraction.

    Args:
        checkpoint: Local file containing a raw state dictionary or a checkpoint
            with a state_dict entry. The unused classifier may be omitted.
        device: Backend on which the returned model will run.
        architecture: Torchreid model definition matching the saved parameters.

    Returns:
        A float32 model in evaluation mode. Automatic weight downloads are disabled.

    Raises:
        FileNotFoundError: The checkpoint file is missing.
        ValueError: The checkpoint cannot be read or its embedding parameters do
            not match the selected OSNet architecture.
        RuntimeError: PyTorch cannot move the model to the requested device.
    """
    checkpoint = checkpoint.expanduser().resolve()

    # Torchreid requires a class count at construction; retrieval removes the head.
    model = build_model(architecture, num_classes=1, pretrained=False)
    model.set_submodule("classifier", nn.Identity(), strict=True)
    weights = _embedding_weights(_read_checkpoint(checkpoint), model, checkpoint)

    # With the classifier removed, strict loading checks the embedding network directly.
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Incompatible checkpoint {checkpoint}: {exc}") from exc

    return model.float().eval().to(device)


def _image_tensor(path: Path, transform: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    """Decode a crop and prepare it for inference.

    Args:
        path: Full image path; it is included in any image-loading error.
        transform: Preprocessing applied after converting the image to RGB.

    Returns:
        A channel-first tensor ready to stack with the rest of the batch.

    Raises:
        ValueError: The image cannot be opened or processed.
    """
    try:
        with Image.open(path) as image:
            return transform(image.convert("RGB"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc


def extract_embeddings(
    model: nn.Module,
    root: Path,
    records: Sequence[ImageRecord],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NDArray[np.float32]:
    """Extract one embedding per image, preserving record order.

    Args:
        model: Model returned by load_model, already on device and in eval mode.
        root: Dataset directory used to resolve each record's relative path.
        records: Nonempty sequence of images to process.
        device: Backend used for the model and each input batch.
        batch_size: Maximum number of images per batch; must be at least 1.

    Returns:
        A finite float32 array of shape [len(records), 512]. The vectors are not
        normalized after extraction.

    Raises:
        ValueError: Inputs are empty or invalid, an image cannot be read, or the
            model returns nonfinite values or an unexpected embedding shape.
        RuntimeError: PyTorch cannot complete inference.

    Images are decoded in this process. An unreadable image stops extraction;
    records are never silently skipped.
    """
    if batch_size < 1 or not records:
        raise ValueError("Embedding extraction requires records and a positive batch size.")

    transform = build_transform()
    embeddings = np.empty((len(records), EMBEDDING_DIM), dtype=np.float32)

    # No gradients are needed, and only one image batch occupies the device at a time.
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = [_image_tensor(root / record.path, transform) for record in batch_records]

            features = model(torch.stack(batch).to(device)).cpu().numpy()
            if features.shape != (len(batch), EMBEDDING_DIM):
                raise ValueError(
                    f"Invalid embeddings in the batch starting at {records[start].path}: "
                    f"expected shape {(len(batch), EMBEDDING_DIM)}, got {features.shape}."
                )
            if not np.isfinite(features).all():
                raise ValueError(
                    f"Invalid embeddings in the batch starting at {records[start].path}: "
                    "found NaN or infinity."
                )

            # Keep each output row aligned with its original image record.
            embeddings[start : start + len(batch)] = features

    return embeddings
