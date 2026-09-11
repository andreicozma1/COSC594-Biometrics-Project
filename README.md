# COSC594-Biometrics-Project

A person retrieval baseline using pretrained **OSNet-x1.0** models from
[Torchreid](https://github.com/KaiyangZhou/deep-person-reid). The current command
extracts embeddings; retrieval, evaluation, and visualization will follow.

[Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html) is our
first baseline dataset. More datasets are planned; the
[contender dataset reference](docs/contender-datasets.md) supports their selection.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from this project directory. The project uses Python 3.12 and locked dependencies.
On macOS, installation may require Apple's Command Line Tools
(`xcode-select --install`).

```bash
uv sync --locked
```

Download Market-1501 from its official page above and extract it. The dataset
root must contain `query/` and `bounding_box_test/`. Use the standard gallery,
without the optional 500,000-image extension.

Download either or both plain OSNet-x1.0 checkpoints from the
[official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO):

- [Market-trained](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing): trained on Market-1501.
- [MSMT17-trained, combineall](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing): trained on MSMT17 with combineall.

## Extract embeddings

Extract the query and gallery splits with one local checkpoint. Choose a new
output directory:

```bash
uv run reid-baseline extract \
  --dataset-root /path/to/Market-1501-v15.09.15 \
  --checkpoint /path/to/checkpoint.pth \
  --output results/first-extraction
```

### Options

- `--device auto|cuda|mps|cpu`: defaults to `auto`, which selects an available
  accelerator, falling back to CPU.
- `--batch-size`: defaults to 64 images per batch.

### Processing

1. Read images in filename order, excluding junk identity `-1` and retaining
   gallery distractor identity `0`.
2. Convert images to RGB, resize to **256 high × 128 wide**, and apply ImageNet
   normalization.
3. Extract 512-dimensional float32 embeddings without feature normalization.

### Outputs

| File | Contents |
| --- | --- |
| `embeddings.npz` | Query and gallery arrays, one image per row |
| `images.json` | Dataset location, relative image paths, identities, and cameras in embedding order |
| `settings.json` | Checkpoint path and hash, preprocessing, device, and extraction time |
