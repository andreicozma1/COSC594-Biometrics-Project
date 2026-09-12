# COSC594-Biometrics-Project

A person retrieval baseline using pretrained **OSNet-x1.0** models from
[Torchreid](https://github.com/KaiyangZhou/deep-person-reid). Extract embeddings,
then evaluate identity retrieval from the saved vectors.

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

Download [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html)
and extract it. The dataset root must contain `query/` and `bounding_box_test/`.
Use the standard gallery, without the optional 500,000-image extension.

Download either or both plain OSNet-x1.0 checkpoints from the
[official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO):

- [Market-trained](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing): trained on Market-1501.
- [MSMT17-trained, combineall](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing): trained on MSMT17 with combineall.

## Commands

The saved-embedding workflow runs in two steps:

1. Extract reusable embeddings and image metadata from Market-1501.
2. Evaluate the saved embeddings and write retrieval metrics.

The commands process one checkpoint at a time. Repeat both steps with separate
output directories for each checkpoint you want to compare.

### 1. Extract embeddings

Process the standard query and gallery splits with one local checkpoint. The
output directory must be new:

```bash
uv run reid-baseline extract \
  --dataset-root /path/to/Market-1501-v15.09.15 \
  --checkpoint /path/to/checkpoint.pth \
  --architecture osnet_x1_0 \
  --output results/first-extraction
```

Choose `osnet_x1_0`, `osnet_ibn_x1_0`, or `osnet_ain_x1_0` to match the
checkpoint architecture.

Images are read in filename order, converted to RGB, resized to **256 high × 128
wide**, and normalized with ImageNet statistics. OSNet produces one
512-dimensional float32 embedding per image in the same order as the saved
image records.

Extraction automatically uses an available accelerator and otherwise runs on
CPU.

The extraction directory contains:

- `embeddings.npz`: query and gallery float32 arrays with 512 values per row.
- `images.json`: the dataset location and ordered image paths, identities, and
  cameras for both splits.
- `settings.json`: the model, checkpoint path and hash, preprocessing, device,
  batch size, and extraction time.

### 2. Evaluate saved embeddings

Score the extraction above and save results to a new directory:

```bash
uv run reid-baseline evaluate \
  --extraction results/first-extraction \
  --output results/first-evaluation
```

Evaluation uses squared Euclidean distance by default. Select cosine distance
with `--distance cosine`; normalization is applied while scoring and does not
change the saved embeddings. Distance ties follow gallery filename order.

The evaluation directory contains:

- `metrics.json`: the distance measure, mAP, Rank-1/5/10, and evaluated and
  skipped query counts.
- `per-query.json`: AP and the one-based first correct rank for every query, in
  query order.

Metrics are saved as fractions and printed as percentages. Queries with no
same-identity gallery image after filtering have `null` AP and first-rank values
and are excluded from averages.

## Evaluation protocol

The Market-1501 protocol removes junk identity `-1`, retains distractor identity
`0`, and excludes gallery images with both the query's identity and camera.
Queries with no remaining same-ID images are reported as skipped.

- **Rank-1/5/10:** fraction of evaluated queries with a correct match in the
  first 1, 5, or 10 positions.
- **AP:** precision averaged at each correct match in a query's full ranking.
- **mAP:** mean AP across evaluated queries.

## Planned work

- Add optional feature normalization.
- Add re-ranking.
- Add test-time augmentation during extraction.
- Consider using a stable distance sort so equal distances preserve gallery
  filename order.

## Baseline result

The [OSNet-x1.0 checkpoint trained on MSMT17 with `combineall`](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing)
was evaluated on the standard query and gallery splits from
[Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html). The
gallery contained 15,913 images after excluding identity `-1`. All 3,368 queries
had at least one remaining same-identity image from another camera.

| Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| MSMT17-trained OSNet-x1.0 | 37.37% | 66.33% | 80.29% | 85.75% |
