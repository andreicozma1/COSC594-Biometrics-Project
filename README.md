# COSC594-Biometrics-Project

This project builds a baseline for person re-identification across camera
views. It compares pretrained models, measures retrieval accuracy, and tests
how well learned visual features transfer between datasets and recording
conditions.

## Project overview

### Retrieval task

Given an image of a person, the system ranks images from other cameras so that
images of the same person appear first.

### Data and modality

The baseline uses full-body appearance in visible-light RGB images as its
biometric modality. [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html)
is the first evaluation dataset. A checkpoint trained on Market-1501 provides a
same-dataset reference, while checkpoints trained on other datasets measure
cross-dataset transfer.

Additional datasets would test whether the results extend beyond Market-1501
and expand the evaluation to video, person search, aerial views, infrared
imagery, gait, clothing changes, and multiple modalities. Candidates include
MSMT17 and CUHK03 for image retrieval; MARS, LS-VID, and MEVID for video
retrieval; and CUHK-SYSU and PRW for person search. The
[contender dataset reference](docs/contender-datasets.md) records the complete
set considered for training and evaluation.

### Pipeline scope

The baseline starts with pre-cropped person images and follows four stages:

1. Preprocess the RGB person crops.
2. Extract feature embeddings with pretrained **OSNet-x1.0** models from
   [Torchreid](https://github.com/KaiyangZhou/deep-person-reid).
3. Compare each query embedding with the gallery and rank the matches.
4. Calculate retrieval metrics and inspect the ranked images.

### Motivation

Person re-identification connects observations across cameras that do not share
a continuous field of view. Changes in viewpoint, pose, lighting, occlusion,
image quality, and clothing can make the same person look different. Comparing
checkpoints and datasets shows how well learned identity representations carry
over to new recording conditions.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The
project uses Python 3.12 and locked dependencies. From the project directory,
create the environment with:

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

Use either workflow:

1. Run `extract`, then `evaluate`, to work with one checkpoint and reuse its
   saved embeddings.
2. Run `benchmark` to extract and evaluate multiple checkpoints together.

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

The available distance measures are `squared_euclidean` and `cosine`. Cosine
distance can be selected with `--distance cosine`. It normalizes each embedding
while calculating scores; the saved vectors remain unchanged. If two gallery
images receive exactly the same distance score, they are ordered alphabetically
by filename so repeated evaluations produce the same ranking.

The evaluation directory contains:

- `metrics.json`: the distance measure, mAP, Rank-1/5/10, and evaluated and
  skipped query counts.
- `per-query.json`: AP and the one-based first correct rank for every query, in
  query order.

Metrics are saved as fractions and printed as percentages. Queries with no
same-identity gallery image after filtering have `null` AP and first-rank values
and are excluded from averages.

### 3. Benchmark checkpoints

Pass `--checkpoint` once per model. The command evaluates them in the supplied
order and gives each checkpoint its own result directory:

```bash
uv run reid-baseline benchmark \
  --dataset-root /path/to/Market-1501-v15.09.15 \
  --checkpoint /path/to/market-checkpoint.pth \
  --checkpoint /path/to/msmt-ain-checkpoint.pth osnet_ain_x1_0 cosine \
  --output results/first-benchmark
```

Checkpoint options:

- Repeat `--checkpoint` for each model.
- `--checkpoint PATH` uses `osnet_x1_0` and `squared_euclidean`.
- `--checkpoint PATH ARCHITECTURE DISTANCE` sets both values explicitly.
  Supported architectures are `osnet_x1_0`, `osnet_ibn_x1_0`, and
  `osnet_ain_x1_0`. Supported distances are `squared_euclidean` and `cosine`.
- Checkpoint filenames must have unique stems because each stem names its result
  directory.

Outputs:

- `<checkpoint>/`: the extraction and evaluation files described above.
- `comparison.json`: metrics for all checkpoints in the benchmark.

## Evaluation protocol

The Market-1501 protocol removes junk identity `-1`, retains distractor identity
`0`, and excludes gallery images with both the query's identity and camera.
Queries with no remaining same-ID images are reported as skipped.

- **Rank-1/5/10:** fraction of evaluated queries with a correct match in the
  first 1, 5, or 10 positions.
- **AP:** precision averaged at each correct match in a query's full ranking.
- **mAP:** mean AP across evaluated queries.

## Benchmark results

### Official reference

These Market-1501 results and checkpoint downloads come from the
[Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO).

| Checkpoint | Training data | mAP | Rank-1 |
| --- | --- | ---: | ---: |
| [OSNet-x1.0](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing) | Market-1501 | 82.60% | 94.20% |
| [OSNet-x1.0](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing) | MSMT17 | 37.50% | 66.60% |
| [OSNet-AIN-x1.0](https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing) | MSMT17 | 43.30% | 70.10% |
| [OSNet-x1.0](https://drive.google.com/file/d/1QeQ4WC3i8YGb7Pzd5EX6kHLUUIoGIU_Z/view?usp=sharing) | MS+D+C | 44.20% | 72.50% |
| [OSNet-AIN-x1.0](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view?usp=sharing) | MS+D+C | 45.80% | 73.30% |

`MS+D+C` combines MSMT17, DukeMTMC-reID, and CUHK03 for training, with
Market-1501 held out for evaluation.

### Our results

| Checkpoint | Training data | Distance | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| OSNet-x1.0 | Market-1501 | Squared Euclidean | 82.55% | 94.36% | 97.83% | 98.60% |
| OSNet-x1.0 | MSMT17 | Squared Euclidean | 37.37% | 66.33% | 80.29% | 85.75% |
| OSNet-AIN-x1.0 | MSMT17 | Cosine | 43.28% | 69.83% | 83.97% | 88.45% |
| OSNet-x1.0 | MS+D+C | Cosine | 44.30% | 72.62% | 85.93% | 90.38% |
| OSNet-AIN-x1.0 | MS+D+C | Cosine | 45.80% | 73.01% | 86.55% | 90.50% |

All five runs evaluated 3,368 queries against 15,913 gallery images after junk
removal, with no skipped queries. Compared with the official rounded values,
mAP differs by at most 0.13 percentage points and Rank-1 by at most 0.29 points.

## TODOs

- Compare re-ranking algorithms.
- Evaluate test-time augmentation strategies during extraction.
- Evaluate PCA whitening fitted on training embeddings for cross-domain
  retrieval.
- Define and test a stable policy for resolving equal distance scores.
