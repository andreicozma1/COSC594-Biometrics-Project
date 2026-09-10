# COSC594-Biometrics-Project

A command-line person retrieval baseline using pretrained **OSNet-x1.0** models
from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid). It ranks gallery
images by their similarity to a query image and evaluates identity retrieval.

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

- [Market-trained](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing): evaluate identities held out from Market-1501 training.
- [MSMT17-trained, combineall](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing): evaluate transfer to Market-1501.

Their training recipes differ, so the comparison does not isolate the effect
of the training dataset.
