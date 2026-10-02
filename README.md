# cnn-challenge

ITCS 6169/8169 Computer Vision — Assignment 1, "The CNN Challenge":
16-class scene recognition from 2,400 training images using a CNN.

> Status: project scaffold. Training code lands in Phase A (see `PLAN.md`).

## Layout

```
configs/      YAML experiment configs (one file per experiment; configs/best.yaml = final)
src/          library code: data, models, training engine
train.py      python train.py --config configs/<name>.yaml
evaluate.py   python evaluate.py --checkpoint <path>
scripts/      utilities (dataset inspection, run summaries, plots)
splits/       committed train/val split so every run is comparable
runs/         per-run logs, metrics, configs (git-ignored except summaries)
checkpoints/  model weights (git-ignored; final checkpoint is linked below)
data/         dataset (git-ignored) — data/train/<class>/*, data/test/<class>/*
docs/handout/ assignment PDF + extracted text
reference/    untouched starter notebook
PLAN.md       experimental plan and candidate directions
AI_USAGE.md   required AI-tool reflection (running log)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an
RTX 5080, CUDA 12.8 wheels).

```bash
uv sync --extra dev          # creates .venv with Python 3.12 + torch cu128 + timm
source .venv/bin/activate
```

### Data

Download the dataset from the link in the handout (a Google Drive folder named
`data`; "Download all" from the browser gives a zip) and extract it at the repo
root so that these exist:

```
data/train/<class>/*.jpg   2,400 images, 16 classes × 150
data/test/<class>/*.jpg      400 images, 16 classes × 25   (labelled, final eval only)
data/test2/*.jpg             400 images, unlabelled         (predictions submitted separately)
```

Then sanity-check it (prints class counts, grey/colour split, sizes, duplicates):

```bash
python scripts/inspect_dataset.py --data data
```

Note: the 15 scene classes are grayscale JPEGs; only `Flower` is RGB.

## Reproducing the final result

_To be filled in once a final model exists:_ exact config, seed, checkpoint
link + SHA256, and the `train.py` / `evaluate.py` commands.

## Results

_Experiment table is generated from `runs/` by `scripts/summarise_runs.py`._

## Environment

Python 3.12, PyTorch ≥ 2.7 (cu128), torchvision, timm. Exact versions are
pinned in `uv.lock` and dumped into each run directory.
