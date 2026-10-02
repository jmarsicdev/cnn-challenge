# cnn-challenge

ITCS 6169/8169 Computer Vision, Assignment 1 ("The CNN Challenge"):
16-class scene recognition from 2,400 training images with a convolutional network.

**Final result: 97.25 % test accuracy** (389/400) with an ImageNet-pretrained
ConvNeXt-Tiny fine-tuned on grayscale input. Validation accuracy of the submitted
checkpoint: 97.71 % (3-seed mean of the recipe: 97.29 ± 0.42 %). Starter baseline: 46.3 %.

| | |
|---|---|
| Final config | [`configs/best.yaml`](configs/best.yaml) |
| Final checkpoint | [GitHub Release v1.0](https://github.com/jmarsicdev/cnn-challenge/releases/tag/v1.0) · `convnext_tiny_final_a_s2.pt` (107 MB) · SHA256 in [`docs/final/checkpoint.sha256`](docs/final/checkpoint.sha256) |
| Test metrics | [`docs/final/test_metrics_s2.json`](docs/final/test_metrics_s2.json), confusion matrix [`docs/final/test_confusion.png`](docs/final/test_confusion.png) |
| All experiments | [`docs/results.md`](docs/results.md) (generated, mean ± std over seeds) |
| Predictions for the unlabelled `test2` set | [`predictions/test2.csv`](predictions/test2.csv) |
| AI-tool reflection | [`AI_USAGE.md`](AI_USAGE.md) |

## Final recipe

| Component | Choice | Why it is there |
|---|---|---|
| Architecture | ConvNeXt-Tiny (timm `convnext_tiny`, 27.8 M params) | +2 to +3 points over ResNet-50 / ResNet-18 at the same recipe, stable across seeds |
| Initialisation | ImageNet-1k (timm default weights), **all** parameters fine-tuned | pretraining is worth ~10 points over from-scratch training on this data |
| Input | 224×224, **grayscale replicated to 3 channels** | 15 of 16 classes are grey; feeding colour lets the model identify Flower by colour alone (see below) |
| Augmentation | RandomResizedCrop(scale 0.35–1) + horizontal flip | stronger augmentation (TrivialAugment, RandomErasing) did not help a pretrained model here |
| Optimiser / LR | AdamW, lr 1e-4, weight decay 0.05, 1 warm-up epoch then cosine to 0 | 2× lr hurt, 0.5× lr was neutral |
| Loss | cross-entropy with **label smoothing 0.1** | the one regulariser that reliably helped (+1.0 point, 3 seeds) |
| Schedule | 40 epochs, batch 64, bf16 autocast | 40 vs 20 epochs is within noise; kept for the EMA |
| Weight averaging | EMA of weights, decay 0.995; EMA weights are what is evaluated and saved | small, consistent gain |
| Model selection | best validation-accuracy epoch on a fixed stratified 80/20 split (`splits/val_split.json`); final checkpoint = best-val seed of the final config | test set used exactly once per reported number |
| Not used | Mixup/CutMix, TTA, ensembles | mixup helped neither backbone reliably; flip-TTA slightly hurt on validation |

### The colour shortcut

All 15 scene classes in this dataset are grayscale JPEGs; only `Flower` is RGB. A
model trained on RGB input scores the same as the grayscale model (93.8 % vs 93.8 %
on validation for ResNet-18) but gets **30 % on Flower once the colour is removed**,
versus 100 % for the grayscale-trained model (`docs/probes/`). It had learned
"colourful ⇒ flower". The final pipeline therefore converts every image to grayscale.

## Experimental journey (validation accuracy, 480 images; ± is std over 3 seeds)

| Experiment | Val acc % | Observation |
|---|---|---|
| Starter TNet, grey 64 px | 46.3 | 99 % train acc by epoch 20: pure overfitting |
| ResNet-18 from scratch, 100 ep | 83.3 | 98 % train / 82 % val, still a 16-point gap |
| ResNet-18 ImageNet, frozen (linear probe) | 87.1 | ImageNet features already separate the scenes |
| ResNet-18 ImageNet, full fine-tune | 93.0 ± 0.8 | pretraining ≈ +10 points over scratch |
| + augmentation ladder (none / medium / mixup) | 93.1 / 93.6 / 92.8 | all within seed noise |
| ResNet-50 ImageNet | 94.4 ± 0.6 | capacity helps, but only visible with seeds |
| ConvNeXt-Tiny ImageNet | 96.2 ± 0.1 | architecture is the 2nd biggest lever |
| + label smoothing 0.1 | 97.2 ± 0.2 | only knob with a robust gain |
| + 40 epochs + EMA (**final**) | **97.3 ± 0.4** | test: 97.25 % (sibling seeds 96.75 / 97.0) |

Things that did not work: 40 epochs alone looked like +1.1 on one seed and vanished with
three (96.3 ± 0.9); TrivialAugment on ConvNeXt-T did nothing; Mixup/CutMix was −0.2 on
ResNet-18 and +0.6 on ConvNeXt-T, neither beyond noise. Full table: `docs/results.md`.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5080,
CUDA 12.8 wheels, driver ≥ 570). Exact package versions are pinned in `uv.lock`.

```bash
git clone https://github.com/jmarsicdev/cnn-challenge && cd cnn-challenge
uv sync                      # Python 3.12 venv with torch 2.x cu128, torchvision, timm
source .venv/bin/activate
```

**Data.** Download the Drive folder from the handout ("Download all" gives a zip) and
extract at the repo root so that `data/train/<class>/`, `data/test/<class>/` and
`data/test2/` exist. Check it:

```bash
python scripts/inspect_dataset.py --data data     # counts, grey/colour, sizes, duplicates
```

**Evaluate the released checkpoint** (reproduces 97.25 %):

```bash
curl -L -o checkpoints/convnext_tiny_final_a_s2.pt \
  https://github.com/jmarsicdev/cnn-challenge/releases/download/v1.0/convnext_tiny_final_a_s2.pt
sha256sum -c docs/final/checkpoint.sha256
python evaluate.py --checkpoint checkpoints/convnext_tiny_final_a_s2.pt --split test
python predict.py  --checkpoint checkpoints/convnext_tiny_final_a_s2.pt --input data/test2 --out predictions/test2.csv
```

**Retrain the final model** (~2.5 min on an RTX 5080; GPU nondeterminism means ±0.5 %):

```bash
python train.py --config configs/best.yaml --seed 2
python evaluate.py --checkpoint runs/best/s2/best.pt --split test
```

**Re-run any experiment** in the journey table, e.g. three seeds of the augmentation rung:

```bash
for s in 0 1 2; do python train.py --config configs/resnet18_imagenet_aug_medium.yaml --seed $s; done
python scripts/summarise_runs.py          # regenerates docs/results.md
tensorboard --logdir runs                 # live curves
```

The validation split is fixed and committed (`splits/val_split.json`, 30 images per
class, byte-identical duplicates kept on one side). `python scripts/make_split.py`
regenerates it deterministically.

## Layout

```
configs/        one YAML per experiment; best.yaml = final model
src/data.py     split loader, colour modes (gray / rgb / gray1), augmentation presets
src/models.py   timm factory + starter TNet; ImageNet / Places365 / random init; freezing
src/engine.py   AdamW/SGD, warmup+cosine, bf16, label smoothing, mixup, EMA, logging
train.py        train one config/seed  -> runs/<name>/s<seed>/{best.pt, history.csv, metrics.json, confusion.png, tb/}
evaluate.py     labelled split, --tta, --desaturate (colour-shortcut probe), --confusion-png
predict.py      CSV predictions for an unlabelled folder (data/test2)
scripts/        inspect_dataset, make_split, summarise_runs
splits/         committed train/val split
docs/           handout, dataset summary, results table, probes, final test metrics
predictions/    test2.csv from the final checkpoint
```

## Environment

Python 3.12.14 · PyTorch 2.x (cu128) · torchvision · timm 1.x · scikit-learn · matplotlib.
Each run directory contains `env.json` with the exact versions, GPU and git SHA used.
