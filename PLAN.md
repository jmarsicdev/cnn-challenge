# Plan — ITCS 6169/8169 Assignment 1: The CNN Challenge

Working document. Updated as experiments land. The handout lives in
`docs/handout/`, the untouched starter notebook in `reference/`.

## 0. Facts that constrain the plan

**The task.** 16-way scene classification from 2,400 labelled training images
(150/class). A labelled test set is provided (25/class = 400 images). Starter
baseline: grayscale 64×64, a one-conv-layer net, Adam 2e-3, 20 epochs, under 50%.
We must *substantially* beat it.

**The dataset** (verified 2026-10-01 with `scripts/inspect_dataset.py`; full
output in `docs/dataset_summary.txt`).
- `data/train/`: 2,400 images, 150/class. `data/test/`: 400 labelled, 25/class.
- `data/test2/`: **400 unlabelled images** (`image_N.jpg`), not mentioned in the
  handout. Same composition as test (375 grey + 25 colour, same size mix), no
  byte-identical overlap with train or test. Almost certainly a hidden-label
  test set for the leaderboard. **Plan for it:** `predict.py` must emit a CSV of
  predictions for `test2`, and we never look at it during development.
- Classes are the classic **15-Scene** set (Lazebnik, Schmid & Ponce 2006) plus
  **Flower**. The 15 scene classes are **100% grayscale** (PIL mode `L`); Flower
  is **100% RGB** and genuinely colourful. So *"does the image have colour?"*
  identifies Flower perfectly. Consequences:
  - RGB input gives a free, non-visual shortcut for 1/16 of the data. For an
    honest 16-way recogniser the default pipeline should **convert everything to
    grayscale and replicate to 3 channels** for pretrained nets; Flower stays
    easy by texture/shape anyway. Report the RGB variant as an ablation. *(This
    is a decision for us, not the AI; record it in `AI_USAGE.md`.)*
  - ImageNet/Places pretrained filters expect colour statistics; grey-replicated
    input still transfers well in practice, but it is a thing to measure.
- Resolution: half the images are exactly 256×256, most of the rest ~293×220 or
  330×220 (4:3-ish), a few 666×500. Native detail tops out around 256 px, so
  224 is near-native and 320+ is pure upsampling. Aspect ratios are mixed, so
  Resize-shorter-side + crop vs. squash is a real choice.
- Duplicates: 6 exact-duplicate pairs *within* train (same class), 0 across
  train/test, 1 near-duplicate Bedroom pair across train/test. Negligible, but
  the within-train pairs should not straddle our train/val split (handle in the
  split script).
- 15-Scene is well studied: published pretrained-CNN results are high-80s to
  mid-90s %. Places365-pretrained CNNs do especially well because Places *is* a
  scene dataset. Hypothesis to test, not assume.

**Hardware.** Local RTX 5080 (16 GB, Blackwell), 24 CPU cores, PyTorch cu128 in
`.venv`. Enough to fine-tune ResNet-50 / ConvNeXt-T at 224–320 in minutes per run,
so multi-seed experiments are affordable. No need for Colab.

**Rules.** Primary classifier must be a CNN. Pretrained CNNs allowed if we state
what/where/what-was-tuned. Banned as primary: CLIP, DINO/DINOv2, ViTs, VLM APIs,
foundation embeddings. Test set is for the *final* number only.

**Grading.** Accuracy 30 / experimental reasoning 25 / design 20 / repro+GitHub 15
/ AI usage 10. Reasoning + design + repro = 60%. The story matters more than the
last 0.5%.

**Deadline.** Handout says Sep 25, 2026 (stale). **Actual deadline: Oct 2, 2026,
night.** That is ~24 hours from the start of work, so §5 below is the schedule
that actually governs; §2–3 are the full menu we are cutting from.

**Statistics we must respect.** With a 480-image validation set, one image is
0.21% and the standard error of an 85% accuracy is ≈1.6%. Differences under
~3% on a single split/seed are noise. Any claim in the report about a "small"
improvement needs either multiple seeds or k-fold evidence.

## 1. Validation strategy (decide once, never touch test)

- **Stratified 80/20 split** of the 2,400 training images (120 train / 30 val per
  class), fixed seed, written to `splits/val_split.json` and committed. The 6
  exact-duplicate pairs are kept on the same side of the split. Every
  experiment uses the identical split so numbers are comparable.
- **Model selection** = best validation accuracy epoch (as in the starter), with
  val loss logged too so we can see overconfidence.
- For the final shortlist (2–4 configs), run **5-fold stratified CV** or 3 seeds
  to confirm the ranking before touching test.
- Final model: train on the chosen config with the fixed split (keeps the
  reported val number honest). Optionally also report a "train on all 2,400,
  fixed epoch count" variant, clearly labelled, if CV shows it is safe.
- The test set is evaluated **once per final candidate**, via `evaluate.py`.

## 2. Phases

### Phase A — Infrastructure (first)
Everything later depends on being able to run a controlled experiment from one
command and get a comparable number back.
- `src/data.py` — ImageFolder + split file, transform presets (named, so configs
  reference `aug: light|medium|strong`), grayscale-aware normalisation.
- `src/models.py` — factory over `timm` (resnet18/50, convnext_tiny,
  efficientnet_b0/b2, mobilenetv3) + the starter TNet + a from-scratch ResNet.
  Pretrained source (imagenet / places365 / none) is a config field.
- `src/engine.py` — train/eval loops, AMP, cosine+warmup, EMA, label smoothing,
  mixup/cutmix switches, best-val checkpointing, per-epoch CSV log.
- `train.py --config configs/x.yaml` and `evaluate.py --checkpoint ...` with
  seeds, env/version dump into the run directory.
- `scripts/summarise_runs.py` — builds the experiment table (markdown) from run
  logs, so the report table is generated, not hand-typed.
- Deliverable: reproduce the starter TNet number in the new codebase.

### Phase B — Baselines that establish the big axes
1. Starter TNet, as given (grey 64).
2. Starter-style small CNN but RGB/grey 224 → "does resolution alone matter?"
3. ResNet-18 **from scratch**, 224, standard aug.
4. ResNet-18 **ImageNet-pretrained**, linear probe vs full fine-tune.
Expected: pretraining is the single biggest jump. Quantify it; it's the opening
line of the Experimental Journey.

### Phase C — Controlled experiments (one variable at a time)
Run on the fixed split; replicate the ones that end up in the report.
- **Augmentation ladder:** none → flip+RandomResizedCrop → TrivialAugment /
  RandAugment → +mixup/cutmix. Hypothesis: moderate aug helps; mixup may hurt
  short fine-tunes.
- **Pretraining source:** ImageNet-1k vs Places365 (scene-specific) vs none.
- **Capacity:** ResNet-18 → 50, ConvNeXt-T, EfficientNet-B0/B2, MobileNetV3.
  Record params/FLOPs/latency for the efficiency story.
- **Resolution:** 128 / 224 / 320 for the best family. Cost vs gain.
- **Fine-tuning recipe:** lr for backbone vs head, layer-wise LR decay,
  freeze-then-unfreeze, BN-only tuning, SGD vs AdamW, warmup + cosine, epochs.
- **Regularisation:** label smoothing, weight decay, drop-path, dropout.
- **Cheap wins at the end:** EMA weights, TTA (flip + multi-crop), checkpoint
  averaging. Small ensemble (2–3 seeds) as an "accuracy" leaderboard entry if we
  want it, clearly separated from the single-model result.

### Phase D — Analysis
- Confusion matrix + per-class accuracy; which scene pairs collapse (Bedroom vs
  LivingRoom, Kitchen vs Office, InsideCity vs Street vs TallBuilding, Coast vs
  OpenCountry are the classic confusions).
- Grad-CAM on a handful of errors for the report figure.
- Learning curve (25/50/100/150 imgs per class) for pretrained vs scratch.
- Seed-variance study: this is likely our **"Most Interesting Finding"** entry.

### Phase E — Final selection, writing, release
- CV/multi-seed confirm shortlist → pick → one test evaluation per candidate.
- Upload checkpoint (HF Hub or GitHub Release), pin SHA in README.
- Two-page report (LaTeX in `report/`): Final Result, Secret Recipe, Journey
  table, Failure Analysis, AI+Human.
- Condense `AI_USAGE.md`; tidy README; verify a clean clone reproduces.
- Commit after every experiment: the commit history is graded.

### Minimum viable submission (if time runs short)
Phases A, B, the augmentation ladder, one capacity comparison, EMA+TTA, analysis
of one failure, report. That already tells a complete story.

## 3. Candidate directions — the interesting ones

Ranked roughly by (expected insight × leaderboard relevance) / cost.

1. **Places365 vs ImageNet initialisation.** Scene data + scene-pretrained CNN.
   Clean hypothesis, directly tests "does *what* was pretrained matter, not just
   *that* it was pretrained". Cheap: swap weights. Strong report material.
2. **How much signal is there in a single-split result?** Train the same config
   with 5–10 seeds; plot the spread; show which of our "improvements" survive.
   Most students won't do this; it maps straight onto the 25% reasoning weight
   and the Most Interesting Finding award.
3. **The colour shortcut.** Confirmed: 15/16 classes are grayscale, Flower is
   the only colour class. Compare grey-replicated vs RGB input; check whether an
   RGB model's Flower accuracy survives desaturated Flower test images. Cheap,
   concrete, and a good "dataset understanding beat the default pipeline" story.
4. **Accuracy–efficiency Pareto front.** MobileNetV3 / EfficientNet-B0 /
   ResNet-18 / ResNet-50 / ConvNeXt-T at 128–320 px, plotted as accuracy vs
   FLOPs and vs measured latency. Targets the Efficiency award and the handout's
   "can a smaller model do almost as well?" question.
5. **Knowledge distillation** from the best large model into MobileNetV3. If it
   closes most of the gap, it is both an efficiency entry and a nice story.
6. **Which parameters to fine-tune.** Linear probe → last stage → full →
   layer-wise LR decay → BN-affine-only. Fine-tuning dynamics on 2,400 images
   is exactly what the course objective asks us to reason about.
7. **Learning-curve study.** Accuracy vs training images/class for scratch vs
   pretrained. Shows where the pretraining gap comes from and whether more data
   would even help.
8. **Augmentation ablation with mixup/cutmix as the planned "failure".** Prior:
   heavy mixing regularisers often hurt short fine-tunes of pretrained nets on
   small data. If that reproduces, it is a textbook failure-analysis section
   (what/why/what happened/what learned).
9. **From-scratch ceiling with a modern small ResNet.** How far can a
   well-regularised scratch CNN get on 2,400 images (long schedule, strong aug,
   SAM)? Answers "does ImageNet init help *substantially*" with a real number.
10. **Dataset hygiene.** Near-duplicate detection within train and across
    train/test; label-noise spot check via high-loss examples. Low cost, and if
    duplicates exist it changes how we interpret every number.
11. **Error structure.** Confusion matrix, per-class accuracy, Grad-CAM on
    confusions. Needed for the report regardless; may suggest a targeted fix
    (e.g. higher resolution for indoor classes).
12. **Sharpness-Aware Minimisation.** Reported to help small-data fine-tuning;
    costs 2× compute. Medium interest; good secondary failure candidate if it
    does nothing.
13. **Test-time tricks:** EMA, TTA, checkpoint averaging. Low insight, reliable
    gain. Include, don't centre the story on them.
14. **Multi-seed ensemble** for a leaderboard "best accuracy" entry, reported
    separately from the single-model result.

### Deliberately out of scope / risky
- Self-supervised CNN checkpoints (MoCo/SimCLR/BYOL ResNets): not banned by name,
  but adjacent to the "foundation-model embeddings" ban. Ask the instructor if
  we ever want this; otherwise skip.
- Any transductive use of the test images (pseudo-labelling, test-set BN
  statistics adaptation). Violates the spirit of "test once".
- Giant backbones (ConvNeXt-L and up). Marginal gain, poor efficiency story.

## 4. Decisions taken

- **Colour (2026-10-01, user decision):** primary pipeline converts every image to
  grayscale and replicates to 3 channels, so Flower cannot be identified by
  colour alone. RGB input is run as an ablation, plus a "desaturated Flower"
  test to show whether the RGB model relies on the shortcut.

## 5. 24-hour schedule (deadline Oct 2 night)

Each training run on the RTX 5080 at 224 px is 1–3 minutes, so compute is not
the bottleneck; writing is. Reserve the last 4 hours for the report.

**Tonight (≈3 h) — infrastructure + the big axes**
1. Phase A code: `src/`, `train.py`, `evaluate.py`, `predict.py` (test2 CSV),
   committed split, run logging, run-summary table. Commit.
2. Reproduce starter TNet (grey 64) in the new code → first Journey row.
3. ResNet-18 from scratch @224 vs ResNet-18 ImageNet-pretrained @224. Commit.

**Tomorrow daytime (≈5 h) — controlled experiments, 1 variable each**
4. Augmentation ladder on the pretrained ResNet-18: none → flip+crop →
   TrivialAugment → +mixup/cutmix (expected failure-analysis candidate).
5. Colour ablation: grey-replicated vs RGB; RGB model on desaturated Flower.
6. Capacity: ResNet-50, ConvNeXt-T (same recipe). Record params/latency.
7. Recipe knobs on the leader: LR, label smoothing, weight decay, epochs, EMA.
8. Re-run the 2–3 best configs with 3 seeds. Only seed-robust gains go in the
   report. Commit after every step; `scripts/summarise_runs.py` builds the table.
9. *Stretch, only if ahead:* Places365-initialised ResNet-50; TTA.

**Tomorrow evening (≈4 h) — finalise + write**
10. Pick final config on val → train final → `evaluate.py` on test **once** →
    `predict.py` on test2 → upload checkpoint to a GitHub Release, pin SHA256.
11. Confusion matrix + per-class accuracy figure for the report.
12. Two-page report, `AI_USAGE.md` condensed, README reproduction section.
    Fresh-clone test: `uv sync && python evaluate.py --checkpoint ...` works.

**Cut for time:** k-fold CV (3 seeds instead), distillation, SAM, learning-curve
study, Grad-CAM (optional if a figure slot is free), ensembles.
