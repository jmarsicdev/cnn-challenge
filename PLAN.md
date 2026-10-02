# Plan — ITCS 6169/8169 Assignment 1: The CNN Challenge

Working document. Updated as experiments land. The handout lives in
`docs/handout/`, the untouched starter notebook in `reference/`.

## 0. Facts that constrain the plan

**The task.** 16-way scene classification from 2,400 labelled training images
(150/class). A labelled test set is provided (25/class = 400 images). Starter
baseline: grayscale 64×64, a one-conv-layer net, Adam 2e-3, 20 epochs, under 50%.
We must *substantially* beat it.

**The dataset** (from the Drive folder; confirm with `scripts/inspect_dataset.py`).
Class names are the classic **15-Scene** categories (Lazebnik, Schmid & Ponce 2006:
Bedroom, Coast, Forest, Highway, Industrial, InsideCity, Kitchen, LivingRoom,
Mountain, Office, OpenCountry, Store, Street, Suburb, TallBuilding) **plus Flower**.
Important consequences, to be verified once the download finishes:
- 15-Scene images are **natively grayscale** and small (~200–300 px). Colour may
  only exist in the Flower class, which would make colour a trivial shortcut for
  one class and irrelevant for the rest. This changes what "use RGB" buys us and
  argues for grayscale→3-channel replication for pretrained nets.
- 15-Scene is a well-studied benchmark. Published pretrained-CNN results are in the
  high-80s to mid-90s%. Places365-pretrained CNNs do especially well on it because
  Places *is* a scene dataset. That is a hypothesis worth testing, not assuming.
- 15-Scene has known near-duplicate images across splits. Dedup check before
  trusting val numbers.

**Hardware.** Local RTX 5080 (16 GB, Blackwell), 24 CPU cores, PyTorch cu128 in
`.venv`. Enough to fine-tune ResNet-50 / ConvNeXt-T at 224–320 in minutes per run,
so multi-seed experiments are affordable. No need for Colab.

**Rules.** Primary classifier must be a CNN. Pretrained CNNs allowed if we state
what/where/what-was-tuned. Banned as primary: CLIP, DINO/DINOv2, ViTs, VLM APIs,
foundation embeddings. Test set is for the *final* number only.

**Grading.** Accuracy 30 / experimental reasoning 25 / design 20 / repro+GitHub 15
/ AI usage 10. Reasoning + design + repro = 60%. The story matters more than the
last 0.5%.

**Deadline discrepancy.** The handout says **Sep 25, 2026**, which is already
past (today is Oct 1, 2026). Either the date is stale or an extension applies.
**Confirm the real deadline with the instructor before planning the schedule.**

**Statistics we must respect.** With a 480-image validation set, one image is
0.21% and the standard error of an 85% accuracy is ≈1.6%. Differences under
~3% on a single split/seed are noise. Any claim in the report about a "small"
improvement needs either multiple seeds or k-fold evidence.

## 1. Validation strategy (decide once, never touch test)

- **Stratified 80/20 split** of the 2,400 training images (120 train / 30 val per
  class), fixed seed, written to `splits/val_split.json` and committed. Every
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
3. **Does colour matter at all here?** If 15 of 16 classes are grayscale, RGB
   input is 3× redundant channels except for Flower. Test grey-replicated vs
   RGB, and check whether the model is "cheating" on Flower via colour alone
   (train with Flower desaturated). Possibly a surprising, cheap finding.
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

## 4. Next actions

1. Finish dataset download → `data/` → run `scripts/inspect_dataset.py` →
   amend §0 with verified facts (grey vs colour, sizes, duplicates).
2. Build Phase A and reproduce the starter number.
3. Run Phase B in one evening; write the first Journey-table rows.
4. Confirm the real deadline.
