# consistency-learning branch: results log

Post-submission experiments, for interest only. Nothing here is part of the graded submission on `main`.
All numbers are validation accuracy (%) on the same fixed 480-image split as the main study.

## 1. Does the unlabeled pool matter more without pretraining? (seed 0)

Same ResNet-18 in both arms; only the initialization differs. Low-label protocol as in the
main study: keep k labels per class, treat the other training images as an unlabeled pool
(never test images). Teacher-student = EMA teacher pseudo-labels weak views, student learns
them on strong views, tau 0.8, burn-in first. Budgets: pretrained ~1,000 steps, scratch ~3,000.

| ResNet-18, labels/class | 10 | 20 | 40 | 120 (all labels) |
|---|---|---|---|---|
| ImageNet init, supervised | 83.1 | 86.9 | 91.7 | 93.0 ± 0.8 |
| ImageNet init, teacher-student | 84.0 | 89.2 | 92.7 | |
| **gain from unlabeled pool** | +0.8 | +2.3 | +1.0 | |
| Scratch, supervised | 60.4 | 68.3 | 76.3 | 83.3 (3k steps) / 85.0 (10k steps) |
| Scratch, teacher-student | 67.9 | 75.8 | 80.6 | |
| **gain from unlabeled pool** | **+7.5** | **+7.5** | **+4.4** | |

Figures: `lowlabel_r18_imagenet.png`, `lowlabel_r18_scratch.png`.

Reading it:
- With ImageNet weights the pool buys 1–2 points, like ConvNeXt in the main study. Without
  them it buys 4–8 points. The pretrained features already encode most of what the unlabeled
  images could teach; from scratch, the pool is the main source of signal.
- From scratch with 40 labels per class plus the pool (80.6) nearly matches training on all
  120 labels per class (83.3 at the same step budget). Two-thirds of the labels were
  replaceable by consistency training.
- Teacher diagnostics at the end of the scratch runs: mask rate 0.57 / 0.62 / 0.65 and
  pseudo-label accuracy 82 / 88 / 92 % for k = 10 / 20 / 40. The scratch teacher is both
  less confident and less accurate than the pretrained one (which reached ~98 % mask rate at
  ~95 % accuracy), so these runs are still limited by pseudo-label quality, and all three
  were still improving at the end of the budget. Extended (~10k-step) runs are in progress.
- Single seed so far. Seed-to-seed spread for scratch runs is probably larger than the
  0.6–0.8 seen for pretrained ones; replication pending.

## 2. Distillation into a small custom student (seed 0, 100 epochs ≈ 3k steps)

| Model, all from scratch | Val acc | Params | Best epoch |
|---|---|---|---|
| ResNet-18, labels only | 83.3 | 11.2M | 80/100 |
| ResNet-18, distilled from ConvNeXt-T | 85.2 | 11.2M | 94/100 |
| SceneNet-S, labels only | 84.0 | 0.86M | 100/100 |
| SceneNet-S, distilled from ConvNeXt-T | 84.8 | 0.86M | 100/100 |
| ConvNeXt-T teacher (ImageNet init) | 97.7 | 27.8M | |

SceneNet-S (`src/models.py`): depthwise-separable inverted-residual CNN, 4 stages, ~181 MMACs
at 224 px (ResNet-18 ≈ 1,800). Distillation: KL to the teacher's temperature-2 softmax on the
same augmented batch, weight 0.7. Both distilled runs peaked on their last epoch; 300-epoch
versions are queued.

## 3. Extended-budget runs (seed 0, finished 2026-10-03 00:02)

### 3a. From-scratch low-label study at ~10k steps (vs ~3k above); 10k rows are mean ± std over 3 seeds

| ResNet-18 scratch, labels/class | 10 | 20 | 40 | 120 (all) |
|---|---|---|---|---|
| supervised, 3k steps (seed 0) | 60.4 | 68.3 | 76.2 | 83.3 |
| supervised, 10k steps (3 seeds) | 63.1 ± 0.6 | 70.8 ± 0.1 | 77.8 ± 0.5 | 84.8 ± 0.6 |
| teacher-student, 3k steps (seed 0) | 67.9 | 75.8 | 80.6 | |
| teacher-student, 10k steps (3 seeds) | **70.9 ± 2.9** | **77.3 ± 0.9** | **83.5 ± 1.2** | |
| gain from pool at 10k (mean) | +7.8 | +6.5 | +5.8 | |

Final teacher diagnostics at 10k steps (mask rate / pseudo-label accuracy): k=10: 0.83 / 0.742, k=20: 0.91 / 0.756, k=40: 0.94 / 0.843.

Figure: `lowlabel_r18_scratch_long.png`.

Reading it: the longer budget lifts the supervised floors by 1–3 points and the teacher-student runs by
0.5–4 points, so the pool's advantage holds at +5 to +8 points. At 40 labels per class with the pool
(82.3) the model is within 2.7 points of using all 120 labels (85.0). The teacher's
pseudo-label accuracy at the end is well below its coverage in every case, so confirmation bias remains the
limiting factor; the fixes listed above are the next experiments if this continues.

### 3b. Distillation at 300 epochs (~9k steps)

| Model, all from scratch | 100 epochs | 300 epochs | Params |
|---|---|---|---|
| ResNet-18, labels only | 83.3 | 85.0 | 11.18M |
| ResNet-18, distilled | 85.2 | 86.0 | 11.18M |
| SceneNet-S, labels only | 84.0 | 88.7 | 0.86M |
| SceneNet-S, distilled | 84.8 | 89.0 | 0.86M |

(ResNet-18 labels-only at 300 epochs is the `resnet18_scratch_xl` run, 333 epochs.) Best epochs at 300: SceneNet-S labels-only 135, distilled 250.

Reading it: with a long enough schedule the 0.86M-parameter SceneNet-S reaches ~89 % from scratch, above
ResNet-18 from scratch (85.0) at a 13x smaller parameter count, and the distillation gain shrinks to
+0.2 points for the student and +1.0 for ResNet-18. The soft targets mostly accelerate
learning rather than raise the ceiling here; the architecture choice mattered more than the teacher.

## 4. Confirmation-bias fixes (k = 20 labels/class, scratch, ~10k steps, seed 0)

The baseline teacher ended at 91 % coverage of the pool but only 76 % pseudo-label accuracy (section 3a).
Each row changes one thing relative to `semisup_r18sc_ts_k20_long`.

| Variant | Val acc | Best epoch | Teacher at end: coverage / pseudo-label acc |
|---|---|---|---|
| Baseline: tau 0.8, EMA 0.995, hard pseudo-labels | 76.2 | 486 | 0.91 / 0.756 |
| Threshold tau 0.95 | 70.8 | 252 | 0.64 / 0.817 |
| Slower teacher, EMA 0.999 | 76.5 | 474 | 0.85 / 0.800 |
| **Distribution alignment** (teacher probs rebalanced to the uniform prior) | **81.0** | 480 | 0.92 / 0.822 |
| All three combined | 72.7 | 612 | 0.23 / 0.966 |
| Mean Teacher soft MSE consistency, no threshold, lambda 10 ramped, EMA 0.999 | 77.7 | 564 | 1.00 / 0.742 |

Reading it:
- Distribution alignment is the fix: +4.8 points, with coverage unchanged (0.92) and pseudo-label accuracy up
  from 0.76 to 0.82. The failure mode was class imbalance in the pseudo-labels: the teacher over-predicted some
  classes, the student learned the bias, the averaged teacher inherited it. Rebalancing to the known uniform prior
  breaks that loop. (Classes are exactly balanced here, which is the easiest case for this fix.)
- A stricter threshold went the wrong way. At tau 0.95 the teacher labeled only ~17 % of the pool when the run
  peaked, so the student saw too little unlabeled data; accuracy of the few labels was high (97 %) but useless.
  Combining all three inherited this problem (coverage 0.23).
- A slower teacher alone did nothing (+0.3). Soft MSE consistency without a threshold gave +1.5 and never filters,
  so its 'pseudo-label accuracy' is just teacher accuracy on the pool (0.74).
- Next experiments, not yet run: distribution alignment at k = 10 and 40, DA + tau 0.9, DA with the slower teacher,
  and a per-class pseudo-label histogram in the diagnostics to see the imbalance directly.
