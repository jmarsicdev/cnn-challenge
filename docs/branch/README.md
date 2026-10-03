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

## 3. Extended-budget runs (in progress)

Queue: `queues/extended_scratch_s0.txt`, log: `runs/queue_extended_scratch_s0.log`.
Done so far: all-labels scratch ceiling at ~10k steps = **85.0** (vs 83.3 at 3k).
- `semisup_r18sc_ts_k20_long` (10k steps): **76.3** vs 75.8 at 3k steps. Best epoch 486/625; final mask rate 0.91, pseudo-label accuracy 0.756. Tripling the budget bought ~0.5 points: the scratch teacher-student runs are limited by pseudo-label quality, not steps.
