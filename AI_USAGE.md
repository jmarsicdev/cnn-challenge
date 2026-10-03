# AI Usage

## What I used

**Claude Code** running as a terminal agent inside this repo.
I used it for setup, boilerplate, debugging, running experiment batches, and drafting docs.
I made the calls on what to test, checked the results, and wrote the final report.

## How AI helped

1. **Setup and the training pipeline.** It set up the `uv` project, figured out that my
   system Python 3.14 wasn't supported by PyTorch yet and that my RTX 5080 needed the CUDA
   12.8 build, and wrote the config-driven `train.py` / `evaluate.py` / `predict.py` with
   YAML config inheritance, CSV and TensorBoard logging, and best-validation checkpointing.
   Before I trusted any of it, we ran a 2-epoch smoke test of the starter network through
   the new code and confirmed `evaluate.py` reproduced the same validation number the
   training loop printed.

2. **Dataset inspection.** It wrote `scripts/inspect_dataset.py` (class counts, PIL color
   mode, image sizes, md5 and perceptual-hash duplicate checks). That script is how we
   found out that the 15 scene classes are all grayscale and only Flower is in color, and
   that the training set has six exact duplicate pairs. The split script keeps those
   duplicates on one side because of that.

3. **Results table generation.** `scripts/summarize_runs.py` pulls every run's
   `metrics.json` into a mean ± std table across seeds. The experiment table in the report
   comes from that script.

4. **Running controlled experiments.** It turned the experiment plan into config files
   (augmentation ladder, model capacity, recipe knobs) and ran them as background batches
   while I watched the curves in TensorBoard and asked for clarification as needed.

## Where the AI was wrong or not helpful

### 1. An experiment queue that deadlocked itself
- **What it did:** To chain experiment batches on my one GPU, it wrote a background shell
  loop that waited until the last `metrics.json` from the previous batch existed *and*
  `pgrep -f "train.py --config"` found no running process.
- **Why it seemed fine:** That's the normal "wait for the GPU to free up" pattern.
- **What was actually wrong:** The waiting shell's own command line contained the string
  `train.py --config`, so `pgrep` always matched the waiter itself. Two batches (the seed
  replication and the ConvNeXt recipe sweep) sat idle for about 20 minutes with the GPU at
  0% while the AI told me they were "queued" and "running automatically".
- **How I caught it:** I asked for a progress check. No new run folders had shown up and
  `nvidia-smi` showed nothing running, which didn't match what it was telling me.
- **Fix:** Killed both waiters and launched one sequential job. So a status update
  from the assistant isn't 100%. Check the run folders, GPU usage, or TensorBoard.

### 2. "40 epochs helps," based on one seed
- **What it did:** After the first ConvNeXt-T knob sweep it ranked a 40-epoch schedule as
  the second-best change (+1.1 points) and suggested putting it in the final recipe.
- **What was actually wrong:** With seeds 1 and 2 the 40-epoch run averaged 96.3 ± 0.9,
  no better than the 20-epoch base (96.2 ± 0.1) and noisier. The single-seed number was
  luck.
- **How I caught it:** The seed replication step we had planned for this reason.
  
- **Outcome:** The final recipe keeps 40 epochs only because the EMA benefits from a longer
  averaging window. It's documented as "within noise," not as a gain.

### 3. Evaluating twice per epoch no matter how short the epoch is
- **What it did:** The training loop evaluated the model and its EMA copy on all 480
  validation images after every epoch. That was fine for the main experiments (30 steps
  per epoch). For the low-label runs an epoch is 5 to 20 steps, so the first batch spent
  most of its time evaluating: the 10-labels-per-class run was on pace for 15 minutes
  instead of 3, and the whole batch for several hours.
- **How I caught it:** The run was at epoch 112 of 200 after 8 minutes while a full
  ConvNeXt fine-tune on 6x more data takes 1 minute. The per-epoch time in `history.csv`
  made it obvious the cost wasn't the training steps.
- **Fix:** Added an `eval_every` option, set it so the low-label runs evaluate about 50
  times total, killed the batch and restarted it so all seeds use the same cadence.

## Decisions I made (not just accepted)

- **Grayscale input as the main pipeline.** Once inspection showed only Flower is in color,
  I set up an experiment to compare and test the potential for the shortcut. I picked grayscale as the
  default and asked to keep the RGB model as an ablation with a de-saturation test.
  That test showed the RGB model drops from 100% to 30% on Flower when color is removed
  while the grayscale model doesn't move, which became one of the main findings in the report.
- **Backbone scope.** I chose to focus on ResNet-18 vs ResNet-50 vs ConvNeXt-Tiny with
  ImageNet weights and treat Places365 as a stretch goal.
- **Pick the final config by 3-seed mean, not best single run.** The final configuration
  was chosen on average validation accuracy over three seeds. The submitted checkpoint is the
  best-validation seed of that config, and the other two seeds' test accuracies are
  reported next to it.
- **Semi-supervised twist.** The teacher-student (EMA teacher) experiment on a low-label
  split was the idea I have been most interested since learning about it in class;
  the AI's role was to implement it.



