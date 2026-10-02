# AI Usage

Required by the handout (§4). Written as engineering/research interactions rather
than a prompt transcript.

## Tool

**Claude Code** (Anthropic, model Claude Fable 5.1), run as a terminal agent in this
repository for scaffolding, boilerplate, debugging, experiment plumbing and
drafting documentation. All experimental decisions, verification and the final
write-up are mine.

## Representative examples of how AI assisted

1. **Project scaffolding and environment.** The AI set up the `uv` project, chose
   the CUDA 12.8 PyTorch build needed for an RTX 5080 (Blackwell, `sm_120`) after
   the system Python 3.14 turned out to be unsupported, and wrote the config-driven
   `train.py` / `evaluate.py` / `predict.py` pipeline with YAML inheritance, CSV +
   TensorBoard logging and best-validation checkpointing.
   *Verification:* I ran a 2-epoch smoke test of the starter network through the
   new pipeline and checked that `evaluate.py` on the same checkpoint reproduced
   the training-time validation number before any real experiment was trusted.

2. **Dataset inspection script.** The AI wrote `scripts/inspect_dataset.py`
   (per-class counts, PIL colour mode, size histogram, md5 and perceptual-hash
   duplicate check). Its output is what revealed that the 15 scene classes are
   grayscale and only Flower is RGB, and that six exact duplicate pairs exist in
   the training set. The split script keeps duplicates on one side as a result.

3. **Run-summary tooling.** `scripts/summarise_runs.py` aggregates every run's
   `metrics.json` into a mean ± std table over seeds. The report's experiment table
   is generated from it rather than typed by hand, so numbers are traceable to
   run directories.

4. **Experiment batching.** The AI drafted the sequence of controlled experiments
   (augmentation ladder, capacity, recipe knobs) as config files and ran them as
   background batches while I watched the curves in TensorBoard and asked
   questions about what each curve meant.

## Incorrect / ineffective / questionable AI suggestions

### 1. Self-deadlocking experiment queue
- **Suggestion:** To chain experiment batches on the single GPU, the AI wrote a
  background shell loop that waited `until` the previous batch's last
  `metrics.json` existed *and* `pgrep -f "train.py --config"` found no process.
- **Why it looked plausible:** Standard "wait until the GPU is free" idiom.
- **What was wrong:** The waiting shell's own command line contained the string
  `train.py --config`, so `pgrep -f` always matched the waiter itself. Both queued
  batches sat idle for ~20 minutes with the GPU at 0 % while the AI reported them
  as "queued" and "running automatically".
- **How I detected it:** I asked for a progress check. No new run directories had
  appeared and `nvidia-smi` showed no utilization, contradicting the status.
- **What was done instead:** Killed both waiters and launched one sequential job.
  Lesson: an assistant's status claim is not evidence; check artifacts (run
  directories, GPU utilization, TensorBoard) before believing "running".

### 2. "40 epochs helps" from a single seed
- **Suggestion:** After the first ConvNeXt-T knob sweep, the AI ranked a 40-epoch
  schedule as the second-best knob (+1.1 points) and proposed folding it into the
  final recipe.
- **What was wrong:** With seeds 1 and 2 the 40-epoch run averaged 96.3 ± 0.9, i.e.
  no better than the 20-epoch base (96.2 ± 0.1) and noisier. The single-seed gain
  was luck.
- **How detected:** The replication step I had insisted on earlier for exactly this
  reason. The AI's own earlier analysis had said single-seed differences under
  ~3 points are noise, and then it ranked knobs by single seeds anyway.
- **Outcome:** The final recipe keeps 40 epochs only because EMA benefits from the
  longer averaging window; it is documented as "within noise", not as a gain.

## Decisions I made rather than accepting the AI recommendation

- **Grayscale input as the primary pipeline.** When the inspection showed that
  only Flower is in color, the AI presented both options (RGB for maximum
  leaderboard accuracy, or grayscale for an honest 16-way recognizer) and left the
  choice to me. I chose grayscale as the default and asked for the RGB model to be
  kept as an ablation with a de-saturation probe. The probe showed the RGB model
  drops from 100 % to 30 % on Flower when color is removed, while the grayscale
  model is unaffected, which became a central finding of the report.
- **Backbone scope.** I chose to center the comparison on ResNet-18 vs ResNet-50
  vs ConvNeXt-Tiny with ImageNet initialization and to treat Places365 only as a
  stretch goal.
- **Selecting on 3-seed means, not best single run.** Final configuration was
  chosen by mean validation accuracy over three seeds; the submitted checkpoint is
  then the best-validation seed of that configuration, and the sibling seeds'
  test accuracies are reported alongside it.

## Habit followed when accepting generated code

Before running AI-written code: what do I expect it to do, what tensor-shape or
data assumption does it make, how will I know if it is wrong, and what experiment
distinguishes a real improvement from noise?
