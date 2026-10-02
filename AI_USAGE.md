# AI Usage

> Required by the assignment handout (Section 4). This file is kept as a running
> log during development and condensed before submission. Entries are written as
> *engineering/research interactions*, not "I asked for code".

## Tools used

- **Claude Code** (Anthropic, model Claude Fable 5.1) run as a terminal agent for
  project scaffolding, boilerplate, debugging, and experiment plumbing.
- *(add others here if used: Copilot, ChatGPT, Cursor, ...)*

## Representative examples of AI assistance (target 3–5 in the final version)

| # | Date | What AI did | How it was verified / modified |
|---|------|-------------|-------------------------------|
| 1 | 2026-10-01 | Read the handout + starter notebook, proposed a phased experimental plan and a list of candidate directions (`PLAN.md`), scaffolded the repo and the `uv` environment (CUDA 12.8 PyTorch for the RTX 5080), and wrote a dataset inspection script. | Plan reviewed and pruned by me; the dataset facts the script printed (class counts, grey vs colour images, duplicates) were checked against the raw folders before being used to make decisions. |
| 2 | | | |
| 3 | | | |

## Incorrect, ineffective, or questionable AI suggestions

Record at least one. Template:

- **Suggestion:**
- **Why it looked plausible:**
- **What was actually wrong / what happened when tried:**
- **How I detected it (test, shape check, ablation, reading the docs):**
- **What I did instead:**

## Decisions I made (not simply accepted from the AI)

Record at least one important experimental or architectural decision. Template:

- **Decision:**
- **AI's recommendation (if any):**
- **Why I chose differently / what evidence drove it:**

## Habits followed when accepting generated code

Before running AI-written code: what do I expect it to do, what tensor-shape or
data assumptions does it make, how will I know if it is wrong, and what experiment
distinguishes a real improvement from noise? (From the starter notebook, §7.)
