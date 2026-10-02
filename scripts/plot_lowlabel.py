"""Learning-curve figure: validation accuracy vs labeled images per class,
supervised-only vs EMA teacher-student (unlabeled pool = the remaining training images).

    python scripts/plot_lowlabel.py --out docs/final/lowlabel_curve.png
Reads runs/semisup_sup_k*/s*/metrics.json and runs/semisup_ts_k*/s*/metrics.json,
plus the fully supervised reference (convnext_tiny_final_a, 120 labels/class).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {  # fixed categorical order (validated palette slots 1 and 2)
    "supervised": ("Supervised only", "#2a78d6"),
    "teacher_student": ("EMA teacher-student (+ unlabeled pool)", "#eb6834"),
}


def collect(runs: Path):
    pts = defaultdict(lambda: defaultdict(list))  # mode -> k -> [acc]
    for mf in runs.glob("semisup_*/s*/metrics.json"):
        m = json.loads(mf.read_text())
        if m.get("mode") in SERIES and "fixmatch" not in m["name"]:
            pts[m["mode"]][int(m["labels_per_class"])].append(m["best_val_acc"] * 100)
    ref = [json.loads(f.read_text())["best_val_acc"] * 100 for f in runs.glob("convnext_tiny_final_a/s*/metrics.json")]
    return pts, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/final/lowlabel_curve.png")
    args = ap.parse_args()
    pts, ref = collect(Path(args.runs))

    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for mode, (label, color) in SERIES.items():
        ks = sorted(pts[mode])
        if not ks:
            continue
        mean = [st.mean(pts[mode][k]) for k in ks]
        sd = [st.stdev(pts[mode][k]) if len(pts[mode][k]) > 1 else 0 for k in ks]
        ax.errorbar(ks, mean, yerr=sd, color=color, lw=2, marker="o", ms=5, capsize=3, label=label, zorder=3)
        for k, m in zip(ks, mean):  # selective direct labels: endpoints only
            if k in (ks[0], ks[-1]):
                ax.annotate(f"{m:.1f}", (k, m), textcoords="offset points", xytext=(0, 7 if mode == "teacher_student" else -13),
                            ha="center", fontsize=7.5, color="#52514e")
    if ref:
        r = st.mean(ref)
        ax.axhline(r, color="#52514e", lw=1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"all 120 labels/class: {r:.1f}", (max(max(pts[m]) for m in pts if pts[m]), r),
                    textcoords="offset points", xytext=(0, -11), ha="right", fontsize=7.5, color="#52514e")
    ax.set_xlabel("labeled training images per class", fontsize=9, color="#0b0b0b")
    ax.set_ylabel("validation accuracy (%)", fontsize=9, color="#0b0b0b")
    ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 120]); ax.set_xticklabels(["10", "20", "40", "120"])
    ax.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("ConvNeXt-T: value of unlabeled images when labels are scarce", fontsize=9.5, loc="left", color="#0b0b0b")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    # companion table (markdown) so the numbers are not only in a picture
    lines = ["| labels/class | " + " | ".join(SERIES[m][0] for m in SERIES) + " |", "|---|---|---|"]
    for k in sorted({k for m in pts for k in pts[m]}):
        cells = []
        for m in SERIES:
            v = pts[m].get(k, [])
            cells.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f} (n={len(v)})" if len(v) > 1 else (f"{v[0]:.2f}" if v else "-"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if ref:
        lines.append(f"| 120 (all) | {st.mean(ref):.2f} ± {st.stdev(ref):.2f} (n={len(ref)}) | - |")
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("saved", args.out)


if __name__ == "__main__":
    main()
