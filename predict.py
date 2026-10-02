#!/usr/bin/env python
"""Predict labels for a folder of unlabelled images (e.g. data/test2).

    python predict.py --checkpoint runs/<name>/s0/best.pt --input data/test2 --out predictions/test2.csv

Writes CSV: filename,label,label_index,confidence  (sorted by numeric suffix).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.data import IMG_EXT, build_transforms
from evaluate import load_checkpoint


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", default="data/test2")
    ap.add_argument("--out", default="predictions/test2.csv")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    tf = build_transforms(cfg["img_size"], cfg["color"], "none", train=False, resize_mode=cfg.get("resize_mode", "crop"))
    files = sorted((p for p in Path(args.input).iterdir() if p.suffix.lower() in IMG_EXT), key=natural_key)

    rows = []
    with torch.inference_mode():
        for i in range(0, len(files), 64):
            batch = files[i:i + 64]
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in batch]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p = model(x).float().softmax(1)
                if args.tta:
                    p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
            conf, idx = p.max(1)
            for f, c, k in zip(batch, conf.tolist(), idx.tolist()):
                rows.append({"filename": f.name, "label": classes[k], "label_index": k, "confidence": f"{c:.4f}"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "label_index", "confidence"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"wrote {len(rows)} predictions to {args.out}")
    print("label histogram:", dict(sorted(Counter(r['label'] for r in rows).items())))


if __name__ == "__main__":
    main()
