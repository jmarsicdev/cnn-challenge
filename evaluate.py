#!/usr/bin/env python
"""Evaluate a checkpoint on a labelled split.

    python evaluate.py --checkpoint runs/<name>/s0/best.pt --split test
    python evaluate.py --checkpoint ... --split val --tta
    python evaluate.py --checkpoint ... --split test --desaturate   # colour-shortcut probe

--desaturate forces grayscale input regardless of how the model was trained
(only Flower images actually change, the other 15 classes are already grey).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.data import ListDataset, build_transforms, folder_items, list_classes, load_split
from src.engine import evaluate, per_class_accuracy
from src.models import build_model


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, classes = ck["config"], ck["classes"]
    model = build_model({**cfg, "pretrained": "none"}, len(classes))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split-file", default="splits/val_split.json")
    ap.add_argument("--tta", action="store_true", help="average with horizontal flip")
    ap.add_argument("--desaturate", action="store_true", help="force grayscale input")
    ap.add_argument("--out", default=None, help="json file to write metrics to")
    ap.add_argument("--confusion-png", default=None, help="save a confusion-matrix figure here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    assert classes == list_classes(Path(args.data) / "train"), "class list mismatch"

    color = cfg["color"]
    if args.desaturate and color == "rgb":
        color = "gray"
    tf = build_transforms(cfg["img_size"], color, "none", train=False, resize_mode=cfg.get("resize_mode", "crop"))
    if args.split == "test":
        items, root = folder_items(Path(args.data) / "test", classes), Path(args.data) / "test"
    else:
        items, root = load_split(args.split_file, classes)[1], Path(args.data) / "train"
    loader = DataLoader(ListDataset(root, items, tf), batch_size=128, num_workers=8)

    res = evaluate(model, loader, device, tta=args.tta)
    pc = per_class_accuracy(res["preds"], res["targets"], len(classes))
    out = {"checkpoint": args.checkpoint, "split": args.split, "n": len(items), "tta": args.tta,
           "desaturate": args.desaturate, "acc": res["acc"], "loss": res["loss"],
           "per_class": dict(zip(classes, pc)), "train_val_acc": float(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["val_acc"])}
    print(f"{args.split} accuracy: {res['acc']:.4f}  (loss {res['loss']:.4f}, n={len(items)}"
          f"{', tta' if args.tta else ''}{', desaturated' if args.desaturate else ''})")
    for c, a in sorted(zip(classes, pc), key=lambda t: t[1]):
        print(f"  {c:<14} {a:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    if args.confusion_png:
        from sklearn.metrics import confusion_matrix
        from train import plot_confusion
        cm = confusion_matrix(res["targets"], res["preds"], labels=range(len(classes)))
        Path(args.confusion_png).parent.mkdir(parents=True, exist_ok=True)
        plot_confusion(cm, classes, args.confusion_png, f"{args.split} acc {res['acc']:.3f} (n={len(items)})")
        np.save(Path(args.confusion_png).with_suffix(".npy"), cm)


if __name__ == "__main__":
    main()
