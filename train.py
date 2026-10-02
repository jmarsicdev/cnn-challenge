#!/usr/bin/env python
"""Train one experiment.

    python train.py --config configs/resnet18_imagenet.yaml [--seed 0] [--set lr=1e-3 epochs=30]

Writes runs/<name>/s<seed>/{config.json, env.json, history.csv, metrics.json, best.pt,
per_class.json, confusion.npy, confusion.png}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_loaders
from src.engine import evaluate, per_class_accuracy, train
from src.models import build_model
from src.utils import count_params, environment_info, load_config, save_json, set_seed


def plot_confusion(cm, classes, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="splits/val_split.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None, help="run name (default: config stem)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    name = args.name or Path(args.config).stem
    run_dir = Path(args.runs) / name / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "train.log").open("w")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    set_seed(seed, cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (train_loader, val_loader, _), classes = build_loaders(cfg, Path(args.data), Path(args.split))
    cfg["num_classes"] = len(classes)
    model = build_model(cfg, len(classes))
    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    save_json(cfg, run_dir / "config.json")
    save_json(environment_info(), run_dir / "env.json")
    log(f"run {name} seed {seed} | {cfg['model']} pretrained={cfg.get('pretrained','none')} "
        f"| params {n_params/1e6:.2f}M (trainable {n_train/1e6:.2f}M) | "
        f"train {len(train_loader.dataset)} val {len(val_loader.dataset)} | {device}")
    log(f"config: {cfg}")

    best = train(cfg, model, train_loader, val_loader, device, run_dir, log=log)

    # reload best weights, detailed validation metrics
    model.load_state_dict(best["state"])
    val = evaluate(model, val_loader, device)
    pc = per_class_accuracy(val["preds"], val["targets"], len(classes))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(val["targets"], val["preds"], labels=range(len(classes)))
    np.save(run_dir / "confusion.npy", cm)
    plot_confusion(cm, classes, run_dir / "confusion.png", f"{name} s{seed} val acc {val['acc']:.3f}")
    save_json(dict(zip(classes, pc)), run_dir / "per_class.json")

    # throughput (images/sec, eval mode, batch of val loader) for the efficiency story
    x = next(iter(val_loader))[0].to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(3): model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t = time.time()
        for _ in range(10): model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
    ips = 10 * x.size(0) / (time.time() - t)

    metrics = {"name": name, "seed": seed, "model": cfg["model"], "pretrained": cfg.get("pretrained", "none"),
               "color": cfg["color"], "aug": cfg["aug"], "img_size": cfg["img_size"],
               "epochs": cfg["epochs"], "lr": cfg["lr"], "optimizer": cfg.get("optimizer", "adamw"),
               "mixup": cfg.get("mixup", 0.0), "cutmix": cfg.get("cutmix", 0.0),
               "label_smoothing": cfg.get("label_smoothing", 0.0), "weight_decay": cfg.get("weight_decay", 0.0),
               "ema": bool(cfg.get("ema_decay")), "freeze_backbone": cfg.get("freeze_backbone", False),
               "params_M": n_params / 1e6, "trainable_M": n_train / 1e6,
               "best_val_acc": best["acc"], "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
               "minutes": best["minutes"], "eval_img_per_sec": ips,
               "worst_classes": sorted(zip(pc, classes))[:3]}
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": best["state"], "config": cfg, "classes": classes,
                "epoch": best["epoch"], "val_acc": best["acc"]}, run_dir / "best.pt")
    log(f"BEST val acc {best['acc']:.4f} @ epoch {best['epoch']} | {best['minutes']:.1f} min | "
        f"{ips:.0f} img/s | worst: {metrics['worst_classes']}")


if __name__ == "__main__":
    main()
