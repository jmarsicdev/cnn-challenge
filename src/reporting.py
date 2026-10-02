"""End-of-run reporting shared by the training entry points."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .engine import evaluate, per_class_accuracy
from .utils import save_json


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


def throughput(model, loader, device):
    x = next(iter(loader))[0].to(device)
    model.eval()
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(3): model(x)
        sync(); t = time.time()
        for _ in range(10): model(x)
        sync()
    return 10 * x.size(0) / (time.time() - t)


def finalize_run(cfg, model, best, val_loader, classes, run_dir: Path, name, seed,
                 n_params, n_train, device, log, extra: dict | None = None):
    """Reload best weights, write per-class/confusion/metrics/best.pt. Returns metrics."""
    from sklearn.metrics import confusion_matrix
    model.load_state_dict(best["state"])
    val = evaluate(model, val_loader, device)
    pc = per_class_accuracy(val["preds"], val["targets"], len(classes))
    cm = confusion_matrix(val["targets"], val["preds"], labels=range(len(classes)))
    np.save(run_dir / "confusion.npy", cm)
    plot_confusion(cm, classes, run_dir / "confusion.png", f"{name} s{seed} val acc {val['acc']:.3f}")
    save_json(dict(zip(classes, pc)), run_dir / "per_class.json")
    ips = throughput(model, val_loader, device)

    metrics = {"name": name, "seed": seed, "model": cfg["model"], "pretrained": cfg.get("pretrained", "none"),
               "color": cfg["color"], "aug": cfg["aug"], "img_size": cfg["img_size"],
               "epochs": cfg["epochs"], "lr": cfg["lr"], "optimizer": cfg.get("optimizer", "adamw"),
               "mixup": cfg.get("mixup", 0.0), "cutmix": cfg.get("cutmix", 0.0),
               "label_smoothing": cfg.get("label_smoothing", 0.0), "weight_decay": cfg.get("weight_decay", 0.0),
               "ema": bool(cfg.get("ema_decay")), "freeze_backbone": cfg.get("freeze_backbone", False),
               "params_M": n_params / 1e6, "trainable_M": n_train / 1e6,
               "best_val_acc": best["acc"], "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
               "minutes": best["minutes"], "eval_img_per_sec": ips,
               "worst_classes": sorted(zip(pc, classes))[:3], **(extra or {})}
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": best["state"], "config": cfg, "classes": classes,
                "epoch": best["epoch"], "val_acc": best["acc"]}, run_dir / "best.pt")
    log(f"BEST val acc {best['acc']:.4f} @ epoch {best['epoch']} | {best['minutes']:.1f} min | "
        f"{ips:.0f} img/s | worst: {metrics['worst_classes']}")
    return metrics
