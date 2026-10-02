"""Training / evaluation loops."""
from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter

from .models import param_groups


def build_optimizer(cfg, model):
    groups = param_groups(model, cfg["lr"], cfg.get("weight_decay", 0.0),
                          cfg.get("backbone_lr_mult", 1.0))
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(name)


def build_scheduler(cfg, optimizer, steps_per_epoch):
    """Per-step linear warmup then cosine to ~0. Returns None for constant LR."""
    name = cfg.get("scheduler", "cosine")
    if name == "none":
        return None
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg.get("warmup_epochs", 1) * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate(model, loader, device, tta: bool = False):
    """Returns dict(loss, acc, preds, targets, probs)."""
    model.eval()
    losses, preds, targets, probs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
            if tta:
                logits = (logits.softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)).log()
        logits = logits.float()
        losses.append(F.cross_entropy(logits, y, reduction="sum").item())
        probs.append(logits.softmax(1).cpu())
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets, probs = torch.cat(preds), torch.cat(targets), torch.cat(probs)
    return {"loss": sum(losses) / len(targets), "acc": (preds == targets).float().mean().item(),
            "preds": preds.numpy(), "targets": targets.numpy(), "probs": probs.numpy()}


def per_class_accuracy(preds, targets, num_classes):
    out = []
    for c in range(num_classes):
        m = targets == c
        out.append(float((preds[m] == c).mean()) if m.any() else float("nan"))
    return out


def train(cfg: dict, model: nn.Module, train_loader, val_loader, device, run_dir: Path, log=print):
    model.to(device)
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt, len(train_loader))
    epochs = cfg["epochs"]
    num_classes = cfg["num_classes"]

    mix = None
    if cfg.get("mixup", 0.0) > 0 or cfg.get("cutmix", 0.0) > 0:
        mix = Mixup(mixup_alpha=cfg.get("mixup", 0.0), cutmix_alpha=cfg.get("cutmix", 0.0),
                    label_smoothing=cfg.get("label_smoothing", 0.0), num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    ema = ModelEmaV3(model, decay=cfg["ema_decay"]) if cfg.get("ema_decay") else None
    use_amp = device.type == "cuda" and cfg.get("amp", True)

    best = {"acc": -1.0, "epoch": 0, "state": None, "val_loss": None}
    hist_path = run_dir / "history.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "ema_val_acc", "sec"]
    with hist_path.open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    tb = SummaryWriter(log_dir=str(run_dir / "tb"))
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_correct, n = 0.0, 0, 0
        te = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_in = y
            if mix is not None:
                x, y_in = mix(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y_in)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            tot_loss += loss.item() * y.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        val = evaluate(model, val_loader, device)
        ema_val = evaluate(ema.module, val_loader, device) if ema is not None else None
        # model selection: EMA weights if enabled, else raw weights
        sel_acc, sel_loss = (ema_val["acc"], ema_val["loss"]) if ema is not None else (val["acc"], val["loss"])
        if sel_acc > best["acc"]:
            src = ema.module if ema is not None else model
            best = {"acc": sel_acc, "epoch": epoch, "val_loss": sel_loss,
                    "state": copy.deepcopy(src.state_dict())}

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss / n,
               "train_acc": tot_correct / n, "val_loss": val["loss"], "val_acc": val["acc"],
               "ema_val_acc": ema_val["acc"] if ema_val else "", "sec": time.time() - te}
        with hist_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
            tb.add_scalar(k, row[k], epoch)
        if ema_val:
            tb.add_scalar("ema_val_acc", ema_val["acc"], epoch)
        tb.add_scalar("gap/train_minus_val_acc", row["train_acc"] - val["acc"], epoch)
        tb.flush()
        log(f"ep {epoch:03d}/{epochs} lr {row['lr']:.2e} | train loss {row['train_loss']:.3f} "
            f"acc {row['train_acc']:.3f} | val loss {val['loss']:.3f} acc {val['acc']:.4f}"
            + (f" | ema {ema_val['acc']:.4f}" if ema_val else "") + f" | {row['sec']:.0f}s")

    best["minutes"] = (time.time() - t0) / 60
    tb.add_hparams({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in cfg.items()},
                   {"hp/best_val_acc": best["acc"], "hp/best_epoch": best["epoch"]}, run_name=".")
    tb.close()
    return best
