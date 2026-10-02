#!/usr/bin/env python
"""Low-label experiments: supervised-only on k labels/class, or teacher-student
semi-supervised using the remaining training images as an unlabeled pool.

    python train_semisup.py --config configs/semisup_sup_k20.yaml --seed 0
    python train_semisup.py --config configs/semisup_ts_k20.yaml  --seed 0

Config keys (on top of the usual ones):
    labels_per_class : k  (split file splits/semisup_k<k>.json is created if missing)
    mode             : supervised | teacher_student
    mu, tau, lambda_u, burnin_epochs, weak_aug, strong_aug, pseudo_source (teacher|student)
Validation set is the same 480 images as every other experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from scripts.make_label_split import make_label_split
from src.data import build_semisup_loaders
from src.engine import train as train_supervised
from src.models import build_model
from src.reporting import finalize_run
from src.semisup import train_teacher_student
from src.utils import count_params, environment_info, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="splits/val_split.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    k, mode = int(cfg["labels_per_class"]), cfg.get("mode", "supervised")
    name = args.name or Path(args.config).stem
    run_dir = Path(args.runs) / name / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "train.log").open("w")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    label_split = Path(f"splits/semisup_k{k}.json")
    if not label_split.exists():  # deterministic (seed 0), so safe to create on demand
        make_label_split(Path(args.data), Path(args.split), k, 0, label_split)
        log(f"created {label_split}")

    set_seed(seed, cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (lab_loader, unl_loader, val_loader), classes = build_semisup_loaders(cfg, label_split, Path(args.data), Path(args.split))
    cfg["num_classes"] = len(classes)
    model = build_model(cfg, len(classes))
    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    save_json(cfg, run_dir / "config.json")
    save_json(environment_info(), run_dir / "env.json")
    log(f"run {name} seed {seed} | {mode} k={k} | labeled {len(lab_loader.dataset)} "
        f"unlabeled {len(unl_loader.dataset) if unl_loader else 0} val {len(val_loader.dataset)} | "
        f"{cfg['model']} pretrained={cfg.get('pretrained','none')} | params {n_params/1e6:.2f}M | {device}")
    log(f"config: {cfg}")

    if mode == "supervised":
        best = train_supervised(cfg, model, lab_loader, val_loader, device, run_dir, log=log)
    elif mode == "teacher_student":
        best = train_teacher_student(cfg, model, lab_loader, unl_loader, val_loader, device, run_dir, log=log)
    else:
        raise ValueError(mode)

    finalize_run(cfg, model, best, val_loader, classes, run_dir, name, seed, n_params, n_train, device, log,
                 extra={"labels_per_class": k, "mode": mode})


if __name__ == "__main__":
    main()
