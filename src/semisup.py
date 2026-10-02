"""Teacher-student semi-supervised training (Mean Teacher / FixMatch / Unbiased-Teacher style).

Student  : the network trained by gradient descent.
Teacher  : an exponential moving average (EMA) of the student's weights. It is
           never trained directly; it is updated a little toward the student
           after every optimizer step (the same ModelEmaV3 used for weight
           averaging in the supervised recipe).

Each step:
  1. supervised cross-entropy on a labeled batch (label smoothing allowed);
  2. after `burnin_epochs`, the teacher (or the student, cfg.pseudo_source)
     predicts on a *weakly* augmented view of an unlabeled batch; predictions
     with confidence >= tau become pseudo-labels; the student is trained to
     reproduce them from a *strongly* augmented view of the same images;
  3. loss = sup + lambda_u * masked unsup; EMA update of the teacher.

At the end of burn-in the teacher is reset to a copy of the student (Unbiased
Teacher), so the first pseudo-labels come from a model that has already fitted
the labeled set. Model selection and the saved checkpoint use the teacher.

Diagnostics logged per epoch: mask rate (fraction of unlabeled images that
cleared tau) and pseudo-label accuracy among those (computed with the hidden
labels, which are used for nothing else).
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter

from .engine import build_optimizer, build_scheduler, evaluate


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train_teacher_student(cfg: dict, model: nn.Module, lab_loader, unl_loader, val_loader,
                          device, run_dir: Path, log=print):
    model.to(device)
    opt = build_optimizer(cfg, model)
    steps_per_epoch = len(unl_loader)
    sched = build_scheduler(cfg, opt, steps_per_epoch)
    epochs = cfg["epochs"]
    burnin = cfg.get("burnin_epochs", 0)
    tau, lam = cfg["tau"], cfg["lambda_u"]
    pseudo_source = cfg.get("pseudo_source", "teacher")
    use_amp = device.type == "cuda" and cfg.get("amp", True)

    teacher = ModelEmaV3(model, decay=cfg["ema_decay"])
    teacher.module.eval()  # pseudo-labels and validation always in eval mode (no drop-path)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    lab_iter = _cycle(lab_loader)

    best = {"acc": -1.0, "epoch": 0, "state": None, "val_loss": None}
    fields = ["epoch", "lr", "sup_loss", "unsup_loss", "mask_rate", "pseudo_acc",
              "student_val_acc", "val_acc", "val_loss", "sec"]
    hist_path = run_dir / "history.csv"
    with hist_path.open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        pseudo_on = epoch > burnin
        if epoch == burnin + 1 and burnin > 0:
            teacher.module.load_state_dict(model.state_dict())  # reset teacher to burnt-in student
            log(f"-- burn-in finished at epoch {burnin}; teacher reset to student; pseudo-labeling on")
        s_sup = s_unsup = 0.0
        n_mask = n_pseudo_correct = n_unl = 0
        te = time.time()

        for xw, xs, y_hidden in unl_loader:
            xl, yl = next(lab_iter)
            xl, yl = xl.to(device, non_blocking=True), yl.to(device, non_blocking=True)
            xw, xs = xw.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            y_hidden = y_hidden.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss_sup = criterion(model(xl).float(), yl)
                loss_unsup = torch.zeros((), device=device)
                if pseudo_on:
                    with torch.no_grad():
                        if pseudo_source == "teacher":
                            probs = teacher.module(xw).float().softmax(1)
                        else:  # FixMatch: the student itself labels the weak view
                            was_training = model.training
                            model.eval(); probs = model(xw).float().softmax(1); model.train(was_training)
                    conf, pseudo = probs.max(1)
                    mask = (conf >= tau).float()
                    logits_s = model(xs).float()
                    loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
                    n_mask += int(mask.sum().item())
                    n_pseudo_correct += int(((pseudo == y_hidden).float() * mask).sum().item())
                n_unl += xw.size(0)
                loss = loss_sup + lam * loss_unsup

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            teacher.update(model)
            s_sup += loss_sup.item(); s_unsup += loss_unsup.item()

        student_val = evaluate(model, val_loader, device)
        teacher_val = evaluate(teacher.module, val_loader, device)
        if teacher_val["acc"] > best["acc"]:
            best = {"acc": teacher_val["acc"], "epoch": epoch, "val_loss": teacher_val["loss"],
                    "state": copy.deepcopy(teacher.module.state_dict())}

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "sup_loss": s_sup / steps_per_epoch,
               "unsup_loss": s_unsup / steps_per_epoch, "mask_rate": n_mask / max(1, n_unl),
               "pseudo_acc": n_pseudo_correct / max(1, n_mask), "student_val_acc": student_val["acc"],
               "val_acc": teacher_val["acc"], "val_loss": teacher_val["loss"], "sec": time.time() - te}
        with hist_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        for k in fields[1:-1]:
            tb.add_scalar(k, row[k], epoch)
        tb.flush()
        log(f"ep {epoch:03d}/{epochs} lr {row['lr']:.2e} | sup {row['sup_loss']:.3f} unsup {row['unsup_loss']:.3f} "
            f"| mask {row['mask_rate']:.2f} pseudo-acc {row['pseudo_acc']:.3f} "
            f"| val student {student_val['acc']:.4f} teacher {teacher_val['acc']:.4f} | {row['sec']:.0f}s")

    best["minutes"] = (time.time() - t0) / 60
    tb.close()
    return best
