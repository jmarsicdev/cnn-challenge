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

Options against confirmation bias (cfg keys):
  dist_align: true     scale teacher probabilities by (uniform prior / running mean of teacher
                       predictions) before thresholding, so no class can hog the pseudo-labels
                       (distribution alignment, ReMixMatch / FixMatch-DA; classes here are balanced)
  unsup_loss: ce|mse   ce = hard pseudo-labels above tau (FixMatch); mse = soft consistency on
                       probabilities with no threshold (original Mean Teacher)
  lambda_rampup_epochs sigmoid ramp of lambda_u from 0 over this many epochs after burn-in
                       (Mean Teacher's schedule); 0 = constant

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
    dist_align = bool(cfg.get("dist_align", False))
    unsup_kind = cfg.get("unsup_loss", "ce")
    rampup = int(cfg.get("lambda_rampup_epochs", 0))
    num_classes = cfg["num_classes"]
    p_running = torch.full((num_classes,), 1.0 / num_classes, device=device)  # running mean of teacher probs
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

    eval_every = int(cfg.get("eval_every", 1))
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
        # Mean-Teacher style sigmoid ramp-up of the unsupervised weight after burn-in
        if pseudo_on and rampup > 0:
            t = min(1.0, (epoch - burnin - 1) / rampup)
            lam_now = lam * float(torch.exp(torch.tensor(-5.0 * (1 - t) ** 2)))
        else:
            lam_now = lam

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
                    if dist_align:
                        p_running.mul_(0.99).add_(0.01 * probs.mean(0))
                        probs = probs * ((1.0 / num_classes) / p_running.clamp_min(1e-6))
                        probs = probs / probs.sum(1, keepdim=True)
                    conf, pseudo = probs.max(1)
                    logits_s = model(xs).float()
                    if unsup_kind == "mse":  # soft consistency, no threshold (Mean Teacher)
                        mask = torch.ones_like(conf)
                        loss_unsup = F.mse_loss(logits_s.softmax(1), probs)
                    else:  # hard pseudo-labels above tau (FixMatch / Unbiased Teacher)
                        mask = (conf >= tau).float()
                        loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
                    n_mask += int(mask.sum().item())
                    n_pseudo_correct += int(((pseudo == y_hidden).float() * mask).sum().item())
                n_unl += xw.size(0)
                loss = loss_sup + lam_now * loss_unsup

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            teacher.update(model)
            s_sup += loss_sup.item(); s_unsup += loss_unsup.item()

        do_eval = (epoch % eval_every == 0) or epoch == epochs or epoch == burnin
        student_val = teacher_val = None
        if do_eval:
            student_val = evaluate(model, val_loader, device)
            teacher_val = evaluate(teacher.module, val_loader, device)
            if teacher_val["acc"] > best["acc"]:
                best = {"acc": teacher_val["acc"], "epoch": epoch, "val_loss": teacher_val["loss"],
                        "state": copy.deepcopy(teacher.module.state_dict())}

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "sup_loss": s_sup / steps_per_epoch,
               "unsup_loss": s_unsup / steps_per_epoch, "mask_rate": n_mask / max(1, n_unl),
               "pseudo_acc": n_pseudo_correct / max(1, n_mask),
               "student_val_acc": student_val["acc"] if student_val else "",
               "val_acc": teacher_val["acc"] if teacher_val else "",
               "val_loss": teacher_val["loss"] if teacher_val else "", "sec": time.time() - te}
        with hist_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        for k in fields[1:-1]:
            if row[k] != "":
                tb.add_scalar(k, row[k], epoch)
        tb.flush()
        msg = (f"ep {epoch:03d}/{epochs} lr {row['lr']:.2e} | sup {row['sup_loss']:.3f} unsup {row['unsup_loss']:.3f} "
               f"| mask {row['mask_rate']:.2f} pseudo-acc {row['pseudo_acc']:.3f}")
        if teacher_val:
            msg += f" | val student {student_val['acc']:.4f} teacher {teacher_val['acc']:.4f}"
        log(msg + f" | {row['sec']:.0f}s")

    best["minutes"] = (time.time() - t0) / 60
    tb.close()
    return best
