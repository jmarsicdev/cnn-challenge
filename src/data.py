"""Dataset + transform presets.

Colour modes
    rgb   : images as stored (15 scene classes are grey, Flower is colour)
    gray  : everything converted to grayscale, replicated to 3 channels
    gray1 : single-channel grayscale (starter-notebook style)

Augmentation presets (train only; eval is always resize/crop + normalise)
    none   : same as eval
    light  : RandomResizedCrop + horizontal flip
    medium : light + TrivialAugmentWide
    strong : medium + RandomErasing
Mixup / CutMix are handled in the engine, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_classes(train_root: Path) -> list[str]:
    return sorted(p.name for p in train_root.iterdir() if p.is_dir())


def build_transforms(img_size: int, color: str, aug: str, train: bool,
                     resize_mode: str = "crop", rrc_scale=(0.35, 1.0)) -> T.Compose:
    ops: list = []
    if color == "gray":
        ops.append(T.Grayscale(num_output_channels=3))
    elif color == "gray1":
        ops.append(T.Grayscale(num_output_channels=1))
    elif color != "rgb":
        raise ValueError(f"unknown color mode {color}")

    if train and aug != "none":
        ops += [T.RandomResizedCrop(img_size, scale=tuple(rrc_scale)),
                T.RandomHorizontalFlip()]
        if aug in ("medium", "strong"):
            ops.append(T.TrivialAugmentWide())
        elif aug != "light":
            raise ValueError(f"unknown aug preset {aug}")
    else:
        if resize_mode == "squash":
            ops.append(T.Resize((img_size, img_size)))
        else:  # resize shorter side then centre crop (standard ImageNet eval)
            ops += [T.Resize(int(round(img_size / 0.875))), T.CenterCrop(img_size)]

    ops.append(T.ToTensor())
    if color == "gray1":
        ops.append(T.Normalize([0.5], [0.5]))
    else:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if train and aug == "strong":
        ops.append(T.RandomErasing(p=0.25))
    return T.Compose(ops)


class ListDataset(Dataset):
    """Images given as (relative_path, label) pairs under `root`."""

    def __init__(self, root: Path, items: list[tuple[str, int]], transform):
        self.root = Path(root)
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        with Image.open(self.root / rel) as im:
            im = im.convert("RGB")
        return self.transform(im), y


def folder_items(split_root: Path, classes: list[str]) -> list[tuple[str, int]]:
    c2i = {c: i for i, c in enumerate(classes)}
    out = []
    for c in classes:
        for f in sorted((split_root / c).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                out.append((f"{c}/{f.name}", c2i[c]))
    return out


def load_split(split_file: Path, classes: list[str]):
    """splits/val_split.json: {"train": ["Bedroom/image_0001.jpg", ...], "val": [...]}"""
    d = json.loads(Path(split_file).read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    to_items = lambda rels: [(r, c2i[r.split("/")[0]]) for r in rels]
    return to_items(d["train"]), to_items(d["val"])


def build_loaders(cfg: dict, data_root: Path = Path("data"), split_file: Path = Path("splits/val_split.json")):
    classes = list_classes(data_root / "train")
    tr_items, va_items = load_split(split_file, classes)
    te_items = folder_items(data_root / "test", classes)

    common = dict(img_size=cfg["img_size"], color=cfg["color"],
                  resize_mode=cfg.get("resize_mode", "crop"))
    tf_train = build_transforms(aug=cfg["aug"], train=True,
                                rrc_scale=cfg.get("rrc_scale", (0.35, 1.0)), **common)
    tf_eval = build_transforms(aug="none", train=False, **common)

    ds_train = ListDataset(data_root / "train", tr_items, tf_train)
    ds_val = ListDataset(data_root / "train", va_items, tf_eval)
    ds_test = ListDataset(data_root / "test", te_items, tf_eval)

    nw = cfg.get("num_workers", 8)
    pin = torch.cuda.is_available()
    mk = lambda ds, shuffle, bs: DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                                            pin_memory=pin, persistent_workers=nw > 0,
                                            drop_last=False)
    bs = cfg["batch_size"]
    return (mk(ds_train, True, bs), mk(ds_val, False, bs * 2), mk(ds_test, False, bs * 2)), classes
