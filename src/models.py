from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TNet(nn.Module):
    """The deliberately weak starter-notebook CNN (1x64x64 grey input)."""

    def __init__(self, num_classes=16, in_chans=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 15 * 15, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_classifier(self):
        return self.classifier[-1]


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    name = cfg["model"]
    pretrained = cfg.get("pretrained", "none")
    in_chans = 1 if cfg["color"] == "gray1" else 3

    if name == "tnet":
        assert cfg["img_size"] == 64, "TNet's flatten size assumes 64x64 input"
        return TNet(num_classes, in_chans)

    kwargs = dict(num_classes=num_classes, in_chans=in_chans,
                  drop_rate=cfg.get("dropout", 0.0))
    if cfg.get("drop_path", 0.0):
        kwargs["drop_path_rate"] = cfg["drop_path"]

    if pretrained == "imagenet":
        model = timm.create_model(name, pretrained=True, **kwargs)
    elif pretrained == "none":
        model = timm.create_model(name, pretrained=False, **kwargs)
    elif pretrained == "places365":
        model = timm.create_model(name, pretrained=False, **kwargs)
        load_places365(model, name)
    else:
        raise ValueError(f"unknown pretrained source {pretrained}")

    if cfg.get("freeze_backbone", False):
        head = model.get_classifier()
        head_params = {id(p) for p in head.parameters()}
        for p in model.parameters():
            p.requires_grad = id(p) in head_params
    return model


def load_places365(model: nn.Module, name: str) -> None:
    """Load MIT CSAIL Places365 weights (torchvision-style ResNet keys) into a timm ResNet."""
    urls = {"resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
            "resnet18": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"}
    if name not in urls:
        raise ValueError(f"no Places365 weights known for {name}")
    sd = torch.hub.load_state_dict_from_url(urls[name], map_location="cpu", check_hash=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}  # drop 365-way head
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("fc.")]
    assert not unexpected and not missing, (missing, unexpected)


def param_groups(model: nn.Module, lr: float, weight_decay: float, backbone_lr_mult: float = 1.0):
    """Head vs backbone LR split; no weight decay on biases/norm params."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {("head", True): [], ("head", False): [], ("bb", True): [], ("bb", False): []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        decay = p.ndim > 1  # conv/linear weights decay; biases & norm params don't
        groups[("head" if is_head else "bb", decay)].append(p)
    out = []
    for (part, decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr if part == "head" else lr * backbone_lr_mult,
                        "weight_decay": weight_decay if decay else 0.0,
                        "name": f"{part}_{'decay' if decay else 'nodecay'}"})
    return out
