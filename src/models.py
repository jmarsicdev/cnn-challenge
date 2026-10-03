from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TNet(nn.Module):
    """The deliberately weak starter-notebook CNN (1x64x64 gray input)."""

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
    if name == "scenenet_s":
        assert pretrained == "none", "scenenet_s has no pretrained weights"
        return SceneNetS(num_classes, in_chans, drop=cfg.get("dropout", 0.0))

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


# ---------------------------------------------------------------------------
# Small custom student for distillation (designed for 224x224 grayscale scenes)
# ---------------------------------------------------------------------------
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class DSBlock(nn.Module):
    """Depthwise-separable residual block (MobileNet-style), optional stride-2 downsample."""

    def __init__(self, cin, cout, stride=1, expand=4):
        super().__init__()
        mid = cin * expand
        self.body = nn.Sequential(
            ConvBNAct(cin, mid, 1),                      # expand
            ConvBNAct(mid, mid, 3, stride, groups=mid),  # depthwise spatial
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout),  # project (linear)
        )
        self.skip = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.skip else y


class SceneNetS(nn.Module):
    """~1.6M-parameter CNN: stem /4, four stages (/8, /16, /32, /32), global pool, linear.

    Design notes: depthwise-separable inverted-residual blocks keep FLOPs low at 224 px;
    SiLU + BN; no dropout (regularized by distillation instead); widths chosen so eval
    throughput is several times ResNet-18's while keeping a 7x7 final feature map.
    """

    def __init__(self, num_classes=16, in_chans=3, widths=(24, 48, 96, 192), depths=(2, 3, 4, 2), drop=0.0):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_chans, 16, 3, 2), ConvBNAct(16, widths[0], 3, 2))  # /4
        stages, cin = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = []
            for j in range(d):
                stride = 2 if (j == 0 and i > 0) else 1  # stage 0 stays at /4; stages 1-3 downsample -> /8, /16, /32
                blocks.append(DSBlock(cin, w, stride))
                cin = w
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head_conv = ConvBNAct(widths[3], 512, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stages(self.stem(x))
        x = self.pool(self.head_conv(x)).flatten(1)
        return self.fc(self.drop(x))

    def get_classifier(self):
        return self.fc


def load_teacher(ckpt_path: str, device) -> tuple[nn.Module, dict]:
    """Load a trained checkpoint (as written by train.py) as a frozen eval-mode teacher."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tcfg = ck["config"]
    teacher = build_model({**tcfg, "pretrained": "none"}, len(ck["classes"]))
    teacher.load_state_dict(ck["model"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, tcfg
