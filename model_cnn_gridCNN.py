"""
model_cnn_gridCNN.py

Grid-search version of model_cnn.py. The three backbones used by the four
ICLR CIFAR-10 configs are kept with IDENTICAL layer definitions and parameter
initialisation order (so a given seed gives the same initial weights as the
original code):

  * resnet18 family (_NarrowResNet)  -> cifar10_ICLR_resnet10 / resnet18narrow
  * SimpleCNN                        -> cifar10_ICLR_plaincnn_bottleneck50
  * MyrtleCNN                        -> cifar10_ICLR_myrtle5

Removed (not needed, and some of it cost time on every forward pass):
  * the forward hook that cloned the pre-classifier features on EVERY forward
  * _TabularStem / _CNNWithStem wrapper (CIFAR-10 is image shaped)
  * TorchModelAdapter + all jax imports
  * load_pretrained_cnn, MultiStepLR helper, MSE loss, the per-step training
    loop (moved, optimised, into training_runner_gridCNN.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# ---------------------------------------------------------------------------
# ResNet-18 family (unchanged math)
# ---------------------------------------------------------------------------

class _NarrowResNet(ResNet):
    def __init__(self, blocks_per_stage, num_classes: int, width_mult: float = 1.0):
        self.width_mult = float(width_mult)
        super().__init__(BasicBlock, list(blocks_per_stage), num_classes=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        scaled_planes = max(1, int(round(planes * self.width_mult)))
        return super()._make_layer(block, scaled_planes, blocks, stride, dilate)


def build_resnet(
    num_classes: int,
    input_ch: int = 3,
    blocks_per_stage=(2, 2, 2, 2),
    num_stages: int = 4,
    width_mult: float = 1.0,
) -> nn.Module:
    if not (1 <= num_stages <= 4):
        raise ValueError(f"num_stages must be in [1, 4], got {num_stages}")
    if width_mult <= 0:
        raise ValueError(f"width_mult must be > 0, got {width_mult}")
    if len(blocks_per_stage) != 4:
        raise ValueError("blocks_per_stage must have length 4")

    effective_blocks = list(blocks_per_stage[:num_stages]) + [1] * (4 - num_stages)
    resnet = _NarrowResNet(effective_blocks, num_classes=num_classes, width_mult=width_mult)
    resnet.conv1 = nn.Conv2d(input_ch, resnet.conv1.weight.shape[0], 3, 1, 1, bias=False)
    resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    base_widths = [64, 128, 256, 512]
    for stage_idx in range(num_stages, 4):
        setattr(resnet, f"layer{stage_idx + 1}", nn.Identity())
    out_channels = max(1, int(round(base_widths[num_stages - 1] * width_mult)))
    resnet.fc = nn.Linear(out_channels, num_classes)
    return resnet


# ---------------------------------------------------------------------------
# SimpleCNN (unchanged math)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        channels: list[int],
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        pool_every_block: bool = True,
        dropout: float = 0.0,
        fc_hidden_dim: int | None = None,
    ):
        super().__init__()
        if not channels:
            raise ValueError("SimpleCNN requires at least one entry in `channels`.")
        layers: list[nn.Module] = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size, padding=kernel_size // 2, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            if pool_every_block:
                layers.append(nn.MaxPool2d(2))
            prev_c = c
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if fc_hidden_dim:
            self.pre_fc = nn.Sequential(nn.Linear(prev_c, fc_hidden_dim), nn.ReLU(inplace=True))
            classifier_in = fc_hidden_dim
        else:
            self.pre_fc = nn.Identity()
            classifier_in = prev_c
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(classifier_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.pre_fc(x)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# MyrtleCNN (unchanged math)
# ---------------------------------------------------------------------------

class MyrtleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        channels: list[int],
        blocks_per_stage: list[int] | None = None,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        pool_last_stage: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not channels:
            raise ValueError("MyrtleCNN requires at least one entry in `channels`.")
        if blocks_per_stage is None:
            blocks_per_stage = [1] * len(channels)
        if len(blocks_per_stage) != len(channels):
            raise ValueError("blocks_per_stage must have the same length as channels.")
        layers: list[nn.Module] = []
        prev_c = in_channels
        n_stages = len(channels)
        for stage_idx, (c, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            for _ in range(n_blocks):
                layers.append(nn.Conv2d(prev_c, c, kernel_size, padding=kernel_size // 2, bias=not use_batchnorm))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(c))
                layers.append(nn.ReLU(inplace=True))
                prev_c = c
            if stage_idx != n_stages - 1 or pool_last_stage:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(prev_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Config -> model (same yaml keys / defaults as training_runner.py)
# ---------------------------------------------------------------------------

def build_model_from_config(model_cfg: dict, num_classes: int, input_ch: int = 3) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "cnn")).strip().lower()
    if architecture not in {"resnet18_cnn", "cnn"}:
        raise ValueError(
            f"model.architecture '{architecture}' not supported by the gridCNN runner "
            "(only 'resnet18_cnn' and 'cnn')."
        )
    cnn_cfg = model_cfg.get("cnn", {}) or {}
    default_backbone = "resnet18" if architecture == "resnet18_cnn" else "simple_cnn"
    backbone = str(cnn_cfg.get("backbone", default_backbone)).strip().lower()

    if backbone == "resnet18":
        r = cnn_cfg.get("resnet", {}) or {}
        return build_resnet(
            num_classes=num_classes,
            input_ch=input_ch,
            blocks_per_stage=tuple(int(b) for b in r.get("blocks_per_stage", [2, 2, 2, 2])),
            num_stages=int(r.get("num_stages", 4)),
            width_mult=float(r.get("width_mult", 1.0)),
        )

    if backbone == "simple_cnn":
        fc_hidden_raw = cnn_cfg.get("fc_hidden_dim")
        return SimpleCNN(
            num_classes=num_classes,
            in_channels=input_ch,
            channels=[int(c) for c in cnn_cfg.get("channels", [16, 32])],
            kernel_size=int(cnn_cfg.get("kernel_size", 3)),
            use_batchnorm=_as_bool(cnn_cfg.get("use_batchnorm", True)),
            pool_every_block=_as_bool(cnn_cfg.get("pool_every_block", True)),
            dropout=float(cnn_cfg.get("dropout", 0.0)),
            fc_hidden_dim=int(fc_hidden_raw) if fc_hidden_raw else None,
        )

    if backbone == "myrtle":
        m = cnn_cfg.get("myrtle", {}) or {}
        channels_raw = m.get("channels")
        if channels_raw:
            channels = [int(c) for c in channels_raw]
        else:
            base_width = int(m.get("base_width", 64))
            num_stages = int(m.get("num_stages", 4))
            channels = [int(round(base_width * (2 ** i))) for i in range(num_stages)]
        bps_raw = m.get("blocks_per_stage")
        return MyrtleCNN(
            num_classes=num_classes,
            in_channels=input_ch,
            channels=channels,
            blocks_per_stage=[int(b) for b in bps_raw] if bps_raw else None,
            kernel_size=int(m.get("kernel_size", 3)),
            use_batchnorm=_as_bool(m.get("use_batchnorm", True)),
            pool_last_stage=_as_bool(m.get("pool_last_stage", True)),
            dropout=float(m.get("dropout", 0.0)),
        )

    raise ValueError(f"Unknown model.cnn.backbone '{backbone}'. Expected resnet18, simple_cnn or myrtle.")
