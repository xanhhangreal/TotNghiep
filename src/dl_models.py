"""Deep-learning models for multimodal stress detection.

Three 1-D architectures operating on hand-crafted feature vectors:
    1. StressCNN1D   – simple 1-D convolutional network
    2. StressUNet1D  – U-Net encoder with multi-scale classification head
    3. StressResNet1D – 1-D adaptation of ResNet-34

All models accept ``(batch, 1, n_features)`` tensors produced by
``FeatureExtractor`` and output class logits.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  StressCNN1D – Baseline 1-D CNN
# ═══════════════════════════════════════════════════════════════════════════════

class StressCNN1D(nn.Module):
    """Three-block 1-D CNN for feature-vector classification.

    Architecture:
        Conv1d(1→32, k=3) → BN → ReLU → Conv1d(32→64, k=3) → BN → ReLU
        → MaxPool(2) → Dropout → Conv1d(64→128, k=3) → BN → ReLU
        → AdaptiveAvgPool(1) → FC(128→64) → ReLU → Dropout → FC(64→C)
    """

    def __init__(self, n_features: int, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 1.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape: (B, 1, n_features)."""
        h = self.features(x)            # (B, 128, 1)
        h = h.view(h.size(0), -1)       # (B, 128)
        return self.classifier(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  StressUNet1D – U-Net Encoder with Multi-Scale Classification Head
# ═══════════════════════════════════════════════════════════════════════════════

class _DoubleConv1d(nn.Module):
    """Two consecutive Conv1d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StressUNet1D(nn.Module):
    """U-Net encoder (no decoder) with multi-scale feature fusion.

    Encoder levels capture features at different resolutions.  Skip
    connections from each level are globally-average-pooled and
    concatenated before the final classifier.

    Architecture:
        Encoder L1: DoubleConv(1→32)  → pool
        Encoder L2: DoubleConv(32→64) → pool
        Encoder L3: DoubleConv(64→128)→ pool
        Bottleneck: DoubleConv(128→256)
        Classification: GAP(L1‖L2‖L3‖BN) → FC → ReLU → Dropout → FC(C)
    """

    def __init__(self, n_features: int, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.enc1 = _DoubleConv1d(1, 32)
        self.enc2 = _DoubleConv1d(32, 64)
        self.enc3 = _DoubleConv1d(64, 128)
        self.bottleneck = _DoubleConv1d(128, 256)

        self.pool = nn.MaxPool1d(2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 32 + 64 + 128 + 256 = 480
        fused_ch = 32 + 64 + 128 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fused_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)               # (B, 32, L)
        e2 = self.enc2(self.pool(e1))    # (B, 64, L/2)
        e3 = self.enc3(self.pool(e2))    # (B, 128, L/4)
        bn = self.bottleneck(self.pool(e3))  # (B, 256, L/8)

        # Multi-scale fusion via GAP
        g1 = self.gap(e1).squeeze(-1)    # (B, 32)
        g2 = self.gap(e2).squeeze(-1)    # (B, 64)
        g3 = self.gap(e3).squeeze(-1)    # (B, 128)
        g4 = self.gap(bn).squeeze(-1)    # (B, 256)

        fused = torch.cat([g1, g2, g3, g4], dim=1)  # (B, 480)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  StressResNet1D – 1-D ResNet-34
# ═══════════════════════════════════════════════════════════════════════════════

class _BasicBlock1D(nn.Module):
    """ResNet BasicBlock adapted for 1-D convolution."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class StressResNet1D(nn.Module):
    """1-D ResNet-34 for feature-vector classification.

    Layer configuration ``[3, 4, 6, 3]`` with channels ``[64, 128, 256, 512]``
    mirrors the original ResNet-34 architecture.
    """

    def __init__(self, n_features: int, n_classes: int = 2,
                 dropout: float = 0.3,
                 layers: tuple = (3, 4, 6, 3)):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)

        # Weight initialization (Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_ch: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [_BasicBlock1D(self.in_channels, out_ch, stride, downsample)]
        self.in_channels = out_ch
        for _ in range(1, blocks):
            layers.append(_BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model registry & factory
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, type] = {
    "cnn1d": StressCNN1D,
    "unet1d": StressUNet1D,
    "resnet1d": StressResNet1D,
}


def build_dl_model(arch: str, n_features: int, n_classes: int = 2,
                   dropout: float = 0.3, **kwargs) -> nn.Module:
    """Instantiate a model by name from :data:`MODEL_REGISTRY`."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. "
                         f"Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](n_features, n_classes, dropout, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: save / load
# ═══════════════════════════════════════════════════════════════════════════════

def save_dl_model(
    model: nn.Module,
    path: str,
    meta: Optional[Dict] = None,
    scaler=None,
):
    """Save model weights + metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "arch": type(model).__name__,
        "n_features": model.n_features,
        "n_classes": model.n_classes,
        "state_dict": model.state_dict(),
    }
    if meta:
        state["meta"] = meta
    if scaler is not None:
        state["scaler"] = scaler
    torch.save(state, path)
    logger.info("DL model saved → %s", path)


def load_dl_model(path: str, device: str = "cpu", return_state: bool = False):
    """Load a saved DL model.

    Set ``return_state=True`` to also return the raw checkpoint dict.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # reverse lookup in registry
    cls_name = ckpt["arch"]
    for key, cls in MODEL_REGISTRY.items():
        if cls.__name__ == cls_name:
            model = cls(ckpt["n_features"], ckpt["n_classes"])
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device)
            if return_state:
                return model, ckpt
            return model
    raise ValueError(f"Unknown arch class: {cls_name}")
