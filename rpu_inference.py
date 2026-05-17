"""Reusable RPU CIFAR-10 inference utilities.

Provides the model architecture, checkpoint loading, top-k prediction, and
device resolution used by the web demo. This module deliberately contains no
CLI / evaluation loop — that logic was specific to the original
inference_demo.py script and is not needed for the illustration interface.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# torchvision's CIFAR-10 loader triggers a NumPy 2 deprecation warning; mute it
# so the demo's console output stays clean.
_VISIBLE_DEPRECATION = getattr(getattr(np, "exceptions", np), "VisibleDeprecationWarning", None)
if _VISIBLE_DEPRECATION is not None:
    warnings.filterwarnings("ignore", category=_VISIBLE_DEPRECATION)


class CIFAR10VGG16(nn.Module):
    """VGG16 feature extractor + adaptive 1x1 pool + linear CIFAR-10 head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        backbone = models.vgg16(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not load as a checkpoint dictionary")
    return checkpoint


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key.removeprefix("module.") if isinstance(key, str) else key): value
        for key, value in state_dict.items()
    }


def _extract_state_dict(checkpoint: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_module_prefix(value)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _strip_module_prefix(checkpoint)
    raise KeyError(f"Could not find a model state dict in {path}")


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load a CIFAR10VGG16 checkpoint into ``device`` and switch to eval mode."""
    checkpoint = _torch_load(path)
    model = CIFAR10VGG16(num_classes=10)
    state_dict = _extract_state_dict(checkpoint, path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def resolve_device(requested: str = "auto") -> torch.device:
    name = requested.lower().strip()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@torch.inference_mode()
def predict_one(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    topk: int = 3,
) -> list[tuple[str, float]]:
    """Return the top-k (class_name, probability) pairs for one normalised image."""
    logits = model(image.unsqueeze(0).to(device))
    probs = logits.softmax(dim=1).squeeze(0)
    values, indices = torch.topk(probs, k=min(int(topk), len(CIFAR10_CLASSES)))
    return [
        (CIFAR10_CLASSES[int(index)], float(value))
        for value, index in zip(values.cpu(), indices.cpu(), strict=True)
    ]
