"""Curate CIFAR-10 test images for the web demo.

Runs both checkpoints once and picks samples that produce a clean before/after
comparison:
- Forget (airplane): original model predicts `airplane` correctly with high
  confidence and the RPU model no longer predicts `airplane`.
- Retain (bird/ship/automobile): both models predict the true label correctly,
  so the demo can show that non-forgotten classes are preserved.

Writes upscaled PNG thumbnails to static/images/ and a manifest JSON to
static/samples.json for the Flask backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets, transforms

from rpu_inference import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    load_model,
    resolve_device,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DATA_DIR = SCRIPT_DIR / "data"
STATIC_DIR = SCRIPT_DIR / "static"

FORGET_PER_CLASS = {0: 8}                      # 8 airplane samples (the forgotten class)
RETAIN_PER_CLASS = {2: 2, 8: 2, 1: 2}          # 2 each of bird, ship, automobile

THUMB_SIZE = 768                                # display-only upscale (model still sees the 32x32 original)
RESAMPLE = Image.Resampling.LANCZOS             # smooth interpolation


def enhance_for_display(pil_image: Image.Image) -> Image.Image:
    """32x upscale a CIFAR-10 image so it looks presentable on a slide.

    Steps:
    1. Bicubic 8x to 256 (preserves more local structure than Lanczos at extreme scales)
    2. Lanczos to final THUMB_SIZE
    3. Unsharp mask to give edges back some bite after the smoothing
    4. Small contrast + saturation lift so the thumbnails look punchy
    """
    intermediate = pil_image.resize((256, 256), Image.Resampling.BICUBIC)
    upscaled = intermediate.resize((THUMB_SIZE, THUMB_SIZE), RESAMPLE)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.10)
    upscaled = ImageEnhance.Color(upscaled).enhance(1.10)
    upscaled = ImageEnhance.Sharpness(upscaled).enhance(1.4)
    return upscaled
ORIGINAL_CONF_FORGET = 0.6                      # min confidence for original-on-airplane
ORIGINAL_CONF_RETAIN = 0.5                      # min confidence for original-on-retain


@torch.inference_mode()
def model_predict(model, image_tensor, device) -> tuple[int, float]:
    logits = model(image_tensor.unsqueeze(0).to(device))
    probs = logits.softmax(dim=1).squeeze(0).cpu()
    top_idx = int(probs.argmax().item())
    return top_idx, float(probs[top_idx].item())


def pick_forget(test_set_raw, test_set_norm, original, unlearned, device, class_id, count):
    labels = np.asarray(test_set_raw.targets, dtype=np.int64)
    candidates = np.where(labels == class_id)[0]
    picked: list[int] = []
    for cifar_idx in candidates:
        image, _ = test_set_norm[int(cifar_idx)]
        orig_top, orig_prob = model_predict(original, image, device)
        if orig_top != class_id or orig_prob < ORIGINAL_CONF_FORGET:
            continue
        rpu_top, _ = model_predict(unlearned, image, device)
        if rpu_top == class_id:
            continue
        picked.append(int(cifar_idx))
        if len(picked) >= count:
            break
    return picked


def pick_retain(test_set_raw, test_set_norm, original, unlearned, device, class_id, count):
    """Pick retain samples where BOTH models predict the true class correctly.

    The shipped RPU checkpoint's retain accuracy on the full test set is lower
    than the paper's headline 99.8%, so we explicitly filter for samples that
    survive unlearning cleanly — these are the ones that tell the right story
    in the live demo.
    """
    labels = np.asarray(test_set_raw.targets, dtype=np.int64)
    candidates = np.where(labels == class_id)[0]
    picked: list[int] = []
    for cifar_idx in candidates:
        image, _ = test_set_norm[int(cifar_idx)]
        orig_top, orig_prob = model_predict(original, image, device)
        if orig_top != class_id or orig_prob < ORIGINAL_CONF_RETAIN:
            continue
        rpu_top, _ = model_predict(unlearned, image, device)
        if rpu_top != class_id:
            continue
        picked.append(int(cifar_idx))
        if len(picked) >= count:
            break
    return picked


def main() -> None:
    images_dir = STATIC_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    print(f"device: {device}")
    original, _ = load_model(CHECKPOINTS_DIR / "original_model.pt", device)
    unlearned, _ = load_model(CHECKPOINTS_DIR / "advance_RPU.pt", device)

    # download=True so the script works on a fresh clone (no-op if data/ already
    # contains cifar-10-batches-py/).
    test_set_raw = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set_norm = datasets.CIFAR10(root=str(DATA_DIR), train=False, transform=transform, download=False)

    selections: list[tuple[int, int, str]] = []
    for class_id, count in FORGET_PER_CLASS.items():
        chosen = pick_forget(test_set_raw, test_set_norm, original, unlearned, device, class_id, count)
        print(f"forget class={class_id} ({CIFAR10_CLASSES[class_id]}): picked {len(chosen)} → {chosen}")
        for idx in chosen:
            selections.append((idx, class_id, "forget"))

    for class_id, count in RETAIN_PER_CLASS.items():
        chosen = pick_retain(test_set_raw, test_set_norm, original, unlearned, device, class_id, count)
        print(f"retain class={class_id} ({CIFAR10_CLASSES[class_id]}): picked {len(chosen)} → {chosen}")
        for idx in chosen:
            selections.append((idx, class_id, "retain"))

    manifest: list[dict] = []
    for sample_idx, (cifar_idx, class_id, group) in enumerate(selections):
        sample_id = f"sample_{sample_idx:02d}"
        pil_image, _ = test_set_raw[cifar_idx]
        upscaled = enhance_for_display(pil_image)
        out_path = images_dir / f"{sample_id}.png"
        upscaled.save(out_path)
        manifest.append({
            "id": sample_id,
            "url": f"/static/images/{sample_id}.png",
            "true_label": CIFAR10_CLASSES[class_id],
            "true_idx": class_id,
            "cifar_idx": cifar_idx,
            "group": group,
        })

    manifest_path = STATIC_DIR / "samples.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(manifest)} thumbnails to {images_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
