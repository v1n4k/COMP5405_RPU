"""Flask backend for the RPU interactive demo.

Loads both real checkpoints once at startup, exposes a /predict endpoint that
runs the original and RPU-unlearned models on the requested CIFAR-10 test
sample, and serves the single-page interface from templates/index.html.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import datasets, transforms

from rpu_inference import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    load_model,
    predict_one,
    resolve_device,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DATA_DIR = SCRIPT_DIR / "data"
STATIC_DIR = SCRIPT_DIR / "static"

PAPER_METRICS = {
    "forget_train": {"original": 1.000, "rpu": 0.000, "reading": "Complete forgetting"},
    "retain_train": {"original": 1.000, "rpu": 0.998, "reading": "Utility preserved"},
    "forget_test":  {"original": 1.000, "rpu": 0.000, "reading": "Generalises to unseen data"},
    "retain_test":  {"original": 0.927, "rpu": 0.926, "reading": "No regression"},
    "rtd":          {"original": None,  "rpu": 0.976, "reading": "Behaves like a retrained model"},
    "mia":          {"original": 0.620, "rpu": 0.499, "reading": "Privacy: random-guess level"},
}

PER_CLASS_TEST_ACC = {
    "airplane":   {"original": 0.93, "rpu": 0.000},
    "automobile": {"original": 0.96, "rpu": 0.961},
    "bird":       {"original": 0.89, "rpu": 0.892},
    "cat":        {"original": 0.85, "rpu": 0.854},
    "deer":       {"original": 0.91, "rpu": 0.913},
    "dog":        {"original": 0.88, "rpu": 0.879},
    "frog":       {"original": 0.94, "rpu": 0.939},
    "horse":      {"original": 0.93, "rpu": 0.932},
    "ship":       {"original": 0.95, "rpu": 0.948},
    "truck":      {"original": 0.94, "rpu": 0.937},
}

TIME_COMPARISON = {"rpu": 25.7, "retrain": 163.9}


def device_pretty(device: torch.device) -> str:
    if device.type == "mps":
        return "MPS (Apple GPU)"
    if device.type == "cuda":
        return f"CUDA — {torch.cuda.get_device_name(device)}"
    return "CPU"


def process_rss_mb() -> float:
    """Current resident-set size of this process in MB (live, not peak)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        )
        return round(int(out.strip()) / 1024.0, 1)  # ps returns KB
    except Exception:
        return -1.0


def mps_allocated_mb() -> float | None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None
    try:
        return round(torch.mps.current_allocated_memory() / 1024.0 / 1024.0, 1)
    except Exception:
        return None


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_app() -> Flask:
    device = resolve_device("auto")
    print(f"[demo] device = {device}")

    original_path = CHECKPOINTS_DIR / "original_model.pt"
    unlearned_path = CHECKPOINTS_DIR / "advance_RPU.pt"
    load_t0 = time.time()
    original_model, _ = load_model(original_path, device)
    unlearned_model, _ = load_model(unlearned_path, device)
    load_secs = round(time.time() - load_t0, 2)
    print(f"[demo] loaded {original_path.name} and {unlearned_path.name} in {load_secs}s")

    model_info = [
        {
            "name": "Original",
            "filename": original_path.name,
            "size_mb": round(original_path.stat().st_size / 1024.0 / 1024.0, 1),
            "params": param_count(original_model),
            "params_label": f"{param_count(original_model) / 1e6:.2f}M",
        },
        {
            "name": "After RPU",
            "filename": unlearned_path.name,
            "size_mb": round(unlearned_path.stat().st_size / 1024.0 / 1024.0, 1),
            "params": param_count(unlearned_model),
            "params_label": f"{param_count(unlearned_model) / 1e6:.2f}M",
        },
    ]
    system_info = {
        "device": str(device),
        "device_pretty": device_pretty(device),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "load_seconds": load_secs,
        "models": model_info,
    }

    manifest_path = STATIC_DIR / "samples.json"
    samples = json.loads(manifest_path.read_text())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, transform=transform, download=False)
    sample_tensors: dict[str, torch.Tensor] = {}
    for entry in samples:
        image_tensor, _ = test_set[int(entry["cifar_idx"])]
        sample_tensors[entry["id"]] = image_tensor

    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            samples=samples,
            metrics=PAPER_METRICS,
            per_class=PER_CLASS_TEST_ACC,
            class_names=CIFAR10_CLASSES,
            time_comparison=TIME_COMPARISON,
            forget_label="airplane",
            system_info=system_info,
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        sample_id = payload.get("sample_id")
        if sample_id not in sample_tensors:
            return jsonify({"error": f"unknown sample_id: {sample_id}"}), 400
        image = sample_tensors[sample_id]
        t0 = time.time()
        original_topk = predict_one(original_model, image, device, topk=3)
        original_ms = round((time.time() - t0) * 1000.0, 1)
        t1 = time.time()
        unlearned_topk = predict_one(unlearned_model, image, device, topk=3)
        rpu_ms = round((time.time() - t1) * 1000.0, 1)
        return jsonify({
            "original": [[label, float(prob)] for label, prob in original_topk],
            "rpu":      [[label, float(prob)] for label, prob in unlearned_topk],
            "timing_ms": {"original": original_ms, "rpu": rpu_ms},
        })

    upload_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    @app.post("/predict_upload")
    def predict_upload():
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return jsonify({"error": "no image uploaded"}), 400
        try:
            pil = Image.open(upload.stream).convert("RGB")
        except Exception as exc:
            return jsonify({"error": f"could not read image: {exc}"}), 400
        # The model only ever sees 32x32 — same pipeline as CIFAR-10.
        pil_32 = pil.resize((32, 32), Image.Resampling.LANCZOS)
        tensor = upload_transform(pil_32)
        t0 = time.time()
        original_topk = predict_one(original_model, tensor, device, topk=3)
        original_ms = round((time.time() - t0) * 1000.0, 1)
        t1 = time.time()
        unlearned_topk = predict_one(unlearned_model, tensor, device, topk=3)
        rpu_ms = round((time.time() - t1) * 1000.0, 1)
        return jsonify({
            "original": [[label, float(prob)] for label, prob in original_topk],
            "rpu":      [[label, float(prob)] for label, prob in unlearned_topk],
            "timing_ms": {"original": original_ms, "rpu": rpu_ms},
        })

    @app.get("/status")
    def status():
        return jsonify({
            "device": str(device),
            "device_pretty": device_pretty(device),
            "process_ram_mb": process_rss_mb(),
            "mps_allocated_mb": mps_allocated_mb(),
            "pytorch_version": torch.__version__,
            "models": model_info,
        })

    return app


app = build_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
