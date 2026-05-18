"""Flask backend for the RPU interactive demo.

Loads both real checkpoints once at startup, exposes a /predict endpoint that
runs the original and RPU-unlearned models on the requested CIFAR-10 test
sample, and serves the single-page interface from templates/index.html.

The page content is structured for a short video walkthrough:
  1. What is RPU (problem + one-sentence definition + mechanism flow)
  2. Training & Evaluation protocol (recipe + RPU hyperparams + metrics legend)
  3. Results from the paper (Main Table on CIFAR-10 + RTD ranking + findings)
  4. Live demo (real checkpoints, real inference)
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

# --------------------------------------------------------------------------
# Section 2: training & evaluation protocol
# Values lifted from RPU/experiment_setting/experiment_setting.tex:
#   - training recipe Table 1 (CIFAR-10 row)
#   - RPU defaults (tau, kappa, lambda_f, epochs, batch, lr)
#   - forget-class protocol (3 classes x 5 seeds = 15 runs)
# --------------------------------------------------------------------------

TRAINING_RECIPE = [
    {"k": "Architecture",   "v": "VGGNet-16"},
    {"k": "Optimizer",      "v": "SGD, momentum 0.9"},
    {"k": "Learning rate",  "v": "1e-3 (fixed, no decay)"},
    {"k": "Batch size",     "v": "32"},
    {"k": "Weight decay",   "v": "1e-4"},
    {"k": "Epochs",         "v": "15"},
]

RPU_HYPERPARAMS = [
    {"k": "Initialised from",   "v": "the original checkpoint"},
    {"k": "Epochs",             "v": "5 (no LR decay)"},
    {"k": "Learning rate",      "v": "2e-3"},
    {"k": "Batch size",         "v": "64"},
    {"k": "τ (prototype temp.)","v": "0.05"},
    {"k": "κ (anchor baseline)","v": "0.5"},
    {"k": "λ_f (forget KL)",    "v": "1.0"},
]

METRICS_LEGEND = [
    {"key": "UA&nbsp;&darr;",  "desc": "Forget-train accuracy — lower means stronger removal of the forgotten samples."},
    {"key": "RA&nbsp;&uarr;",  "desc": "Retain-train accuracy — higher means retained utility is preserved."},
    {"key": "TUA&nbsp;&darr;", "desc": "Forget-test accuracy — checks that forgetting generalises to unseen forget-class images."},
    {"key": "TRA&nbsp;&uarr;", "desc": "Retain-test accuracy — held-out utility on retained classes."},
    {"key": "RTD&nbsp;&uarr;", "desc": "Retrain-Aligned Transition Decontamination — KL-based score; 1.0 = exact retrain behaviour."},
    {"key": "MIA (≈0.5)",      "desc": "Membership-inference ROC-AUC — closer to 0.5 means weaker privacy leakage."},
    {"key": "Time&nbsp;&darr;","desc": "Wall-clock unlearning runtime (seconds)."},
]

# --------------------------------------------------------------------------
# Section 3: Main Results — CIFAR-10 / VGGNet-16
# Values lifted directly from RPU/Table/Main Table.tex (centralized results).
# Each cell is (value, is_unique_best_among_unlearning_methods).
# Reference rows (Original, Retrain) are tagged but excluded from "best".
# --------------------------------------------------------------------------

# (method, type, ua, ra, tua, tra, rtd, mia, time)
# type ∈ {reference, baseline, ours}
_CIFAR10_MAIN_TABLE_RAW: list[tuple] = [
    ("Original",     "reference", 0.902, 0.948, 0.791, 0.916, None,   None,  None),
    ("Retrain",      "reference", 0.000, 0.944, 0.003, 0.915, 1.000,  0.502, 163.93),
    ("Bad Teacher",  "baseline",  0.076, 0.934, 0.090, 0.904, 0.436,  0.482, 59.51),
    ("Random Label", "baseline",  0.004, 0.928, 0.011, 0.903, 0.434,  0.494, 60.37),
    ("SCRUB",        "baseline",  0.067, 0.813, 0.046, 0.793, 0.076,  0.513, 116.27),
    ("GA",           "baseline",  0.007, 0.563, 0.000, 0.545, -1.776, 0.492, 66.28),
    ("SalUn",        "baseline",  0.000, 0.904, 0.000, 0.914, 0.803,  0.509, 32.95),
    ("SSD",          "baseline",  0.000, 0.828, 0.000, 0.815, 0.784,  0.775, 14.54),
    ("AL+WPGD",      "baseline",  0.018, 0.847, 0.005, 0.907, 0.770,  0.521, 25.87),
    ("DELETE",       "baseline",  0.000, 0.989, 0.000, 0.934, 0.771,  0.536, 23.97),
    ("TRW",          "baseline",  0.000, 0.978, 0.000, 0.902, -1.078, 0.513, 24.78),
    ("RPU (Ours)",   "ours",      0.000, 0.998, 0.000, 0.926, 0.976,  0.499, 25.67),
]

# Unique-best markers (from the paper's bolding scheme). For metrics where
# several methods tie at the best value (e.g. UA=0, TUA=0), no bold is applied.
_BEST_CELLS = {
    ("RPU (Ours)", "ra"),     # 0.998 — best RA
    ("DELETE",     "tra"),    # 0.934 — best TRA (above RPU's 0.926)
    ("RPU (Ours)", "rtd"),    # 0.976 — best RTD
    ("RPU (Ours)", "mia"),    # 0.499 — closest to 0.5
    ("SSD",        "time"),   # 14.54s — fastest among unlearning methods
}


def _fmt_cell(method: str, key: str, value):
    if value is None:
        return None
    is_time = key == "time"
    formatted = f"{value:.2f}" if is_time else f"{value:.3f}"
    if value < 0:
        formatted = f"−{abs(value):.3f}" if not is_time else f"−{abs(value):.2f}"
    return {"fmt": formatted, "bold": (method, key) in _BEST_CELLS}


def build_main_table() -> list[dict]:
    """Format the raw paper numbers for template rendering."""
    rows = []
    for method, kind, ua, ra, tua, tra, rtd, mia, t in _CIFAR10_MAIN_TABLE_RAW:
        rows.append({
            "method": method,
            "type": kind,
            "ua":   _fmt_cell(method, "ua",   ua),
            "ra":   _fmt_cell(method, "ra",   ra),
            "tua":  _fmt_cell(method, "tua",  tua),
            "tra":  _fmt_cell(method, "tra",  tra),
            "rtd":  _fmt_cell(method, "rtd",  rtd),
            "mia":  _fmt_cell(method, "mia",  mia),
            "time": _fmt_cell(method, "time", t),
        })
    return rows


def build_rtd_ranking() -> list[dict]:
    """Horizontal bar chart data: sort unlearning methods + Retrain by RTD desc.

    RTD ranges roughly [-1.8, 1.0]. We map this onto a 0–100% bar with a
    centered "zero" anchor so negative RTDs visibly fall to the left of the
    axis.
    """
    rtd_min, rtd_max = -1.8, 1.0
    span = rtd_max - rtd_min
    zero_offset = (0 - rtd_min) / span * 100   # x-position of RTD=0

    rows = []
    for method, kind, _, _, _, _, rtd, _, _ in _CIFAR10_MAIN_TABLE_RAW:
        if rtd is None:
            continue
        # bar from 0 → rtd (left if negative, right if positive)
        if rtd >= 0:
            offset = zero_offset
            width = rtd / span * 100
        else:
            width = abs(rtd) / span * 100
            offset = zero_offset - width
        rows.append({
            "method": method,
            "rtd": rtd,
            "rtd_fmt": f"{rtd:.3f}" if rtd >= 0 else f"−{abs(rtd):.3f}",
            "width": max(width, 0.5),  # keep a sliver visible at RTD=0
            "offset": offset,
            "is_ours": kind == "ours",
            "is_reference": kind == "reference",
        })
    rows.sort(key=lambda r: r["rtd"], reverse=True)
    return rows


TIME_COMPARISON = {"rpu": 25.67, "retrain": 163.93}

EXTRA_FINDINGS = [
    "<strong>CIFAR-100 / ResNet-34</strong> &mdash; the same trends hold: RPU again leads on RTD (0.847) at a small fraction of the retrain runtime (59&nbsp;s vs 540&nbsp;s).",
    "<strong>Component ablation</strong> &mdash; full RPU is the unique best on <em>RA</em> (0.988), <em>TRA</em> (0.728), and <em>RTD</em> (0.847) over either ablated variant. The renormalisation anchor alone preserves preferences but stays contaminated; prototypes alone correct structure but lose retained utility &mdash; the two are complementary, not interchangeable.",
    "<strong>Multi-class forgetting</strong> &mdash; scaling the forget set to 3&nbsp;/&nbsp;5&nbsp;/&nbsp;10&nbsp;/&nbsp;15 classes on CIFAR-100, RPU's RTD stays between 0.876 and 0.912, while baselines collapse on either forgetting or alignment.",
    "<strong>Hyperparameter robustness</strong> &mdash; varying &tau;, &kappa;, &lambda;<sub>f</sub>, and the unlearning epoch budget one at a time, RPU stays within a narrow RA / TRA / RTD band across all reasonable settings &mdash; the defaults are not a single-point sweet spot.",
]


# --------------------------------------------------------------------------
# Live-demo helpers
# --------------------------------------------------------------------------

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

    main_table = build_main_table()
    rtd_ranking = build_rtd_ranking()

    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            samples=samples,
            class_names=CIFAR10_CLASSES,
            system_info=system_info,
            training_recipe=TRAINING_RECIPE,
            rpu_hyperparams=RPU_HYPERPARAMS,
            metrics_legend=METRICS_LEGEND,
            main_table=main_table,
            rtd_ranking=rtd_ranking,
            time_comparison=TIME_COMPARISON,
            extra_findings=EXTRA_FINDINGS,
            forget_label="airplane",
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
