# RPU Interactive Demo

A single-page web interface that illustrates **RPU — Representation-Prototype-guided Unlearning**: a machine-unlearning method that makes a deployed model "forget" a class as cleanly as if it had been retrained from scratch on only the remaining data.

The page has three sections:

1. **What is RPU** — plain-English explanation of the two-stage method.
2. **Unlearning Results on CIFAR-10** — the headline numbers from the paper and a per-class accuracy bar chart.
3. **Live Inference Demo** — click a curated CIFAR-10 thumbnail or drop in your own image and watch both real checkpoints predict side by side. The original model recognises the airplane class; the RPU-unlearned model no longer does.

Styling follows USYD's orange-and-white palette.

---

## Quick start

From this directory:

```bash
make
```

That single command takes care of everything end-to-end. On a fresh machine it will:

1. Create a Python virtual environment in `.venv/`.
2. Install Flask, PyTorch, torchvision, NumPy, and Pillow into it.
3. **Download both checkpoints** (`original_model.pt`, `advance_RPU.pt`, ~112 MB total) from the project's GitHub repo at <https://github.com/v1n4k/COMP5405_RPU> into `checkpoints/`.
4. **Download CIFAR-10** (~170 MB) from `cs.toronto.edu` via torchvision into `data/`.
5. Pick 14 curated thumbnails that produce a clean before/after story and write them to `static/images/`.
6. Start the Flask server on <http://localhost:5050>.

Open the URL in a browser when the console prints the banner. First run takes ~3 min on a normal connection; subsequent runs skip every step that's already done.

### Network sources

| What       | Where it comes from                                                                              | Size    |
| ---------- | ------------------------------------------------------------------------------------------------ | ------- |
| Checkpoints | `https://raw.githubusercontent.com/v1n4k/COMP5405_RPU/main/checkpoints/*.pt`                    | ~112 MB |
| CIFAR-10    | `https://www.cs.toronto.edu/~kriz/cifar.html` (handled by `torchvision.datasets.CIFAR10`)        | ~170 MB |

### Overrides

```bash
make PORT=8000             # different port
make PYTHON=python3.11     # different interpreter for the venv
make ORIGINAL_URL=<url> RPU_URL=<url>   # mirror the checkpoints somewhere else
```

### Other targets

```bash
make install              # only create .venv and install dependencies
make fetch-checkpoints    # only download the two .pt files
make setup                # only (re)pick the curated thumbnails (auto-downloads CIFAR-10 if needed)
make run                  # only start the server (assumes everything above is done)
make clean                # remove .venv, generated thumbnails, and samples.json
make distclean            # clean + remove downloaded checkpoints and CIFAR-10 data
make help                 # list everything above
```

---

## Submission contents

The submitted `.zip` is intentionally minimal — only source files, no binaries:

```
web_demo/
├── Makefile
├── README.md
├── requirements.txt
├── app.py
├── extract_samples.py
├── rpu_inference.py
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

That's roughly **75 KB**. `make` recreates everything else (venv, checkpoints, data, thumbnails, samples manifest) on first run. The marker only needs Python ≥ 3.10 and internet access.

---

## Repository layout (after `make` has run)

```
.
├── Makefile, README.md, requirements.txt    # shipped
├── app.py, extract_samples.py, rpu_inference.py  # shipped
├── templates/index.html                     # shipped
├── static/
│   ├── css/style.css, js/main.js            # shipped
│   ├── images/                              # generated (`make setup`)
│   └── samples.json                         # generated (`make setup`)
├── checkpoints/                             # fetched (`make fetch-checkpoints`)
│   ├── original_model.pt
│   └── advance_RPU.pt
├── data/                                    # fetched (torchvision, on `make setup`)
│   └── cifar-10-batches-py/
└── .venv/                                   # created (`make install`)
```

---

## How it works

### `rpu_inference.py`
- Defines `CIFAR10VGG16` — a `torchvision.models.vgg16` backbone with an adaptive 1×1 pool and a linear CIFAR-10 head.
- Implements `load_model`, tolerant of common checkpoint wrappings (`model_state_dict`, `state_dict`, `model`, or a bare state dict; with or without a `module.` prefix from `nn.DataParallel`).
- Exposes `predict_one(model, image, device, topk)` for single-image top-k inference.
- `resolve_device("auto")` picks CUDA → Apple MPS → CPU automatically.

### `extract_samples.py`
Runs both checkpoints once and **filters the test set for samples that tell the right story**:

- **Forget set (8 airplane images)**: original model is confidently correct AND the RPU model's top-1 has shifted away from `airplane`.
- **Retain set (2 bird + 2 ship + 2 automobile)**: both models predict the true class as top-1.

Each chosen sample is upscaled to 768×768 with a bicubic → Lanczos → UnsharpMask pipeline so the thumbnails look presentable on a projector. The model itself still sees the original 32×32 normalised tensor — display sharpening is illustration-only.

### `app.py` (Flask)

| Route              | Method | What it does                                                                                                            |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------- |
| `/`                | GET    | Renders the page with the curated samples list and the paper's metrics baked in.                                        |
| `/predict`         | POST   | `{"sample_id":"sample_NN"}` → top-3 predictions from both models on the pre-cached 32×32 CIFAR-10 tensor.               |
| `/predict_upload`  | POST   | `multipart/form-data` upload — resizes to 32×32, normalises with CIFAR-10 statistics, predicts on both models.          |
| `/status`          | GET    | Live telemetry: device, process RAM, MPS allocated memory, PyTorch version, both models' file sizes and parameter counts. |

The page polls `/status` every 3 seconds so the audience can watch resident memory tick during the demo.

---

## Requirements

- Python ≥ 3.10
- ~2 GB free disk for the virtual environment (PyTorch wheels are large)
- Internet on first run (for the two downloads above)
- macOS / Linux / WSL. CUDA and Apple MPS are used automatically when available; otherwise CPU.

---

## Tips for a live presentation

- **Warm-up**: load the page and click one airplane thumbnail before the audience arrives. The first prediction triggers an MPS kernel compile (~250 ms); subsequent calls are ~5 ms.
- **Upload demo**: drag any real airplane photo onto the upload zone. The original model still says `airplane`; the RPU model no longer does — same forgetting behaviour on arbitrary input.
- **Status panel**: the live RAM / MPS pulsing dot is good visual proof that the models are actually loaded; flag it during the intro.
- **Port conflict on macOS**: port 5000 is reserved by AirPlay Receiver — that's why the default here is 5050. Override with `make PORT=...` if needed.
- **Offline at presentation time?** Run `make` once at home with internet so the venv, checkpoints, and CIFAR-10 are cached locally. From then on `make run` works offline.

---

## Acknowledgements

The two checkpoints (`original_model.pt`, `advance_RPU.pt`) were produced by the training and unlearning pipeline in this submission's `code/` directory. Quantitative metrics in Section 2 come from the report (`asm_submission/RPU/main.pdf`, Table 1).
