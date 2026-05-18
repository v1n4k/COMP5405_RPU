# RPU Interactive Demo

A single-page Flask app that runs two CIFAR-10 VGG-16 checkpoints — original vs. RPU-unlearned — side by side on curated thumbnails or uploaded images.

---

## Quick start

From this directory:

```bash
make
```

Installs the venv, downloads the checkpoints and CIFAR-10, extracts curated thumbnails, and serves the page at <http://localhost:5050>. First run takes ~3 min; subsequent runs are instant.

### Targets

```bash
make                    # full pipeline: install + fetch + setup + run
make install            # create .venv and install dependencies
make fetch-checkpoints  # download the two .pt files
make setup              # extract curated CIFAR-10 thumbnails
make run                # start the Flask server only
make clean              # remove .venv, thumbnails, samples.json
make distclean          # clean + remove checkpoints and CIFAR-10 data
make help               # list every target
```

### Overrides

```bash
make PORT=8000                            # different port
make PYTHON=python3.11                    # different interpreter
make ORIGINAL_URL=<url> RPU_URL=<url>     # mirror the checkpoints elsewhere
```

---

## Repo layout

```
web_demo/
├── Makefile, README.md, requirements.txt
├── app.py                 # Flask server
├── extract_samples.py     # curated thumbnail picker
├── rpu_inference.py       # model loading + prediction helpers
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

`make` generates `.venv/`, `checkpoints/`, `data/`, `static/images/`, and `static/samples.json` on first run.

---

## Requirements

- Python ≥ 3.10
- ~2 GB free disk (PyTorch wheels)
- Internet on first run
- macOS / Linux / WSL — CUDA or Apple MPS is used automatically when available
