# RPU interactive web demo — single entry point.
#
# Quick start (from this directory):
#   make             # install deps + fetch checkpoints + extract samples + run
#   make run         # only start the server (everything else already done)
#   make clean       # remove .venv and generated thumbnails/samples
#   make distclean   # also remove downloaded checkpoints and CIFAR-10 data
#
# Overrides:
#   make PORT=8000
#   make PYTHON=python3.11
#   make ORIGINAL_URL=... RPU_URL=...     (point at a different checkpoint host)

PYTHON       ?= python3
PORT         ?= 5050
VENV         := .venv
VPIP         := $(VENV)/bin/pip
VPYTHON      := $(VENV)/bin/python

# Checkpoint URLs — raw GitHub by default. Override on the command line for
# local mirrors or alternative hosts.
ORIGINAL_URL ?= https://raw.githubusercontent.com/v1n4k/COMP5405_RPU/main/checkpoints/original_model.pt
RPU_URL      ?= https://raw.githubusercontent.com/v1n4k/COMP5405_RPU/main/checkpoints/advance_RPU.pt

REQUIRED_CHECKPOINTS := checkpoints/original_model.pt checkpoints/advance_RPU.pt
SAMPLES_MANIFEST     := static/samples.json

.PHONY: all help install fetch-checkpoints setup run clean distclean

# --- Default: bring everything up and start the server ---
all: run

help:
	@echo "RPU Interactive Demo"
	@echo ""
	@echo "Targets:"
	@echo "  make                       (default) install + fetch + setup + run on http://localhost:$(PORT)"
	@echo "  make install               create $(VENV)/ and install dependencies"
	@echo "  make fetch-checkpoints     download both .pt files from GitHub if missing"
	@echo "  make setup                 extract curated thumbnails (auto-downloads CIFAR-10 on first run)"
	@echo "  make run                   start the Flask server"
	@echo "  make clean                 remove $(VENV)/, generated thumbnails, and samples.json"
	@echo "  make distclean             clean + remove downloaded checkpoints and CIFAR-10 data"
	@echo ""
	@echo "Overrides:"
	@echo "  make PORT=8000                                 (different port)"
	@echo "  make PYTHON=python3.11                         (different interpreter)"
	@echo "  make ORIGINAL_URL=... RPU_URL=...              (different checkpoint host)"

# --- 1. Install: venv + pip install ---
$(VENV)/.installed: requirements.txt
	@echo ">>> Creating virtual environment in $(VENV)/"
	$(PYTHON) -m venv $(VENV)
	$(VPIP) install --upgrade pip
	$(VPIP) install -r requirements.txt
	@touch $(VENV)/.installed

install: $(VENV)/.installed

# --- 2. Fetch checkpoints from GitHub ---
# Each .pt has its own rule with no prerequisites — make only triggers the
# download when the file is missing. A partial download is written to .part
# first so an interrupted curl doesn't leave a corrupt .pt in place.
checkpoints/original_model.pt:
	@mkdir -p checkpoints
	@echo ">>> Fetching original_model.pt"
	@echo "    from $(ORIGINAL_URL)"
	@curl -fL --retry 3 -o checkpoints/original_model.pt.part "$(ORIGINAL_URL)" \
		&& mv checkpoints/original_model.pt.part checkpoints/original_model.pt \
		|| (rm -f checkpoints/original_model.pt.part; \
		    echo ""; echo "ERROR: failed to download original_model.pt"; \
		    echo "  Tried: $(ORIGINAL_URL)"; \
		    echo "  Check network or override with: make ORIGINAL_URL=<url>"; \
		    exit 1)

checkpoints/advance_RPU.pt:
	@mkdir -p checkpoints
	@echo ">>> Fetching advance_RPU.pt"
	@echo "    from $(RPU_URL)"
	@curl -fL --retry 3 -o checkpoints/advance_RPU.pt.part "$(RPU_URL)" \
		&& mv checkpoints/advance_RPU.pt.part checkpoints/advance_RPU.pt \
		|| (rm -f checkpoints/advance_RPU.pt.part; \
		    echo ""; echo "ERROR: failed to download advance_RPU.pt"; \
		    echo "  Tried: $(RPU_URL)"; \
		    echo "  Check network or override with: make RPU_URL=<url>"; \
		    exit 1)

fetch-checkpoints: $(REQUIRED_CHECKPOINTS)

# --- 3. Setup: extract curated thumbnails (also triggers CIFAR-10 download) ---
$(SAMPLES_MANIFEST): $(VENV)/.installed extract_samples.py rpu_inference.py $(REQUIRED_CHECKPOINTS)
	@echo ">>> Generating curated CIFAR-10 thumbnails"
	@echo "    (torchvision auto-downloads CIFAR-10 from cs.toronto.edu on first run)"
	$(VPYTHON) extract_samples.py

setup: $(SAMPLES_MANIFEST)

# --- 4. Run the Flask server ---
run: $(VENV)/.installed $(SAMPLES_MANIFEST)
	@echo ""
	@echo ">>> RPU demo is ready at http://localhost:$(PORT)"
	@echo "    Press Ctrl+C to stop."
	@echo ""
	PORT=$(PORT) $(VPYTHON) app.py

# --- 5. Cleanup ---
clean:
	rm -rf $(VENV)
	rm -f $(SAMPLES_MANIFEST)
	rm -f static/images/sample_*.png
	rm -rf __pycache__ */__pycache__

distclean: clean
	rm -rf checkpoints data
