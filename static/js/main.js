(() => {
    const cards = document.querySelectorAll(".sample-card");
    const previewImg = document.getElementById("preview-image");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const previewTruth = document.getElementById("preview-truth");
    const originalList = document.getElementById("original-preds");
    const rpuList = document.getElementById("rpu-preds");

    const verdicts = {
        original: {
            wrap: document.getElementById("original-verdict"),
            label: document.getElementById("original-verdict-label"),
            prob: document.getElementById("original-verdict-prob"),
            tag: document.getElementById("original-verdict-tag"),
        },
        rpu: {
            wrap: document.getElementById("rpu-verdict"),
            label: document.getElementById("rpu-verdict-label"),
            prob: document.getElementById("rpu-verdict-prob"),
            tag: document.getElementById("rpu-verdict-tag"),
        },
    };

    const statusRam = document.getElementById("status-ram");
    const statusMps = document.getElementById("status-mps");

    const uploadInput = document.getElementById("upload-input");
    const uploadZone = document.getElementById("upload-zone");

    // ---------- Status polling ----------
    const refreshStatus = async () => {
        try {
            const res = await fetch("/status");
            if (!res.ok) return;
            const data = await res.json();
            if (statusRam) {
                statusRam.textContent = data.process_ram_mb > 0
                    ? `${data.process_ram_mb.toFixed(0)} MB`
                    : "—";
            }
            if (statusMps && data.mps_allocated_mb !== null && data.mps_allocated_mb !== undefined) {
                statusMps.textContent = `${data.mps_allocated_mb.toFixed(0)} MB`;
            }
        } catch (err) {
            // leave last value
        }
    };
    refreshStatus();
    setInterval(refreshStatus, 3000);

    // ---------- Rendering helpers ----------
    const clearSelection = () => {
        cards.forEach((c) => c.classList.remove("selected"));
        uploadZone.classList.remove("uploaded");
    };

    const renderPreview = (url, truthLabel) => {
        previewImg.src = url;
        previewImg.alt = truthLabel;
        previewImg.hidden = false;
        previewPlaceholder.hidden = true;
        previewTruth.textContent = truthLabel;
    };

    const renderLoading = () => {
        const html = '<p class="pred-empty"><span class="spinner"></span>Running inference…</p>';
        originalList.innerHTML = html;
        rpuList.innerHTML = html;
        verdicts.original.wrap.hidden = true;
        verdicts.rpu.wrap.hidden = true;
    };

    const renderPredList = (target, predictions) => {
        if (!predictions || !predictions.length) {
            target.innerHTML = '<p class="pred-empty">No predictions returned.</p>';
            return;
        }
        const items = predictions.map(([label, prob], idx) => {
            const pct = (prob * 100).toFixed(1);
            const topClass = idx === 0 ? " is-top" : "";
            return `
                <div class="pred-bar${topClass}">
                    <div class="pred-bar-header">
                        <span class="pred-label">${label}</span>
                        <span class="pred-prob">${pct}%</span>
                    </div>
                    <div class="bar-fill-track">
                        <div class="bar-fill" style="width: ${pct}%;"></div>
                    </div>
                </div>
            `;
        });
        target.innerHTML = items.join("");
    };

    let lastOriginalTop = null;

    const renderVerdict = (which, predictions, trueLabel) => {
        const v = verdicts[which];
        if (!predictions || !predictions.length) {
            v.wrap.hidden = true;
            return;
        }
        const [topLabel, topProb] = predictions[0];
        v.label.textContent = topLabel;
        v.prob.textContent = `${(topProb * 100).toFixed(1)}%`;
        v.tag.className = "verdict-tag";

        if (which === "original") {
            lastOriginalTop = topLabel;
        }

        if (trueLabel === null) {
            // Uploaded image — no ground truth. Tag describes cross-model behaviour.
            if (which === "original") {
                v.tag.textContent = "";
            } else {
                if (topLabel === "airplane") {
                    v.tag.classList.add("tag-mismatch");
                    v.tag.textContent = "still predicts airplane";
                } else if (lastOriginalTop === "airplane") {
                    v.tag.classList.add("tag-forgot");
                    v.tag.textContent = "✓ no longer predicts airplane";
                } else if (lastOriginalTop && lastOriginalTop !== topLabel) {
                    v.tag.classList.add("tag-mismatch");
                    v.tag.textContent = "differs from original";
                } else {
                    v.tag.textContent = "";
                }
            }
        } else if (which === "original") {
            v.tag.classList.add(topLabel === trueLabel ? "tag-correct" : "tag-mismatch");
            v.tag.textContent = topLabel === trueLabel ? "✓ correct" : "✗ misclassified";
        } else {
            if (trueLabel === "airplane") {
                v.tag.classList.add(topLabel === "airplane" ? "tag-mismatch" : "tag-forgot");
                v.tag.textContent = topLabel === "airplane" ? "still remembers" : "✓ forgot airplane";
            } else {
                v.tag.classList.add(topLabel === trueLabel ? "tag-correct" : "tag-mismatch");
                v.tag.textContent = topLabel === trueLabel ? "✓ retained" : "✗ drifted";
            }
        }
        v.wrap.hidden = false;
    };

    const renderError = (message) => {
        const html = `<p class="pred-error">${message}</p>`;
        originalList.innerHTML = html;
        rpuList.innerHTML = html;
        verdicts.original.wrap.hidden = true;
        verdicts.rpu.wrap.hidden = true;
    };

    // ---------- Curated-thumbnail click ----------
    const handleCardClick = async (event) => {
        const card = event.currentTarget;
        const trueLabel = card.dataset.label;
        clearSelection();
        card.classList.add("selected");
        renderPreview(card.dataset.url, trueLabel);
        renderLoading();
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_id: card.dataset.id }),
            });
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            const data = await response.json();
            renderPredList(originalList, data.original);
            renderPredList(rpuList, data.rpu);
            renderVerdict("original", data.original, trueLabel);
            renderVerdict("rpu", data.rpu, trueLabel);
        } catch (err) {
            renderError("Backend offline — start `python app.py` and reload.");
            console.error(err);
        }
        refreshStatus();
    };

    cards.forEach((card) => card.addEventListener("click", handleCardClick));

    // ---------- Upload handling ----------
    const handleUpload = async (file) => {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            renderError("Please choose an image file.");
            return;
        }
        clearSelection();
        uploadZone.classList.add("uploaded");

        const reader = new FileReader();
        reader.onload = (e) => renderPreview(e.target.result, "uploaded");
        reader.readAsDataURL(file);

        renderLoading();
        const formData = new FormData();
        formData.append("image", file);
        try {
            const response = await fetch("/predict_upload", {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                throw new Error(body.error || `Server returned ${response.status}`);
            }
            const data = await response.json();
            renderPredList(originalList, data.original);
            renderPredList(rpuList, data.rpu);
            renderVerdict("original", data.original, null);
            renderVerdict("rpu", data.rpu, null);
        } catch (err) {
            renderError(`Upload failed: ${err.message}`);
            console.error(err);
        }
        refreshStatus();
    };

    if (uploadInput) {
        uploadInput.addEventListener("change", (e) => {
            handleUpload(e.target.files[0]);
            e.target.value = "";  // allow re-uploading the same file
        });
    }
    if (uploadZone) {
        uploadZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadZone.classList.add("drag-over");
        });
        uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
        uploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadZone.classList.remove("drag-over");
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (file) handleUpload(file);
        });
    }
})();
