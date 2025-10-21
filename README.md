# A patch-level Siamese CNN trained on Sentinel-2 RGB+NIR imagery from Ukraine to detect urban damage with a ResNet-101 backbone.

---

## Dataset
Dataset built from **Sentinel-2 image pairs** focused on building damage across **22** regions in Ukraine, downloaded via the [Copernicus Browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE).
Sentinel-2 provides ~10 m per pixel for these bands, so a 16×16 patch covers 160 m × 160 m (25,600 m² ≈ 0.0256 km²).

Siamese CNN needs two images so these were before and after:
* **Date rule:** *Before* = any date in **2021** · *After* = any date **on/after 2022-03-01**
* **Labels:** Derived from [UNITAR-UNOSAT](https://unosat.org/products) damage assessment shapefiles (first-instance), aligned to the before/after image pairs

**Regions included**

| 1           | 2          | 3         | 4        | 5                   |
| ----------- | ---------- | --------- | -------- | ------------------- |
| Vorzel      | Mykolaiv   | Okhtyrka  | Schastia | Sumy                |
| Trostianets | Volnovakha | Kharkiv   | Kherson  | Kramatorsk          |
| Lysychansk  | Makariv    | Melitopol | Avdiivka | Azovstal industrial |
| Bucha       | Chernihiv  | Hostomel  | Irpin    | Antonivka           |
| Rubizhne    | Kremenchuk |           |          |                     |

This github only includes the final split dataset as the individual images were preprocessed using GDAL library but are too large to upload individually.
---

## Why this model is strong for **damage detection**

* **Finds damage, rarely misses it**: *Damage* recall **0.92** → only **23** false negatives out of 296 damaged patches.
* **Triage-friendly**: A higher recall on damage is better for rapid response and follow-up assessment, even if it means a few extra false positives.
* **Robust to change**: The Siamese setup compares before vs after directly, helping the network ignore seasonal or lighting differences and focus on structural change.

---

## Model

**Architecture**: Siamese CNN with a **ResNet-101** backbone per branch.
**Feature combined formula**:

[f_before, f_after, |f_before − f_after|, f_before ⊙ f_after] → 8192-D

**MLP head**

```python
self.head = nn.Sequential(
    nn.Linear(8192, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
    nn.Linear(256, 1),
)
```

---

## Results (patch level)

**Test accuracy**: **0.826** (~0.83)

**Confusion matrix** (rows = true, cols = predicted)

|               | No-Damage | Damage |
| ------------- | --------- | ------ |
| **No-Damage** | 216       | 80     |
| **Damage**    | 23        | 273    |

**Classification report**

* No-Damage — precision **0.90**, recall **0.73**, F1 **0.81** (n=296)
* Damage — precision **0.77**, recall **0.92**, F1 **0.84** (n=296)
* Overall — accuracy **0.83**, macro F1 **0.82** (n=592)

Artifacts:

* `results/confusion_matrix_counts.png`
* `results/confusion_matrix_normalized.png`
* `results/metrics.json` with full metrics

---
**Structure of model**

<img width="391" height="511" alt="Untitled Diagram drawio (18)" src="https://github.com/user-attachments/assets/c131d09e-cdca-42e1-a9ef-4b9db13c9641" />

---
## Quick start

```bash
# Train + evaluate on a balanced NPZ of pairs
python main.py train \
  --npz data/processed/FinalBalanced_splits.npz \
  --out results \
  --ckpt models/best_resnet101.pt
```

**Input format**: NPZ with `x_before`, `x_after` shaped `(N, 4, 16, 16)` and labels `y` in `{0,1}`
**Output**: 1-logit per pair (use sigmoid at inference)

---

## Intended use

* Fast triage after events to surface likely damaged locations first
* Patch-level signals that can be aggregated to tiles or AOIs for dashboards and field planning
* Decision support for humanitarian partners, not a final damage authority


