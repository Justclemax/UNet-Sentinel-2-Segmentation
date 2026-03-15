# UNet Sentinel-2 Segmentation

Semantic segmentation pipeline for Sentinel-2 satellite imagery using a UNet architecture. The pipeline ingests geospatial vector data (GeoJSON / Shapefile), downloads Sentinel-2 scenes via Google Earth Engine, trains a UNet model, evaluates it, and runs inference — all orchestrated with ZenML.

---

## Architecture

```
unet-sentinel2-segmentation/
├── src/
│   ├── models/
│   │   └── unet.py                  # UNet architecture (canonical model definition)
│   ├── pipelines/
│   │   └── training_pipeline.py     # ZenML pipeline: preprocess → train → evaluate → inference
│   └── steps/
│       ├── ingest_and_download_data.py  # Validate vector data, list & download S2 scenes
│       ├── preprocess.py                # Normalize bands + rasterize masks → .npy pairs
│       ├── train_model.py               # Train UNet, save best checkpoint
│       ├── evaluate_model.py            # IoU / Dice / pixel accuracy on test split
│       └── inference.py                 # Run predictions, save masks
├── run_pipeline.py                  # Full end-to-end pipeline entry point
├── data/
│   └── training_data.geojson        # Input vector labels
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

### Pipeline steps

```
[GeoJSON]
    │
    ▼
load_and_validate          ← validates vector data, reprojects to EPSG:4326
    │
    ▼
fetch_sentinel2_scenes     ← queries COPERNICUS/S2_SR_HARMONIZED on Earth Engine
    │
    ▼
export_scenes              ← downloads per-feature GeoTIFF patches
    │                         naming: S2_{label}_{id:04d}_{YYYY-MM-DD}.tif
    ▼
preprocess_scenes          ← normalizes bands (/10000, clip [0,1])
    │                         rasterizes vector masks onto each patch grid
    │                         output: (C=10, 256, 256) image + (256, 256) mask .npy
    ▼
train_model                ← trains UNet with BCE + Dice loss
    │                         80/20 train/val split, saves best checkpoint
    ▼
evaluate_model             ← IoU, Dice, pixel accuracy on held-out 20%
    │
    ▼
run_inference              ← predicts masks for all patches
                              output: pred_S2_*.npy
```

---

## Requirements

- Python 3.12
- Google Earth Engine account with a project (`EE_PROJECT`)
- System libraries: GDAL, GEOS, PROJ (see Dockerfile for details)

---

## Setup

### Local

```bash
# Clone and create virtual environment
git clone https://github.com/Justclemax/UNet-Sentinel-2-Segmentation.git
cd UNet-Sentinel-2-Segmentation
python3.12 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Earth Engine credentials
cp .env.example .env   # then fill in EE_PROJECT
```

`.env` file:
```
EE_PROJECT=your-gee-project-id
```

### Docker

```bash
# Build the image
docker compose build

# Run the full pipeline
docker compose up pipeline

# Run training only (requires data/scenes/ already populated)
docker compose up train
```

---

## Usage

### Full pipeline (ingest + train + evaluate + inference)

```bash
python run_pipeline.py
```

Configure dates and paths directly in `run_pipeline.py`:

```python
full_pipeline(
    vector_path="data/training_data.geojson",
    start_date="2025-01-01",
    end_date="2025-01-15",
    data_dir="data",
    epochs=50,
    batch_size=8,
)
```

### Training only (scenes already downloaded)

```bash
python src/pipelines/training_pipeline.py
```

---

## Model

**UNet** — Ronneberger et al., MICCAI 2015

| Parameter | Value |
|---|---|
| Input | (B, 10, 256, 256) — 10 Sentinel-2 bands |
| Output | (B, 1, 256, 256) — binary mask logits |
| Encoder levels | 4 (base features: 64, 128, 256, 512) |
| Bottleneck | 1024 features + Dropout (0.3) |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimizer | Adam (lr=1e-4) |

### Sentinel-2 bands used

| Band | Description | Resolution |
|---|---|---|
| B2 | Blue | 10 m |
| B3 | Green | 10 m |
| B4 | Red | 10 m |
| B5 | Red Edge 1 | 20 m |
| B6 | Red Edge 2 | 20 m |
| B7 | Red Edge 3 | 20 m |
| B8 | NIR | 10 m |
| B8A | Narrow NIR | 20 m |
| B11 | SWIR 1 | 20 m |
| B12 | SWIR 2 | 20 m |

---

## Output files

| Path | Description |
|---|---|
| `data/scenes/S2_{label}_{id}_{date}.tif` | Raw downloaded GeoTIFF patches |
| `data/preprocessed/S2_*.npy` | Normalized image arrays (10, 256, 256) float32 |
| `data/preprocessed/mask_S2_*.npy` | Rasterized label masks (256, 256) uint8 |
| `data/models/unet_best.pt` | Best model checkpoint (by val loss) |
| `data/predictions/pred_S2_*.npy` | Predicted binary masks (256, 256) uint8 |

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

| Job | Trigger | Action |
|---|---|---|
| `lint` | every push / PR | `ruff check src/` |
| `test` | after lint | `pytest tests/` |
| `docker` | after lint | Docker build (no push) |
| `push` | push to `main` only | Push image to GitHub Container Registry (`ghcr.io`) |