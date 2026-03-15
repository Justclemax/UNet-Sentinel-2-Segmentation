"""
Full UNet-Sentinel2 pipeline.

Steps
-----
1. INGEST   — validate vector data, list S2 scenes, download per-feature GeoTIFFs
2. PREPROCESS — normalize bands, rasterize vector masks → (image, mask) .npy pairs
3. TRAIN    — train UNet, save best checkpoint
4. EVALUATE — compute IoU / Dice / pixel accuracy on held-out test split
5. INFERENCE — run predictions on all preprocessed patches
"""

import sys
from pathlib import Path

# Make src/steps importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src" / "steps"))

from zenml import pipeline

from src.steps.ingest_and_download_data import load_and_validate, fetch_sentinel2_scenes, export_scenes
from src.steps.preprocess import preprocess_scenes
from src.steps.train_model import train_model
from src.steps.evaluate_model import evaluate_model
from src.steps.inference import run_inference


@pipeline(name="UNET_S2_FULL_PIPELINE")
def full_pipeline(
    vector_path: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data",
    epochs: int = 50,
    batch_size: int = 8,
    threshold: float = 0.5,
):
    """End-to-end pipeline: ingest → preprocess → train → evaluate → inference."""

    # ── 1. Ingest ────────────────────────────────────────────────────────────
    validated_path = load_and_validate(vector_path)
    scenes = fetch_sentinel2_scenes(validated_path, start_date, end_date)
    scenes_dir = export_scenes(validated_path, scenes, f"{data_dir}/scenes")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    pairs = preprocess_scenes(
        scenes_dir=scenes_dir,
        vector_path=vector_path,
        output_dir=f"{data_dir}/preprocessed",
    )

    # ── 3. Train ─────────────────────────────────────────────────────────────
    model_path = train_model(
        preprocessed_pairs=pairs,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=f"{data_dir}/models",
    )

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    evaluate_model(
        model_path=model_path,
        preprocessed_pairs=pairs,
        batch_size=batch_size,
    )

    # ── 5. Inference ─────────────────────────────────────────────────────────
    run_inference(
        model_path=model_path,
        preprocessed_pairs=pairs,
        output_dir=f"{data_dir}/predictions",
        threshold=threshold,
    )


if __name__ == "__main__":
    full_pipeline(
        vector_path="data/training_data.geojson",
        start_date="2025-01-01",
        end_date="2025-01-15",
        data_dir="data",
        epochs=50,
        batch_size=8,
    )