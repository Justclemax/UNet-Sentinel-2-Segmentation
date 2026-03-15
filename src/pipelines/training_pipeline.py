"""Training pipeline — starts from raw scenes directory (post-ingest).

Steps
-----
1. PREPROCESS — normalize S2 bands + rasterize vector masks → (image, mask) .npy pairs
2. TRAIN      — train UNet, checkpoint saved at data/models/unet_best.pt
3. EVALUATE   — IoU / Dice / pixel accuracy on held-out 20% test split
4. INFERENCE  — run predictions on all preprocessed patches

Use run_pipeline.py for the full end-to-end pipeline (ingest included).
"""

import sys
from pathlib import Path

# Make src/steps importable
sys.path.insert(0, str(Path(__file__).parent.parent / "steps"))

from zenml import pipeline

from src.steps.preprocess import preprocess_scenes
from src.steps.train_model import train_model
from src.steps.evaluate_model import evaluate_model
from src.steps.inference import run_inference


@pipeline(name="TRAINING_PIPELINE")
def training_pipeline(
    scenes_dir: str,
    vector_path: str,
    data_dir: str = "data",
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    in_channels: int = 10,
    base_features: int = 64,
    threshold: float = 0.5,
):
    """Preprocess → train → evaluate → inference.

    Parameters
    ----------
    scenes_dir    : directory containing S2_*.tif files from the ingest step
    vector_path   : GeoJSON / Shapefile with segmentation labels
    data_dir      : root output directory (preprocessed/, models/, predictions/)
    epochs        : number of training epochs
    batch_size    : mini-batch size
    learning_rate : Adam learning rate
    in_channels   : number of input bands (10 for default S2_BANDS)
    base_features : UNet base feature count (doubles each encoder level)
    threshold     : sigmoid threshold for binary predictions
    """

    # ── 1. Preprocess ────────────────────────────────────────────────────────
    pairs = preprocess_scenes(
        scenes_dir=scenes_dir,
        vector_path=vector_path,
        output_dir=f"{data_dir}/preprocessed",
    )

    # ── 2. Train ─────────────────────────────────────────────────────────────
    model_path = train_model(
        preprocessed_pairs=pairs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        in_channels=in_channels,
        base_features=base_features,
        output_dir=f"{data_dir}/models",
    )

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    evaluate_model(
        model_path=model_path,
        preprocessed_pairs=pairs,
        batch_size=batch_size,
    )

    # ── 4. Inference ─────────────────────────────────────────────────────────
    run_inference(
        model_path=model_path,
        preprocessed_pairs=pairs,
        output_dir=f"{data_dir}/predictions",
        threshold=threshold,
    )


if __name__ == "__main__":
    training_pipeline(
        scenes_dir="../../data/scenes",
        vector_path="../../data/training_data.geojson",
        data_dir="../../data",
        epochs=50,
        batch_size=8,
    )