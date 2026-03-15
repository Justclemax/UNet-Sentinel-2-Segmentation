from pathlib import Path

import numpy as np
import torch
from loguru import logger
from zenml import step, pipeline

from src.models.unet import UNet

# ── ZenML step ────────────────────────────────────────────────────────────────


@step
def run_inference(
    model_path: str,
    preprocessed_pairs: list[dict],
    output_dir: str = "data/predictions",
    threshold: float = 0.5,
) -> list[str]:
    """Run trained UNet on preprocessed patches and save predicted masks.

    Inputs
    ------
    model_path         : path to the .pt checkpoint from train_model
    preprocessed_pairs : list[{"image": str, "mask": str}] from preprocess_scenes
                         (only the "image" key is used — labels are ignored)
    output_dir         : where to write predicted mask .npy files
    threshold          : sigmoid threshold for binary classification

    Returns
    -------
    List of paths to saved prediction .npy files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu")
    model = UNet(
        in_channels=checkpoint["in_channels"],
        base_features=checkpoint["base_features"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"🔍 Running inference on {len(preprocessed_pairs)} patches")

    saved: list[str] = []
    with torch.no_grad():
        for pair in preprocessed_pairs:
            img_path = Path(pair["image"])
            image = torch.from_numpy(np.load(img_path)).float().unsqueeze(0)  # (1, C, H, W)

            logits = model(image)                          # (1, 1, H, W)
            prob = torch.sigmoid(logits).squeeze().numpy()  # (H, W)
            pred_mask = (prob >= threshold).astype(np.uint8)

            pred_path = out_path / ("pred_" + img_path.stem + ".npy")
            np.save(pred_path, pred_mask)
            saved.append(str(pred_path))
            logger.success(f"✅ {img_path.name} → {pred_mask.sum()} positive pixels")

    logger.info(f"📦 {len(saved)} predictions saved to {out_path}")
    return saved


# ── Pipeline ──────────────────────────────────────────────────────────────────


@pipeline(name="INFERENCE")
def inference_pipeline(
    model_path: str,
    preprocessed_pairs: list[dict],
    output_dir: str = "data/predictions",
    threshold: float = 0.5,
):
    """Run inference on preprocessed S2 patches."""
    run_inference(
        model_path=model_path,
        preprocessed_pairs=preprocessed_pairs,
        output_dir=output_dir,
        threshold=threshold,
    )