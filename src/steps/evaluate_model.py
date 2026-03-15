from pathlib import Path

import numpy as np
import torch
from loguru import logger
from zenml import step, pipeline

from train_model import UNet, S2Dataset
from torch.utils.data import DataLoader

# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute IoU, Dice and pixel accuracy from binary arrays."""
    preds = preds.astype(bool)
    targets = targets.astype(bool)

    tp = (preds & targets).sum()
    fp = (preds & ~targets).sum()
    fn = (~preds & targets).sum()
    tn = (~preds & ~targets).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_acc),
    }


# ── ZenML step ────────────────────────────────────────────────────────────────


@step
def evaluate_model(
    model_path: str,
    preprocessed_pairs: list[dict],
    batch_size: int = 8,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate the trained UNet on the held-out 20% test split.

    Returns
    -------
    Dict with iou, dice, pixel_accuracy.
    """
    # Use the same 80/20 split as training — last 20% = test
    split = max(1, int(0.8 * len(preprocessed_pairs)))
    test_pairs = preprocessed_pairs[split:] or preprocessed_pairs

    checkpoint = torch.load(model_path, map_location="cpu")
    model = UNet(
        in_channels=checkpoint["in_channels"],
        base_features=checkpoint["base_features"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(S2Dataset(test_pairs), batch_size=batch_size)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images)                          # (B, 1, H, W)
            probs = torch.sigmoid(logits).squeeze(1)       # (B, H, W)
            preds = (probs >= threshold).numpy().astype(np.uint8)
            all_preds.append(preds)
            all_targets.append(masks.numpy().astype(np.uint8))

    preds_all = np.concatenate(all_preds, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)
    metrics = _compute_metrics(preds_all, targets_all)

    logger.success(
        f"✅ Evaluation ({len(test_pairs)} patches) — "
        f"IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
        f"PixAcc={metrics['pixel_accuracy']:.4f}"
    )
    return metrics


# ── Pipeline ──────────────────────────────────────────────────────────────────


@pipeline(name="EVALUATE_MODEL")
def evaluate_pipeline(
    model_path: str,
    preprocessed_pairs: list[dict],
    batch_size: int = 8,
):
    """Evaluate trained UNet on test split."""
    evaluate_model(
        model_path=model_path,
        preprocessed_pairs=preprocessed_pairs,
        batch_size=batch_size,
    )