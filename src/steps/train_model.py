import sys
from pathlib import Path

# Make src/models importable
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from zenml import step, pipeline

from unet import UNet  # noqa: E402  (after sys.path patch)


# ── Device ───────────────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────


class S2Dataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        image = torch.from_numpy(np.load(pair["image"])).float()   # (C, H, W)
        mask = torch.from_numpy(np.load(pair["mask"])).float()     # (H, W)
        return image, mask


# ── Loss ──────────────────────────────────────────────────────────────────────


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits).view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs * targets_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets_flat.sum() + smooth)


# ── ZenML step ────────────────────────────────────────────────────────────────


@step
def train_model(
    preprocessed_pairs: list[dict],
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    in_channels: int = 10,
    base_features: int = 64,
    output_dir: str = "data/models",
) -> str:
    """Train UNet on (image, mask) pairs. Returns path to best model checkpoint."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    logger.info(f"🖥️  Training on {device}")

    # 80/20 train/val split
    split = max(1, int(0.8 * len(preprocessed_pairs)))
    train_pairs = preprocessed_pairs[:split]
    val_pairs = preprocessed_pairs[split:] or preprocessed_pairs  # fallback if tiny dataset

    train_loader = DataLoader(S2Dataset(train_pairs), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(S2Dataset(val_pairs), batch_size=batch_size)

    model = UNet(in_channels=in_channels, base_features=base_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    model_path = str(out_path / "unet_best.pt")

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)              # (B, 1, H, W)
            target = masks.unsqueeze(1)         # (B, 1, H, W)

            loss = 0.5 * bce_loss(logits, target) + 0.5 * _dice_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                target = masks.unsqueeze(1)
                val_loss += (0.5 * bce_loss(logits, target) + 0.5 * _dice_loss(logits, target)).item()
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch:03d}/{epochs} — train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "in_channels": in_channels,
                    "base_features": base_features,
                },
                model_path,
            )
            logger.success(f"✅ Best model saved (val_loss={val_loss:.4f})")

    logger.success(f"✅ Training done — best val_loss={best_val_loss:.4f} → {model_path}")
    return model_path


# ── Pipeline ──────────────────────────────────────────────────────────────────


@pipeline(name="TRAIN_MODEL")
def train_pipeline(
    preprocessed_pairs: list[dict],
    epochs: int = 50,
    batch_size: int = 8,
    output_dir: str = "data/models",
):
    """Train UNet on preprocessed S2 patches."""
    train_model(
        preprocessed_pairs=preprocessed_pairs,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )