from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio  # noqa: rasterio.open is a module-level alias, not a method
from rasterio.features import rasterize
from loguru import logger
from zenml import step, pipeline

# ── Constants ────────────────────────────────────────────────────────────────

S2_SCALE_FACTOR = 10_000.0
PATCH_SIZE = 256


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_and_normalize(tif_path: Path) -> tuple[np.ndarray, object, tuple]:
    """Read a GeoTIFF, normalize to [0, 1], fill NaN with 0.

    Returns (array (C, H, W) float32, rasterio CRS, (height, width)).
    """
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr /= S2_SCALE_FACTOR
    np.clip(arr, 0.0, 1.0, out=arr)
    np.nan_to_num(arr, nan=0.0, copy=False)
    return arr, transform, crs, shape


def _rasterize_mask(
    gdf: gpd.GeoDataFrame,
    transform,
    crs,
    shape: tuple,
) -> np.ndarray:
    """Burn vector features onto the patch grid. Returns (H, W) uint8 mask."""
    gdf_proj = gdf.to_crs(crs) if crs else gdf
    geometries = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    if not geometries:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(
        geometries,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def _crop_or_pad_2d(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or pad a (H, W) array to (size, size)."""
    h, w = arr.shape
    if h >= size and w >= size:
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        return arr[y0 : y0 + size, x0 : x0 + size]
    out = np.zeros((size, size), dtype=arr.dtype)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def _crop_or_pad_3d(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or pad a (C, H, W) array to (C, size, size)."""
    _, h, w = arr.shape
    if h >= size and w >= size:
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        return arr[:, y0 : y0 + size, x0 : x0 + size]
    out = np.zeros((arr.shape[0], size, size), dtype=arr.dtype)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    out[:, y0 : y0 + h, x0 : x0 + w] = arr
    return out


# ── ZenML step ───────────────────────────────────────────────────────────────


@step
def preprocess_scenes(
    scenes_dir: str,
    vector_path: str,
    output_dir: str = "data/preprocessed",
) -> list[dict]:
    """Normalize S2 patches and rasterize vector masks into 256×256 .npy pairs.

    Returns
    -------
    List of {"image": str, "mask": str} dicts — one per patch.
    """
    scenes_path = Path(scenes_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(scenes_path.glob("S2_*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"⚠️ No S2_*.tif files found in {scenes_path}")

    gdf = gpd.read_file(vector_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    logger.info(f"📂 {len(tif_files)} GeoTIFF(s) | {len(gdf)} vector features")

    pairs: list[dict] = []
    for tif in tif_files:
        arr, transform, crs, shape = _load_and_normalize(tif)
        arr = _crop_or_pad_3d(arr, PATCH_SIZE)

        mask = _rasterize_mask(gdf, transform, crs, shape)
        mask = _crop_or_pad_2d(mask, PATCH_SIZE)

        img_path = out_path / (tif.stem + ".npy")
        mask_path = out_path / ("mask_" + tif.stem + ".npy")

        np.save(img_path, arr)
        np.save(mask_path, mask)

        pairs.append({"image": str(img_path), "mask": str(mask_path)})
        logger.success(f"✅ {tif.stem} → image {arr.shape}, mask {mask.shape} (pos={mask.sum()}px)")

    logger.info(f"📊 {len(pairs)} pairs saved to {out_path}")
    return pairs


# ── Pipeline ─────────────────────────────────────────────────────────────────


@pipeline(name="PREPROCESS_DATA")
def preprocess_pipeline(
    scenes_dir: str = "data/scenes",
    vector_path: str = "data/training_data.geojson",
    output_dir: str = "data/preprocessed",
):
    """Normalize S2 scenes and rasterize masks → 256×256 .npy pairs."""
    preprocess_scenes(
        scenes_dir=scenes_dir,
        vector_path=vector_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    preprocess_pipeline(
        scenes_dir="../../data/scenes",
        vector_path="../../data/training_data.geojson",
        output_dir="../../data/preprocessed",
    )