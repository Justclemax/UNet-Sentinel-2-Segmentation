import os
from pathlib import Path

import ee
import geemap
import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger
from zenml import step, pipeline

load_dotenv()

# ── Sentinel-2 band configuration ───────────────────────────────────────────
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLOUD_THRESHOLD = 60


# ── Earth Engine session ─────────────────────────────────────────────────────


class EarthEngineSession:
    """Manages Earth Engine authentication and initialization."""

    _initialized = False

    @classmethod
    def init(cls) -> None:
        if cls._initialized:
            return
        project = os.getenv("EE_PROJECT")
        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project)
        cls._initialized = True
        logger.success("✅ Earth Engine initialized")


# ── Vector data loader ──────────────────────────────────────────────────────


class VectorLoader:
    """Loads and prepares vector data (GeoJSON / Shapefile) for Earth Engine."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.gdf: gpd.GeoDataFrame | None = None

    def load(self) -> gpd.GeoDataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"⚠️ File not found: {self.path}")

        self.gdf = gpd.read_file(self.path)
        if self.gdf.empty:
            raise ValueError(f"⚠️ The file {self.path} is empty or invalid.")

        if self.gdf.crs and self.gdf.crs.to_epsg() != 4326:
            self.gdf = self.gdf.to_crs(epsg=4326)
            logger.info("Reprojected to EPSG:4326")

        logger.success(f"✅ Loaded {len(self.gdf)} features from {self.path.name}")
        logger.info(f"Columns: {list(self.gdf.columns)} | CRS: {self.gdf.crs}")
        return self.gdf

    def to_ee_geometry(self) -> ee.Geometry:
        """Total bounds as EE geometry (for filtering collections)."""
        if self.gdf is None:
            raise RuntimeError("⚠️ Call load() before to_ee_geometry()")
        bounds = self.gdf.total_bounds
        return ee.Geometry.Rectangle(bounds.tolist())

    def feature_patches(self, buffer_m: float = 1280) -> list[dict]:
        """Return a list of {index, geometry} dicts — one small patch per feature.

        Each patch is the feature centroid buffered by `buffer_m` meters,
        giving a square region suitable for UNet input (e.g. 256×256 px at 10m).
        """
        if self.gdf is None:
            raise RuntimeError("⚠️ Call load() before feature_patches()")
        patches = []
        for idx, row in self.gdf.iterrows():
            centroid = row.geometry.centroid
            # Buffer in degrees (approximate: 1° ≈ 111km at equator)
            buf_deg = buffer_m / 111_000
            bbox = [
                centroid.x - buf_deg,
                centroid.y - buf_deg,
                centroid.x + buf_deg,
                centroid.y + buf_deg,
            ]
            patches.append(
                {
                    "index": int(idx),
                    "name": row.get("name", ""),
                    "bbox": bbox,
                }
            )
        logger.info(f"📐 Created {len(patches)} patches ({buffer_m}m buffer)")
        return patches


# ── Sentinel-2 temporal fetcher ──────────────────────────────────────────────


class Sentinel2Fetcher:
    """Fetches individual Sentinel-2 scenes — temporal, no composite."""

    def __init__(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
        max_cloud_cover: int = CLOUD_THRESHOLD,
        bands: list[str] | None = None,
    ):
        self.geometry = geometry
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud_cover = max_cloud_cover
        self.bands = bands or S2_BANDS
        self._collection: ee.ImageCollection | None = None

    def fetch_collection(self) -> ee.ImageCollection:
        self._collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(self.geometry)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_cover))
            .select(self.bands)
        )
        count = self._collection.size().getInfo()
        logger.info(
            f"🛰️  Found {count} Sentinel-2 scenes "
            f"({self.start_date} → {self.end_date})"
        )
        if count == 0:
            raise RuntimeError(
                "⚠️ No scenes found — try relaxing date range or cloud threshold"
            )
        return self._collection

    def list_scenes(self) -> list[dict]:
        """Return metadata (id, date, cloud %) for each scene."""
        if self._collection is None:
            self.fetch_collection()
        info_list = self._collection.toList(self._collection.size()).getInfo()
        scenes = []
        for img_info in info_list:
            props = img_info["properties"]
            scenes.append(
                {
                    "id": img_info["id"],
                    "date": props.get("system:index", ""),
                    "cloud_cover": props.get("CLOUDY_PIXEL_PERCENTAGE"),
                }
            )
        logger.info(f"📋 Listed {len(scenes)} scenes")
        return scenes

    def get_scene(self, scene_id: str) -> ee.Image:
        """Get a single scene by its Earth Engine ID."""
        return ee.Image(scene_id).select(self.bands)


# ── GeoTIFF exporter ─────────────────────────────────────────────────────────


class GeoTiffExporter:
    """Exports Earth Engine images to GeoTIFF files."""

    def __init__(self, output_dir: str | Path, scale: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale

    def export_scene(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        filename: str,
    ) -> Path:
        output_path = self.output_dir / filename
        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=self.scale,
            region=geometry,
            file_per_band=False,
        )
        logger.success(f"✅ Exported → {output_path}")
        return output_path

    def export_all(
        self,
        fetcher: Sentinel2Fetcher,
        geometry: ee.Geometry,
    ) -> list[Path]:
        """Export each scene as a separate GeoTIFF (temporal stack)."""
        scenes = fetcher.list_scenes()
        paths = []
        for scene_meta in scenes:
            image = fetcher.get_scene(scene_meta["id"])
            safe_name = scene_meta["id"].replace("/", "_")
            path = self.export_scene(image, geometry, f"{safe_name}.tif")
            paths.append(path)
        logger.success(f"✅ Exported {len(paths)} scenes to {self.output_dir}")
        return paths


# ── ZenML steps ──────────────────────────────────────────────────────────────


@step
def load_and_validate(vector_path: str) -> str:
    """Load vector data to validate it, return the path for downstream steps."""
    loader = VectorLoader(vector_path)
    loader.load()
    return vector_path


@step
def fetch_sentinel2_scenes(
    vector_path: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """List individual Sentinel-2 scenes (temporal, no composite)."""
    EarthEngineSession.init()
    loader = VectorLoader(vector_path)
    loader.load()
    geometry = loader.to_ee_geometry()

    fetcher = Sentinel2Fetcher(geometry, start_date, end_date)
    return fetcher.list_scenes()


@step
def export_scenes(
    vector_path: str,
    scenes: list[dict],
    output_dir: str = "data/scenes",
) -> str:
    """Export per-feature patches for each scene as individual GeoTIFFs."""
    EarthEngineSession.init()
    loader = VectorLoader(vector_path)
    loader.load()
    patches = loader.feature_patches(buffer_m=1280)

    fetcher = Sentinel2Fetcher(loader.to_ee_geometry(), "", "")
    exporter = GeoTiffExporter(output_dir)

    for scene_meta in scenes:
        image = fetcher.get_scene(scene_meta["id"])
        # Extract short date from scene ID (e.g. 20250101T...)
        scene_date = scene_meta["id"].split("/")[-1][:8]

        # Format date as YYYY-MM-DD
        date_str = f"{scene_date[:4]}-{scene_date[4:6]}-{scene_date[6:8]}"

        for patch in patches:
            region = ee.Geometry.Rectangle(patch["bbox"])
            label = patch["name"] or f"{patch['index']:04d}"
            filename = f"S2_{label}_{patch['index']:04d}_{date_str}.tif"
            exporter.export_scene(image, region, filename)

    return output_dir


# ── Pipeline ─────────────────────────────────────────────────────────────────


@pipeline(name="INGEST_AND_DOWLOAD_DATA")
def ingest_pipeline(
    vector_path: str,
    start_date: str,
    end_date: str,
    output_dir: str = "../data/",
):
    """Temporal ingestion: validate vector → list S2 scenes → export each as GeoTIFF."""
    validated_path = load_and_validate(vector_path)
    scenes = fetch_sentinel2_scenes(validated_path, start_date, end_date)
    export_scenes(validated_path, scenes, output_dir)

if __name__ == "__main__":
    # Short date range for testing (fewer scenes)
    ingest_pipeline(
        vector_path="../../data/training_data.geojson",
        start_date="2025-01-01",
        end_date="2025-01-15",
    )
