# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UNet-based semantic segmentation pipeline for Sentinel-2 satellite imagery. The project ingests geospatial vector data (GeoJSON/Shapefiles), acquires Sentinel-2 imagery via Google Earth Engine, and processes it for segmentation. Currently in early-stage development.

## Development Environment

- **Python 3.12** with virtual environment at `env/`
- Activate: `source env/bin/activate`
- No requirements.txt or pyproject.toml yet — dependencies are installed directly in the venv
- Environment variables loaded via `python-dotenv` (expects a `.env` file, e.g., for Earth Engine credentials)

## Architecture

- **`main.py`** — Entry point (not yet implemented)
- **`src/ingest_data.py`** — Data ingestion module: loads vector data via geopandas, interfaces with Google Earth Engine via `ee` and `geemap`
- **`data/`** — Directory for input/output data files

### Pipeline Orchestration

Uses **ZenML** (v0.94.0) for ML pipeline steps and orchestration. Import pattern: `from zenml import step, pipeline`.

### Key Libraries

| Domain | Libraries |
|--------|-----------|
| Geospatial vector | geopandas, shapely, pyproj |
| Raster I/O | rasterio, xarray |
| Earth Engine | earthengine-api, geemap |
| ML/Data | numpy, pandas |
| Logging | loguru |

## Code Style

- Black formatter (configured in IDE)
- Uses loguru for logging (not stdlib `logging`)
- Emoji markers in log messages (✅ success, ⚠️ warnings/errors)
