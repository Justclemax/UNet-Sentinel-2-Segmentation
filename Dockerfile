# ── Builder stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System dependencies required by GDAL / rasterio / geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Only the runtime system libs (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project source
COPY src/  src/
COPY run_pipeline.py .
COPY main.py .

# data/ is mounted at runtime — not baked into the image
# .env   is passed via --env-file at runtime

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "run_pipeline.py"]