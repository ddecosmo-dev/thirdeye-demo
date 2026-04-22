# ThirdEye Cloud Inference Setup Guide

## Architecture Overview

The ThirdEye demo now supports two deployment modes:

### Mode 1: Local Inference (Original)
- Run everything on a single machine
- Dashboard runs models locally
- Simpler setup, faster for small datasets
- Use: `python -m uvicorn server:app --port 8000`

### Mode 2: Cloud-Connected (NEW - Recommended for Demo)
- Dashboard handles UI and zip uploads
- Cloud service runs model inference
- Scalable, separates concerns
- Use: `python -m uvicorn server_cloud:app --port 8000`

## Quick Start (Cloud Mode)

### 1. Start the Cloud Service

The cloud service handles all model inference and maintains a persistent datastore of results.

```bash
cd demo/cloud\ services/cloud-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The cloud service will be available at `http://localhost:8001`

### 2. Start the Dashboard

In a new terminal, start the dashboard (cloud-connected version):

```bash
cd demo/dashboard
pip install -r requirements.txt
export CLOUD_SERVICE_URL=http://localhost:8001  # or http://cloud-host:8001 if remote
python -m uvicorn server_cloud:app --host 0.0.0.0 --port 8000
```

The dashboard will be available at `http://localhost:8000`

### 3. Use the Dashboard

1. Navigate to `http://localhost:8000` in your browser
2. Upload a zip file containing images
3. Set clustering parameters (epsilon, min_cluster_size, etc.)
4. Click "Run Pipeline"
5. View results in real-time as they stream from the cloud service

## Data Structure

### Zip File Format (Input)

```
dataset.zip
├── image1.jpg
├── image2.jpg
├── image3.png
├── (optional) config.json  # Custom metadata
└── ... more images
```

### Results Storage (Cloud Service)

The cloud service stores all results persistently:

```
/data/runs/
├── <run_id_1>/
│   ├── metadata.json           # Cycle metadata and status
│   ├── results.json            # Full inference results
│   ├── images/                 # Extracted images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── raw/
│       └── upload.zip          # Original upload
├── <run_id_2>/
│   └── ...
```

### Results Format (results.json)

Each run produces a `results.json` containing:

```json
{
  "run_id": "run_20260416_abc123",
  "image_count": 150,
  "champion_count": 20,
  "inference_params": {
    "epsilon": 0.12,
    "min_cluster_size": 2,
    "ignore_object": false,
    "device": "cpu"
  },
  "results": [
    {
      "index": 0,
      "filename": "image1.jpg",
      "scores": {
        "technical": 65.5,
        "aesthetic": 7.2,
        "object": 6.8
      },
      "normalized_scores": {
        "tech_norm": 6.55,
        "aes_norm": 7.2,
        "obj_norm": 6.8
      },
      "aggregated_score": 7.0,
      "embedding": [0.123, -0.456, ... 384 values total],
      "cluster_id": 5,
      "tsne": {"x": 10.5, "y": -3.2},
      "is_champion": true,
      "tech_penalized": false,
      "rejection_reason": null
    },
    ... more images
  ]
}
```

## Cloud Service API Reference

### Health Check
```bash
GET http://localhost:8001/health
```

### Ingest Images
```bash
curl -X POST http://localhost:8001/ingest \
  -F "file=@dataset.zip" \
  -F "run_id=my_run" \
  -F "metadata_json={\"walk_id\": \"trail_001\"}"
```

### Run Inference
```bash
curl -X POST http://localhost:8001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "my_run",
    "epsilon": 0.12,
    "min_cluster_size": 2,
    "ignore_object": false,
    "device": "cpu"
  }'
```

### Get Run Status
```bash
GET http://localhost:8001/runs/my_run
```

### Get Results
```bash
GET http://localhost:8001/runs/my_run/results
```

## Querying Results

Once inference completes, you can:

1. **Query by run_id**: Retrieve all results for a specific walk/hike
```bash
curl http://localhost:8001/runs/my_run/results | python -m json.tool | grep "is_champion"
```

2. **Filter champions**: Get only selected images
```python
import requests
results = requests.get("http://localhost:8001/runs/my_run/results").json()
champions = [img for img in results["results"] if img["is_champion"]]
```

3. **Compare scores across runs**: Store and compare scores from different dates/conditions
```python
run1 = requests.get("http://localhost:8001/runs/walk_20260401/results").json()
run2 = requests.get("http://localhost:8001/runs/walk_20260415/results").json()
# Compare average scores, cluster distributions, etc.
```

## Configuration Options

### Environment Variables
- `CLOUD_SERVICE_URL`: Dashboard's cloud service URL (default: `http://localhost:8001`)
- `DATA_DIR`: Cloud service data directory (default: `/data`)
- `MAX_UPLOAD_BYTES`: Max zip file size (default: 50MB)
- `MAX_UNCOMPRESSED_BYTES`: Max uncompressed size (default: 100MB)

### Inference Parameters
- `epsilon`: HDBSCAN clustering epsilon (0.0-1.0, default 0.12)
  - Lower = more clusters, higher = fewer clusters
- `min_cluster_size`: Minimum images per cluster (2-100, default 2)
- `device`: GPU/CPU (`cpu`, `cuda`, `mps`, default `cpu`)

## Troubleshooting

### Cloud service is unavailable
```bash
# Check health
curl http://localhost:8001/health

# View logs
docker logs thirdeye-cloud-service  # if containerized
```

### Inference is slow
- Check device: GPU (cuda/mps) is much faster than CPU
- Reduce number of images in zip
- Increase timeout in dashboard (`wait_for_results` max_wait_seconds)

### Results not found
```bash
# List all runs
curl http://localhost:8001/runs

# Check specific run status
curl http://localhost:8001/runs/<run_id>
```

## Performance Notes

- **Model Loading**: ~30-60 seconds (first time, then cached)
- **Image Scoring**: ~100-200ms per image on CPU, ~20-50ms on GPU
- **Clustering**: ~10-50ms depending on image count
- **t-SNE**: ~100-500ms for typical dataset (100-500 images)
- **Overall**: 100 images ≈ 20-30 seconds on CPU, 5-10 seconds on GPU

## Next Steps

- **Multiple edge devices**: Each sends zips to cloud service, stores results indexed by device/walk
- **Database integration**: Replace filesystem with PostgreSQL/MongoDB for easier querying
- **Real-time streaming**: Stream MJPEG from cameras, run inference continuously
- **Web interface**: Build richer UI for exploring results across multiple runs
