# ThirdEye Cloud Service + Dashboard - Complete Demo Setup

## What's New

The ThirdEye demo has been reorganized to separate model inference from the GUI, enabling:

✅ **Independent Model Service**: Run inference separately from the dashboard  
✅ **Persistent Result Storage**: Store scores and embeddings in queryable JSON format  
✅ **Compare Across Runs**: Query different walks/hikes and compare performance  
✅ **Scalable Architecture**: Dashboard and cloud service can run on different machines  
✅ **Maintained User Experience**: Same web UI, but now powered by cloud service  
✅ **Runtime t-SNE**: Compute visualizations at runtime to demo parameter flexibility  
✅ **Containerized**: Easy deployment with Docker (CPU & GPU support)  

## System Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│                 │         │                      │
│   Dashboard     │────────>│  Cloud Service       │
│  (UI Layer)     │         │  (Inference Engine)  │
│    Port 8000    │<────────│    Port 8001         │
│                 │         │                      │
└─────────────────┘         └──────────────────────┘
                                    │
                                    │
                            ┌───────v────────┐
                            │   /data/runs/  │
                            │  (Results DB)  │
                            └────────────────┘
```

## Docker Quick Start (Recommended)

### Prerequisites
- Docker Engine 20.10+ ([install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 1.29+ (usually included with Docker Desktop)
- (Optional) NVIDIA Docker runtime for GPU support

### Option A: CPU Only (Easiest)

```bash
cd demo/cloud\ services/cloud-service

# Start cloud service on CPU (port 8001)
docker-compose up cloud-service-cpu

# In another terminal, start dashboard (port 8000)
cd ../../dashboard
docker-compose -f ../cloud\ services/cloud-service/docker-compose.yml up dashboard
```

Visit: `http://localhost:8000`

### Option B: With GPU Support

First, check if GPU is available:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime nvidia-smi
```

If that works, run:
```bash
cd demo/cloud\ services/cloud-service

# Start GPU version
docker-compose up cloud-service-gpu

# Start dashboard in another terminal
cd ../../dashboard
docker-compose -f ../cloud\ services/cloud-service/docker-compose.yml up dashboard
```

**GPU acceleration**: ~3-5x faster inference (100 images: 20-30s → 5-10s)

### Verify Services Are Running

```bash
# Check cloud service health
curl http://localhost:8001/health
# {"status":"ok"}

# Check dashboard health  
curl http://localhost:8000/health
# {"status":"ok","pipeline_status":"idle","cloud_service":"ok"}
```

### Stopping Containers

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears /data)
docker-compose down -v
```

## Non-Docker Setup (Original)

### Step 1: Prepare Environment

```bash
cd /home/devin_work/work/third-eye/thirdeye-demo

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Start Cloud Service (Terminal 1)

```bash
cd demo/cloud\ services/cloud-service
pip install -r requirements.txt

# Start on port 8001
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Step 3: Start Dashboard (Terminal 2)

```bash
cd demo/dashboard
pip install -r requirements.txt

# Start on port 8000
python -m uvicorn server_cloud:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Use Web UI

Open `http://localhost:8000` in your browser.

## Manual API Testing

### Upload Images to Cloud Service

```bash
# Create a test dataset
mkdir -p /tmp/test_images
cd /tmp/test_images
# Put some .jpg files here

# Create zip
zip test.zip *.jpg

# Upload to cloud (ingest)
curl -X POST http://localhost:8001/ingest \
  -F "file=@test.zip" \
  -F "run_id=test_run_001"

# Response: 
# {
#   "run_id": "test_run_001",
#   "status": "uploaded",
#   "file_count": 42,
#   "total_uncompressed_bytes": 15728640,
#   "checksum_sha256": "abc123def..."
# }
```

### Trigger Inference

```bash
curl -X POST http://localhost:8001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "test_run_001",
    "epsilon": 0.12,
    "min_cluster_size": 2,
    "ignore_object": false,
    "device": "cpu"
  }'

# Response:
# {
#   "run_id": "test_run_001",
#   "status": "completed",
#   "message": "Inference complete: 20 champions selected from 42 images",
#   "image_count": 42,
#   "champion_count": 20,
#   "results_path": "/runs/test_run_001/results.json"
# }
```

### Retrieve Results

```bash
# Get detailed results
curl http://localhost:8001/runs/test_run_001/results | python -m json.tool | less

# Get just champion count
curl http://localhost:8001/runs/test_run_001/results | python -c "import json, sys; data = json.load(sys.stdin); print(f\"Champions: {data['champion_count']} of {data['image_count']}\")"
```

## Data Storage & Querying

### Where Results Are Stored

Cloud service stores all data in `/data/runs/`:

```
/data/runs/
├── test_run_001/
│   ├── metadata.json       # Status, timestamps, params
│   ├── results.json        # Full inference results (scores, cluster IDs, champion flags)
│   ├── embeddings.npz      # 384-dim embeddings (efficient binary format) - NEW!
│   ├── images/             # Extracted images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── raw/
│       └── upload.zip      # Original uploaded zip
├── walk_20260415_morning/
│   └── ...
└── walk_20260415_afternoon/
    └── ...
```

### Docker: Data Persistence

By default, Docker volumes are created in `tmpfs` (RAM). To persist data:

**Option 1: Use Named Volume (Survives Container Restart)**
```bash
# In docker-compose.yml, keep:
volumes:
  thirdeye-data:
    driver: local
```

**Option 2: Bind Mount to Host Directory**
```bash
# In docker-compose.yml, replace thirdeye-data with:
services:
  cloud-service-cpu:
    volumes:
      - /path/to/local/data:/data  # Or ./data:/data for relative path
```

**Option 3: Check Current Data Location**
```bash
# View volume location
docker volume inspect thirdeye-data

# Access data from running container
docker cp cloudsvc:/data ./my_results
```

### Results Format (results.json)

Each image now gets stored with scores and cluster info (NO t-SNE):

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
      "is_champion": true,
      "tech_penalized": false,
      "rejection_reason": null
    },
    ... more images
  ]
}
```

**Important**: t-SNE coordinates are **NOT** stored. Instead, they're computed at runtime (next section).

### Runtime t-SNE: Demonstrating Parameter Flexibility

One of the powerful new features is computing t-SNE visualization at runtime from saved embeddings:

```bash
# Compute t-SNE with perplexity=30 (default)
curl -X POST http://localhost:8001/tsne \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "test_run_001",
    "perplexity": 30,
    "seed": 42
  }'

# Response:
{
  "run_id": "test_run_001",
  "status": "completed",
  "image_count": 150,
  "perplexity": 30,
  "tsne_coordinates": [
    {"index": 0, "filename": "image1.jpg", "x": 10.5, "y": -3.2},
    {"index": 1, "filename": "image2.jpg", "x": -5.1, "y": 8.7},
    ...
  ]
}
```

**Why this approach?**
- ✅ Embeddings are stored once (small, 384-dim vectors)
- ✅ t-SNE can be re-run with different perplexity values
- ✅ Demonstrates how visualization changes without re-inferencing
- ✅ Great for interactive demos: adjust perplexity, see results instantly
- ✅ Efficient: t-SNE is fast compared to model inference

## Deployment Scenarios

### Scenario 1: Local Demo (What we just did)
- Both services on same machine
- Use `localhost` for all URLs
- Perfect for demos and development

### Scenario 2: Edge Device + Cloud Server
- Edge device: Capture images, send zip to cloud
- Cloud server: Run models, store results
- Edge polls cloud for results

```bash
# On edge device
curl -X POST http://CLOUD_HOST:8001/ingest \
  -F "file=@hike_photos.zip" \
  -F "run_id=edge_device_001"
```

### Scenario 3: Multiple Cameras + Central Cloud
- Cameras send zips with timestamps
- Cloud indexes by camera ID and timestamp
- Web interface queries across all cameras

```
Camera A ──┐
Camera B ──┼─> Cloud Service ──> /data/runs/
Camera C ──┘
```

## Monitoring & Troubleshooting

### Check Cloud Service Health

```bash
curl -v http://localhost:8001/health
```

### View Available Runs

```bash
curl http://localhost:8001/runs | python -m json.tool
```

### Monitor Long-Running Inference

```bash
# In a separate terminal, watch the status
watch -n 2 'curl http://localhost:8001/runs/test_run_001 | python -c "import json,sys;data=json.load(sys.stdin);print(f\"Status: {data.get(\\\"status\\\",\\\"unknown\\\")}\")"'
```

### View Service Logs

```bash
# Cloud service logs will show inference progress
# Look for:
# - "Starting inference on X images"
# - "Stage X: ..."
# - "Inference complete"
```

### Check Data Disk Usage

```bash
du -sh /data/runs/
# Shows total size of all stored results
```

## Configuration

### Cloud Service

Environment variables (set before starting):

```bash
export DATA_DIR="/data"                        # Where to store results
export MAX_UPLOAD_BYTES="52428800"             # Max zip size (50MB)
export MAX_UNCOMPRESSED_BYTES="104857600"      # Max extracted size (100MB)
export MAX_FILES_PER_ZIP="500"                 # Max files in zip
```

### Dashboard

Environment variables (set before starting):

```bash
export CLOUD_SERVICE_URL="http://localhost:8001"  # Cloud service URL
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Connection refused" | Make sure cloud service is running on port 8001 |
| "No images found" | Check zip file format - must contain image files in root or subdirs |
| "Out of memory" | Reduce image count or limit resolution; use GPU if available |
| "Slow inference" | GPU is faster; reduce cluster size or increase epsilon |
| "Results not found" | Give inference time to complete; check status with `/runs/{run_id}` |

## Next Steps

1. **Integrate with edge devices**: Send zips from cameras
2. **Build query interface**: Search results by date, camera, scores
3. **Add real-time updates**: WebSocket for live progress
4. **Export pipeline**: Save champion images + metadata
5. **batch processing**: Queue multiple runs, process sequentially

## File Structure

```
thirdeye-demo/
├── CLOUD_SETUP_GUIDE.md          # This file
├── README.md                      # Original readme
├── demo/
│   ├── dashboard/
│   │   ├── server.py             # Local inference (original)
│   │   ├── server_cloud.py        # Cloud-connected (NEW)
│   │   ├── cloud_client.py        # Cloud service client (NEW)
│   │   ├── pipeline.py            # Score aggregation
│   │   ├── model_aesthetic.py     # DINOv2 model
│   │   ├── model_technical.py     # NIMA model
│   │   ├── model_object.py        # MaskFormer model
│   │   ├── index.html             # Web UI
│   │   └── requirements.txt       # Dependencies
│   ├── cloud services/
│   │   └── cloud-service/         # NEW cloud inference service
│   │       ├── app/
│   │       │   ├── main.py        # API endpoints (with /infer)
│   │       │   ├── inference/     # NEW inference module
│   │       │   │   ├── runner.py  # InferenceRunner class
│   │       │   │   ├── model_*.py # Model classes
│   │       │   │   └── __init__.py
│   │       │   ├── storage.py     # Results persistence (updated)
│   │       │   ├── ingest.py      # Zip handling
│   │       │   └── ...
│   │       └── requirements.txt   # Dependencies (updated)
│   └── edge-services/
└── model deployment/
```

---

**You're all set!** The cloud service is now running model inference independently, and the dashboard queries results in real-time. This enables:

- 🎬 **Demo mode**: Upload zips, see results streaming
- 📊 **Result storage**: All scores/embeddings permanently saved
- 🔍 **Querying**: Compare walks, hikes, time of day
- 🚀 **Scalability**: Add cameras, multiple inference servers
- ⚡ **GPU acceleration**: 3-5x faster with NVIDIA/Metal cards

## Performance Benchmarks

| Task | CPU | GPU (CUDA/Metal) |
|------|-----|-----------------|
| Model Load | 30-60s | 30-60s |
| Per Image Scoring | 100-200ms | 20-50ms |
| Clustering (100 images) | 50ms | 50ms |
| t-SNE (100 images) | 500ms | 100ms |
| **Total 100 images** | **20-30s** | **5-10s** |

**Tip**: For demos, keep dataset size ≤ 200 images on CPU, no limit on GPU.

## Docker Troubleshooting

### Container won't start
```bash
# Check logs
docker logs thirdeye-cloud-cpu

# Rebuild
docker-compose build --no-cache cloud-service-cpu
```

### GPU not detected
```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime nvidia-smi

# If not found, install nvidia-docker:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Data not persisting
```bash
# Check volume configuration
docker volume inspect thirdeye-data

# Access data
docker cp thirdeye-cloud-cpu:/data ./backup
```

## Query Code Examples

### Get Champions from Run
```python
import requests

run_id = "my_run"
results = requests.get(f"http://localhost:8001/runs/{run_id}/results").json()
champions = [img for img in results["results"] if img["is_champion"]]

for champ in champions:
    print(f"{champ['filename']}: {champ['aggregated_score']:.2f}")
```

### Compute t-SNE with Different Perplexity
```python
import requests

# Default perplexity=30
resp1 = requests.post("http://localhost:8001/tsne", json={
    "run_id": "my_run",
    "perplexity": 30
})

# Tighter clusters
resp2 = requests.post("http://localhost:8001/tsne", json={
    "run_id": "my_run",
    "perplexity": 50
})

coords1 = resp1.json()["tsne_coordinates"]
coords2 = resp2.json()["tsne_coordinates"]

# Both use same embeddings, just different visualization
```

### Compare Multiple Runs
```python
import requests
import numpy as np

runs = ["walk_morning", "walk_afternoon", "walk_evening"]
for run_id in runs:
    data = requests.get(f"http://localhost:8001/runs/{run_id}/results").json()
    scores = [img["aggregated_score"] for img in data["results"]]
    champs = data["champion_count"]
    
    print(f"{run_id}: avg={np.mean(scores):.2f}, champions={champs}")
```
