# ThirdEye Cloud Service Documentation

## Overview

The Cloud Service is a FastAPI backend that runs ML inference on image datasets using three pre-loaded neural network models. It handles:

- **Image ingestion** (via zip upload)
- **Technical scoring** (sharpness, exposure, composition)
- **Aesthetic scoring** (visual appeal)
- **Object detection** (identifying subjects in frame)
- **Clustering** (grouping similar images via HDBSCAN)
- **Champion selection** (finding best representatives)
- **Real-time progress tracking** (clients can poll for inference status)

## Architecture

### Services

| Service | Port | Purpose | Tech |
|---------|------|---------|------|
| **Cloud Service** | 8001 | ML inference, image processing, results management | FastAPI, PyTorch, CUDA |
| **Dashboard** | 8000 | Web UI, state management, orchestration | FastAPI, asyncio |
| **Frontend** | 8000 | User interface | HTML/CSS/JavaScript, Plotly |

### Models (Pre-loaded at Startup)

All models are loaded once on startup and cached in GPU memory:

1. **Technical Scorer** (`TechnicalScorer`)
   - Measures: sharpness, exposure, composition
   - Output: 0-100 score

2. **Aesthetic Scorer** (`AestheticScorer`) 
   - Measures: visual appeal via DINOv2 + NIMA
   - Output: 0-100 score + 384-dim embedding

3. **Object Scorer** (`ObjectScorer`)
   - Measures: object presence/quality via MaskFormer
   - Output: 0-100 score

## Directory Structure

```
cloud services/cloud-service/
├── app/
│   ├── main.py                 # FastAPI app, endpoints
│   ├── models.py               # Pydantic request/response models
│   ├── config.py               # Settings, constants
│   ├── storage.py              # Metadata/results file I/O
│   ├── cycles.py               # Cycle management (for edge device)
│   ├── ingest.py               # Zip file extraction
│   ├── inference/
│   │   ├── runner.py           # InferenceRunner class, score_images()
│   │   └── models/
│   │       ├── technical.py    # TechnicalScorer
│   │       ├── aesthetic.py    # AestheticScorer  
│   │       └── object.py       # ObjectScorer
│   └── utils.py                # Helpers
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## Key Endpoints

### Inference

**POST /infer** (202 Accepted)
```json
Request: {
  "run_id": "IMG_20260328_171141785_jpg",
  "epsilon": 0.12,
  "min_cluster_size": 2,
  "ignore_object": false
}

Response: {
  "status": "accepted",
  "message": "Inference started for 648 images...",
  "image_count": 648
}
```
- Returns **immediately** (202 status)
- Inference runs in background
- Poll `/progress/{run_id}` to track

**GET /progress/{run_id}**
```json
Response: {
  "stage": "scoring",
  "images_done": 45,
  "images_total": 648,
  "percent_complete": 7
}
```
- Real-time inference progress
- Updated every image scored
- Stages: pending → scoring → converting → saving → completed

**GET /runs/{run_id}/results**
```json
Response: {
  "run_id": "IMG_20260328_171141785_jpg",
  "image_count": 648,
  "champion_count": 20,
  "results": [
    {
      "index": 0,
      "filename": "IMG_00001.jpg",
      "scores": {
        "technical": 72.5,
        "aesthetic": 81.3,
        "object": 88.1
      },
      "normalized_scores": {
        "tech_norm": 0.725,
        "aes_norm": 0.813,
        "obj_norm": 0.881
      },
      "aggregated_score": 0.791,
      "cluster_id": 0,
      "is_champion": true,
      "embedding": [float, ...] // 384-dim vector
    },
    ...
  ]
}
```

### Data Management

**POST /ingest** 
- Upload zip with images to process
- Returns metadata about ingested files

**GET /runs**
- List all available runs

**POST /load-run/{run_id}**
- Load previous results from disk

**GET /images/{filename}**
- Serve image files (JPEG)

### t-SNE Visualization

**POST /tsne**
```json
Request: {
  "run_id": "IMG_20260328_171141785_jpg",
  "perplexity": 30
}

Response: {
  "run_id": "IMG_20260328_171141785_jpg",
  "tsne_coordinates": [
    {"index": 0, "x": 10.5, "y": -5.2, "cluster_id": 0, "is_champion": true},
    ...
  ]
}
```
- Computed on-demand from stored embeddings
- Allows testing different perplexity values without re-running inference

## Data Flow

### Inference Pipeline

```
1. POST /infer
   └─> Create background task
   └─> Return 202 Accepted immediately
   
2. Background Task: _run_inference_background()
   └─> Stage 1: Score images
       ├─ Load image
       ├─ Run 3 models (technical, aesthetic, object)
       ├─ Normalize scores
       ├─ Update _RUN_PROGRESS[run_id]
       └─ Repeat for all images
       
   └─> Stage 2: Aggregate scores
       ├─ Combine model outputs
       ├─ Apply ignore_object logic
       └─ Apply tech_floor penalty
       
   └─> Stage 3: Cluster embeddings
       ├─ Run HDBSCAN
       ├─ Identify noise points
       └─ Assign cluster IDs
       
   └─> Stage 4: Select champions
       ├─ Pick top image per cluster
       ├─ Assign rejection reasons
       └─ Mark final selections
       
   └─> Stage 5: Save results
       ├─> Convert to JSON
       ├─> Save results.json
       ├─> Save embeddings separately
       └─> Update metadata

3. Dashboard polls GET /progress/{run_id}
   └─> Gets updates every 1 second
   └─> Displays progress to user

4. When complete, GET /runs/{run_id}/results
   └─> Returns full results with all scores
```

### Score Calculation

```javascript
// Raw scores (0-100)
tech_score = technical_model(image)
aes_score = aesthetic_model(image) 
obj_score = object_model(image)

// Normalize to 0-1
tech_norm = tech_score / 100
aes_norm = aes_score / 100
obj_norm = obj_score / 100

// Aggregate (configurable weights)
if ignore_object:
  aggregated = aes_norm
else:
  aggregated = 0.6 * aes_norm + 0.4 * obj_norm

// Tech floor penalty
if tech_norm < 3.0:
  aggregated *= 0.5  // Heavy penalty for blurry/bad images

// Clustering uses aggregated score + embeddings
```

## Configuration

### Device Selection

- **Startup**: Auto-detects CUDA GPU
- **Fallback**: Uses CPU if GPU unavailable (slow)
- **Current**: RTX PRO 2000 Blackwell (sm_120, CUDA 13.0)
- **PyTorch**: 2.11.0+cu130 (required for Blackwell support)

### Inference Parameters

- **epsilon**: HDBSCAN epsilon (0.1-0.2 typical)
- **min_cluster_size**: Minimum cluster size (2-5 typical)
- **ignore_object**: Skip object score in aggregation

### Performance

| Operation | Time (GPU) | Time (CPU) |
|-----------|-----------|-----------|
| Load models | 5-10s | N/A |
| Score 1 image | ~200ms | ~2s |
| 648 images | ~200s | ~2000s |
| Clustering | ~5s | ~5s |
| Total | ~4 min | ~40 min |

## Running the Service

### Prerequisites

```bash
cd /home/devin_work/work/third-eye/thirdeye-demo

# Activate venv
source venv/bin/activate

# Install dependencies (if needed)
pip install -r demo/cloud\ services/cloud-service/requirements.txt
```

### Start Service

```bash
cd demo/cloud\ services/cloud-service
DATA_DIR=../../data \
  /home/devin_work/work/third-eye/thirdeye-demo/venv/bin/python3 \
  -m uvicorn app.main:app --port 8001
```

### Expected Output

```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete
WARNING: LOADING ALL MODELS AT STARTUP (Device: CUDA)
WARNING: 🎮 GPU detected: NVIDIA RTX PRO 2000
WARNING: ALL MODELS LOADED SUCCESSFULLY AT STARTUP
```

## Testing the Service

### Health Check

```bash
curl http://localhost:8001/health
```

### Create Dataset

```bash
# Prepare images in a folder
mkdir -p /tmp/test_images
cp /path/to/images/* /tmp/test_images/

# Create zip
cd /tmp
zip -r test_images.zip test_images/
```

### Upload & Infer

```bash
# Get run_id from folder name
RUN_ID="test_images"

# Upload
curl -X POST http://localhost:8001/ingest \
  -F "file=@test_images.zip" \
  -F "run_id=$RUN_ID"

# Start inference
curl -X POST http://localhost:8001/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"run_id\": \"$RUN_ID\",
    \"epsilon\": 0.12,
    \"min_cluster_size\": 2,
    \"ignore_object\": false
  }"

# Poll progress (in loop)
while true; do
  curl http://localhost:8001/progress/$RUN_ID | jq .
  sleep 1
done

# Get results when done
curl http://localhost:8001/runs/$RUN_ID/results | jq .
```

## Troubleshooting

### Service won't start

**Error**: `RuntimeError: CUDA out of memory`
- Models too large for GPU
- Another process using GPU memory
- Solution: `pkill -9 python` to clear processes

**Error**: `ModuleNotFoundError: No module named 'torch'`
- Virtual environment not activated
- PyTorch not installed
- Solution: `source venv/bin/activate` and reinstall requirements

### Inference fails

**Error**: `Device cuda not found`
- GPU not available
- Falls back to CPU automatically
- Performance will be much slower

**Error**: `Images directory does not exist`
- Run was not ingested properly
- Check `/home/devin_work/work/third-eye/thirdeye-demo/demo/data/runs/{run_id}`

### Progress not updating

**Check**: 
1. Is background task running? (check logs for "Starting inference")
2. Is `/progress/{run_id}` being called? (check dashboard logs)
3. Is `_RUN_PROGRESS[run_id]` being updated? (check cloud service logs)

## Development Notes

### Adding New Scoring Models

1. Create model class in `app/inference/models/`
2. Implement `score(image, device)` → float
3. Add to `load_models(device)` in `runner.py`
4. Update score calculation in `aggregate_scores()`
5. Update results JSON schema

### Modifying Clustering

1. Edit `run_clustering()` in `runner.py`
2. Update cluster parameters in request model
3. Update dashboard sliders if needed

### Changing Progress Tracking

1. Update `_RUN_PROGRESS[run_id]` dict structure
2. Update `/progress/{run_id}` response model
3. Update dashboard polling logic
4. Update frontend display

## Related Files

- **Dashboard orchestration**: `/demo/dashboard/server_cloud.py`
- **Frontend UI**: `/demo/dashboard/index.html`
- **Cloud client**: `/demo/dashboard/cloud_client.py`
- **Edge device integration**: `app/cycles.py` (not used in current demo)
