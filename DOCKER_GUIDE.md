# ThirdEye Docker Deployment Guide

## Quick Reference

### CPU Only (Always Works)
```bash
cd demo/cloud\ services/cloud-service
docker-compose up cloud-service-cpu
```

### With GPU (If Available)
```bash
cd demo/cloud\ services/cloud-service
docker-compose up cloud-service-gpu
```

### With Dashboard
```bash
docker-compose --profile with-dashboard up
```

---

## Architecture

### Single Machine (Local Demo)
```
Host Machine
├── Docker Network
│   ├── cloud-service-cpu:8001
│   ├── cloud-service-gpu:8001 (optional, different port in compose)
│   └── dashboard:8000
├── Docker Volume: /data
│   └── runs/
│       ├── run_1/
│       ├── run_2/
│       └── ...
```

### Distributed (Edge + Cloud)
```
Camera / Edge Device           Cloud Server (Docker)
├── Zip Data  ────────────────> cloud-service:8001
│                               ├── Run Inference
│                               └── Store Results (/data/runs)
│
Browser                        Dashboard Server
└── UI ──────────────────────> dashboard:8000
                               └── Query Results
```

---

## Data Persistence Options

### Option 1: In-Memory (Temporary, Fast)
**Default configuration** - data is lost when container stops

```yaml
volumes:
  thirdeye-data:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
```

**Good for**: Demos, testing, ephemeral workflows

**Start**: `docker-compose up`

### Option 2: Named Volume (Persistent, Portable)
Data survives container restarts, stored in Docker's storage location

```yaml
volumes:
  thirdeye-data:
    driver: local
```

**Good for**: Regular use, simple setup

**Data location**:
```bash
# Docker Desktop (Mac/Windows): ~/Library/Containers/com.docker.docker/Volumes
# Linux: /var/lib/docker/volumes/thirdeye-data/_data

# Access:
docker volume ls
docker volume inspect thirdeye-data
docker cp container:/data ./backup
```

### Option 3: Bind Mount (Explicit Directory)
Store data in a specific system directory

```yaml
services:
  cloud-service-cpu:
    volumes:
      - ./data:/data  # Relative path (./data on host)
      # OR absolute:
      - /home/user/thirdeye-results:/data
```

**Good for**: Archives, backups, shared storage

**Start**:
```bash
mkdir -p data  # Create directory first
docker-compose up
```

**Access**: Simple - data is just in `./data` directory

### Option 4: NFS/SMB (Network Storage)
Store on network-attached storage

```yaml
volumes:
  thirdeye-data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,vers=4,soft,timeo=180
      device: ":/export/thirdeye"
```

**Good for**: Multi-device setup, centralized results

---

## GPU Setup

### NVIDIA GPUs (Linux/Windows)

1. **Install nvidia-docker**:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Verify GPU access**:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime nvidia-smi
```

3. **Run GPU container**:
```bash
docker-compose up cloud-service-gpu
```

### Apple Silicon (Mac)

No special setup needed - PyTorch automatically uses Metal Performance Shaders.

Just run:
```bash
docker-compose up cloud-service-gpu
```

---

## Deployment Scenarios

### Scenario 1: Local Development
Perfect for demos and testing

```bash
# Setup
cd demo/cloud\ services/cloud-service
mkdir -p data  # For persistent results

# Start
docker-compose up cloud-service-cpu

# In another terminal:
docker-compose --profile with-dashboard up dashboard

# Browse: http://localhost:8000
```

**Cleanup**:
```bash
docker-compose down
# Data in ./data is preserved
```

### Scenario 2: Production on Single Server

```bash
# Edit docker-compose.yml:
# - Change cloud-service-gpu to use explicit port or external visibility
# - Set volumes to bind mount for persistent storage
# - Add environment variable overrides

docker-compose up -d  # Run in background

# Monitor
docker-compose logs -f cloud-service-cpu

# Backup data
docker cp thirdeye-cloud-cpu:/data ./backup-$(date +%s)
```

### Scenario 3: Edge Device + Cloud Results Server

**On edge device** (Raspberry Pi, etc.):
```bash
# Run lightweight upload agent (not containerized)
python3 << 'EOF'
import requests
import glob

cloud_url = "http://cloud-server:8001"

for zip_file in glob.glob("*.zip"):
    with open(zip_file, "rb") as f:
        requests.post(f"{cloud_url}/ingest", 
                     files={"file": f},
                     data={"run_id": f"edge_{zip_file}"})
    print(f"Uploaded {zip_file}")
EOF
```

**On cloud server**:
```bash
docker-compose up cloud-service-gpu
```

**Results accessible via**:
```bash
curl http://cloud-server:8001/runs
curl http://cloud-server:8001/runs/edge_video1/results
```

### Scenario 4: Multi-GPU Server

Edit `docker-compose.yml`:

```yaml
services:
  cloud-service-gpu-0:
    build:
      context: .
      dockerfile: Dockerfile.multi
      target: gpu
    environment:
      - CUDA_VISIBLE_DEVICES=0  # GPU 0
    ports:
      - "8001:8001"
    volumes:
      - thirdeye-data:/data

  cloud-service-gpu-1:
    build:
      context: .
      dockerfile: Dockerfile.multi
      target: gpu
    environment:
      - CUDA_VISIBLE_DEVICES=1  # GPU 1
    ports:
      - "8002:8001"
    volumes:
      - thirdeye-data:/data  # Shared storage
```

---

## Health Checks & Monitoring

### Verify Services
```bash
# Cloud service
curl http://localhost:8001/health

# Dashboard (if running)
curl http://localhost:8000/health

# Check container status
docker-compose ps

# View logs
docker-compose logs cloud-service-cpu
docker-compose logs cloud-service-gpu
docker-compose logs dashboard
```

### Monitor Resource Usage
```bash
# Real-time stats
docker stats

# Specific container
docker stats thirdeye-cloud-cpu --no-stream
```

### Check Data Size
```bash
# Inside container
docker exec thirdeye-cloud-cpu du -sh /data

# Or access volume directly
du -sh ~/.docker/volumes/thirdeye-data/_data
```

---

## Troubleshooting

### Container crashes on startup

```bash
# Check logs
docker logs thirdeye-cloud-cpu

# Common issues:
# 1. Port 8001 already in use
#    → Change port in docker-compose.yml
# 2. Not enough space for models
#    → Need ~10GB free space
# 3. Out of memory
#    → Limit dataset size or add swap
```

### GPU not detected in container

```bash
# Verify nvidia-docker works
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime nvidia-smi

# If failed, reinstall nvidia-container-toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

# Check docker-compose.yml GPU config
# The deploy.resources section must be present
```

### Volume permission issues

```bash
# If you get "Permission denied" accessing /data:

# Fix 1: Run as same user
docker-compose exec -u $(id -u):$(id -g) cloud-service-cpu ls /data

# Fix 2: Change volume permissions
docker exec thirdeye-cloud-cpu chmod 777 /data
```

### Out of disk space

```bash
# Check docker storage
docker system df

# Clean up
docker system prune -a  # Remove all unused images
docker volume prune     # Remove unused volumes

# Or remove specific volume
docker volume rm thirdeye-data
```

---

## Persistence Comparison

| Feature | tmpfs | Named Vol | Bind Mount | NFS |
|---------|-------|-----------|-----------|-----|
| Data survives restart | ❌ | ✅ | ✅ | ✅ |
| Access from host | ❌ | 🟡 Complex | ✅ | ✅ |
| Network access | ❌ | ❌ | 🟡 SSH only | ✅ |
| Performance | ⚡ Fast | 🟡 Good | 🟡 Good | 🐢 Network |
| Setup complexity | ✅ None | 🟡 Simple | ✅ Simple | 🔴 Moderate |

---

## Advanced: Custom Builds

### Build for specific architecture
```bash
# ARM64 (Raspberry Pi, Apple Silicon)
docker buildx build --platform linux/arm64 -t thirdeye-cloud:arm64 .

# x86-64 (standard)
docker buildx build --platform linux/amd64 -t thirdeye-cloud:x86 .
```

### Build with custom Python version
```dockerfile
# Edit Dockerfile.multi to use different base
FROM python:3.10-slim as base  # Change 3.11 to 3.10
```

### Build without GPU support
```bash
# Use cpu target only
docker build -f Dockerfile.multi --target cpu -t thirdeye:cpu .
```

---

## Production Checklist

- [ ] Persistent volume configured (not tmpfs)
- [ ] Resource limits set (`deployment.resources`)
- [ ] Healthchecks passing
- [ ] Backup strategy in place
- [ ] Monitoring/logging configured
- [ ] Firewall allows port 8001 (cloud) and 8000 (dashboard)
- [ ] GPU enabled if hardware available
- [ ] Data directory has sufficient space (>50GB recommended)
