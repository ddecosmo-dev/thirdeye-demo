#!/bin/bash
# Quick start script for local edge service testing
# This script sets up environment and runs the full test workflow

set -euo pipefail

# Configuration
EDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/tmp/edge_data}"
BLOB_PATH="${BLOB_PATH:-}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8081}"
PROCESSOR_PORT="${PROCESSOR_PORT:-8082}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Validate setup
print_header "Edge Service - Local Test Quick Start"

if [ -z "$BLOB_PATH" ]; then
    print_error "BLOB_PATH not set"
    echo "Usage: export BLOB_PATH=/path/to/student_mobilenet_v3.blob"
    echo "Then: $0"
    exit 1
fi

if [ ! -f "$BLOB_PATH" ]; then
    print_error "Blob file not found: $BLOB_PATH"
    exit 2
fi

print_success "Configuration"
echo "  EDGE_DIR: $EDGE_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  BLOB_PATH: $BLOB_PATH"
echo "  COORDINATOR_PORT: $COORDINATOR_PORT"
echo "  PROCESSOR_PORT: $PROCESSOR_PORT"

# Create venv if needed
if [ ! -d "$EDGE_DIR/.venv" ]; then
    print_header "Setting Up Python Environment"
    cd "$EDGE_DIR"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
    print_success "Virtual environment created and dependencies installed"
else
    source "$EDGE_DIR/.venv/bin/activate"
    print_success "Virtual environment activated"
fi

# Run smoke check
print_header "Running Smoke Check"
cd "$EDGE_DIR"
if bash scripts/smoke_check_local.sh; then
    print_success "Smoke check passed"
else
    print_error "Smoke check failed - review logs in /tmp/edge_smoke_logs"
    exit 3
fi

# Start services
print_header "Starting Services"
export OAK_CONNECTED=true
export BLOB_PATH="$BLOB_PATH"
export DATA_DIR="$DATA_DIR"
export COORDINATOR_PORT="$COORDINATOR_PORT"
export PROCESSOR_PORT="$PROCESSOR_PORT"

mkdir -p "$DATA_DIR"

# Kill any previous instances
pkill -f "python3 -m app.main" || true
sleep 1

# Start in background
nohup python3 -m app.main > /tmp/edge_service.log 2>&1 &
SERVICE_PID=$!
print_info "Service started with PID $SERVICE_PID"
print_info "Logs: /tmp/edge_service.log"

sleep 3

# Verify services are running
if ! kill -0 $SERVICE_PID 2>/dev/null; then
    print_error "Service died. Check /tmp/edge_service.log"
    cat /tmp/edge_service.log | tail -20
    exit 4
fi

print_success "Services running"

# Run end-to-end test
print_header "Running End-to-End Tests"

if python3 test_e2e.py "$COORDINATOR_PORT" "$PROCESSOR_PORT"; then
    print_success "All tests passed!"
    print_info "Results saved to: $DATA_DIR/runs/"
    
    # Show summary
    echo ""
    print_header "Test Results Summary"
    
    for run_dir in "$DATA_DIR"/runs/RUN_*/; do
        if [ -d "$run_dir" ]; then
            run_id=$(basename "$run_dir")
            bundle="$run_dir/bundle.zip"
            
            if [ -f "$bundle" ]; then
                size=$(du -h "$bundle" | cut -f1)
                image_count=$(unzip -l "$bundle" | grep -c "\.jpg$" || true)
                
                echo "  Run: $run_id"
                echo "    Archive: $size"
                echo "    Images: $image_count"
                echo "    Path: $bundle"
            fi
        fi
    done
    
    echo ""
    print_success "Local testing complete! Ready to deploy to Raspberry Pi"
    
else
    print_error "Some tests failed"
    kill $SERVICE_PID || true
    exit 5
fi

# Cleanup option
read -p "Stop services? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kill $SERVICE_PID || true
    print_success "Services stopped"
else
    print_info "Services still running (PID: $SERVICE_PID)"
    print_info "Stop manually: kill $SERVICE_PID"
fi
