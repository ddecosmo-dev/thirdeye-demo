#!/usr/bin/env python3
"""
End-to-end test for edge service pipeline.
Tests: cycle start/stop, image collection, metadata capture, and archive creation.
"""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


class EdgeServiceTester:
    def __init__(self, coord_port: int = 8081, proc_port: int = 8082):
        self.coord_port = coord_port
        self.proc_port = proc_port
        self.coord_url = f"http://localhost:{coord_port}"
        self.proc_url = f"http://localhost:{proc_port}"
        self.run_id = None

    def curl(self, method: str, url: str, json_data: dict = None) -> dict:
        """Execute curl request and return parsed JSON response."""
        cmd = ["curl", "-s", "-X", method, url]
        if json_data:
            cmd.extend(["-H", "Content-Type: application/json"])
            cmd.extend(["-d", json.dumps(json_data)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not result.stdout.strip():
            print(f"  ✗ No response from {url}")
            return {}
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"  ✗ Invalid JSON response: {result.stdout[:100]}")
            return {}

    def check_health(self) -> bool:
        """Check if coordinator and processor are ready."""
        print("\n[TEST 1] Service Health Check")
        print(f"  Testing coordinator: {self.coord_url}/health")
        coord = self.curl("GET", f"{self.coord_url}/health")
        if coord.get("status") != "ok":
            print("  ✗ Coordinator health check failed")
            return False
        print("  ✓ Coordinator OK")

        print(f"  Testing processor: {self.proc_url}/health")
        proc = self.curl("GET", f"{self.proc_url}/health")
        if proc.get("status") != "ok":
            print("  ✗ Processor health check failed")
            return False
        print("  ✓ Processor OK")
        return True

    def start_cycle(self, duration_seconds: int = 20, label: str = "test") -> bool:
        """Start a new capture cycle."""
        print(f"\n[TEST 2] Start Cycle ({duration_seconds}s, label={label})")
        response = self.curl(
            "POST",
            f"{self.coord_url}/cycle/start",
            {"label": label, "duration_seconds": duration_seconds}
        )

        if response.get("status") != "running":
            print(f"  ✗ Failed to start cycle: {response}")
            return False

        self.run_id = response.get("run_id")
        print(f"  ✓ Cycle started: {self.run_id}")
        return True

    def monitor_cycle(self, sample_interval: int = 2) -> bool:
        """Monitor cycle progress."""
        print(f"\n[TEST 3] Monitor Cycle (polling every {sample_interval}s)")
        start_time = time.time()
        
        while time.time() - start_time < 25:  # Monitor for up to 25s
            coord_status = self.curl("GET", f"{self.coord_url}/status")
            proc_status = self.curl("GET", f"{self.proc_url}/status")

            coord_running = coord_status.get("status") == "running"
            proc_running = proc_status.get("status") == "running"
            
            elapsed = int(time.time() - start_time)
            print(f"  [{elapsed:2d}s] Coord: {coord_status.get('frames_sent', 0)} frames | "
                  f"Proc: {proc_status.get('image_count', 0)} images")

            if not coord_running and not proc_running:
                print(f"  ✓ Cycle completed after {elapsed}s")
                return True

            time.sleep(sample_interval)

        print("  ⚠ Cycle still running after timeout")
        return True

    def stop_cycle(self) -> bool:
        """Stop the active cycle."""
        print("\n[TEST 4] Stop Cycle")
        response = self.curl("POST", f"{self.coord_url}/cycle/stop")
        
        if not response.get("run_id"):
            print(f"  ✗ Failed to stop cycle: {response}")
            return False
        
        print(f"  ✓ Stop command sent to {response.get('run_id')}")
        
        # Wait for finalization
        for i in range(10):
            time.sleep(0.5)
            proc_status = self.curl("GET", f"{self.proc_url}/status")
            if proc_status.get("status") != "running":
                print(f"  ✓ Cycle finalized after {(i+1)*0.5:.1f}s")
                return True
        
        print("  ✗ Timeout waiting for cycle to finalize")
        return False

    def verify_archive(self) -> bool:
        """Verify the generated zip archive."""
        print("\n[TEST 5] Verify Archive")
        
        if not self.run_id:
            print("  ✗ No run_id available")
            return False

        data_dir = Path(os.getenv("DATA_DIR", "/tmp/edge_data"))
        archive_path = data_dir / "runs" / self.run_id / "bundle.zip"

        if not archive_path.exists():
            print(f"  ✗ Archive not found: {archive_path}")
            return False

        archive_size = archive_path.stat().st_size
        print(f"  ✓ Archive exists: {archive_path} ({archive_size:,} bytes)")

        # Inspect archive contents
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                files = zf.namelist()
                image_files = [f for f in files if f.endswith('.jpg')]
                metadata_files = [f for f in files if 'metadata' in f]

                print(f"  Contents:")
                print(f"    - {len(image_files)} JPG images")
                print(f"    - {len(metadata_files)} metadata files")
                
                if image_files:
                    print(f"    - First image: {image_files[0]}")
                
                # Display first few metadata entries
                for mf in metadata_files:
                    if 'metadata.jsonl' in mf:
                        with zf.open(mf) as f:
                            first_lines = f.read().decode('utf-8').split('\n')[:3]
                            for line in first_lines:
                                if line.strip():
                                    try:
                                        entry = json.loads(line)
                                        print(f"    - Sample metadata: image={entry.get('filename')}, "
                                              f"model_score={entry.get('model_score', 'N/A')}")
                                    except:
                                        pass

                return len(image_files) > 0
        except zipfile.BadZipFile:
            print(f"  ✗ Archive is not a valid ZIP file")
            return False

    def verify_metadata(self) -> bool:
        """Verify run.json metadata."""
        print("\n[TEST 6] Verify Run Metadata")
        
        if not self.run_id:
            print("  ✗ No run_id available")
            return False

        data_dir = Path(os.getenv("DATA_DIR", "/tmp/edge_data"))
        metadata_path = data_dir / "runs" / self.run_id / "run.json"

        if not metadata_path.exists():
            print(f"  ✗ Metadata not found: {metadata_path}")
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            required_fields = ['run_id', 'label', 'status', 'image_count', 'bytes_written']
            missing = [f for f in required_fields if f not in metadata]
            
            if missing:
                print(f"  ✗ Missing metadata fields: {missing}")
                return False
            
            print(f"  ✓ Run metadata valid:")
            print(f"    - run_id: {metadata['run_id']}")
            print(f"    - label: {metadata['label']}")
            print(f"    - status: {metadata['status']}")
            print(f"    - images: {metadata['image_count']}")
            print(f"    - bytes: {metadata['bytes_written']:,}")
            
            return metadata['status'] in ['stopped', 'aborted']
        except Exception as e:
            print(f"  ✗ Error reading metadata: {e}")
            return False

    def run_full_test(self):
        """Execute complete test workflow."""
        print("=" * 60)
        print("Edge Service End-to-End Test")
        print("=" * 60)

        tests = [
            ("Health Check", self.check_health),
            ("Start Cycle", lambda: self.start_cycle(duration_seconds=20)),
            ("Monitor Cycle", self.monitor_cycle),
            ("Stop Cycle", self.stop_cycle),
            ("Verify Archive", self.verify_archive),
            ("Verify Metadata", self.verify_metadata),
        ]

        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, "✓ PASS" if result else "✗ FAIL"))
            except Exception as e:
                print(f"  ✗ Exception: {e}")
                results.append((name, "✗ FAIL"))

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for name, status in results:
            print(f"{name:25} {status}")
        
        passed = sum(1 for _, s in results if "PASS" in s)
        print(f"\nResult: {passed}/{len(results)} tests passed")
        
        return all("PASS" in s for _, s in results)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage: python3 test_e2e.py [coord_port] [proc_port]")
        print("  coord_port: Coordinator port (default: 8081)")
        print("  proc_port:  Processor port (default: 8082)")
        sys.exit(0)

    coord_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    proc_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8082

    tester = EdgeServiceTester(coord_port, proc_port)
    success = tester.run_full_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
