"""Resource gate for remote-control notebook verification.

Prints current CPU and GPU utilisation and exits 0 only when BOTH are at or
below the threshold (default 70%), so callers can gate compute-heavy notebook
runs on it. Exit 1 means "too busy, wait and retry".

    .venv\\Scripts\\python.exe _resource_check.py [threshold]
"""
import subprocess
import sys

import psutil

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0

cpu = psutil.cpu_percent(interval=1.0)
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"]
    ).decode().strip()
    gpu = max(int(x) for x in out.splitlines() if x.strip())
except Exception as e:  # no GPU / driver issue -> treat GPU as idle
    gpu = 0
    print(f"(nvidia-smi unavailable: {e})")

ok = cpu <= THRESHOLD and gpu <= THRESHOLD
print(f"CPU={cpu:.0f}% GPU={gpu}% threshold={THRESHOLD:.0f}% -> {'OK' if ok else 'BUSY'}")
sys.exit(0 if ok else 1)
