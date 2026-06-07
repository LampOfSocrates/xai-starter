"""Blocking resource gate for the experiment driver.

Blocks until CPU and GPU utilisation are both at or below the threshold,
re-checking every 5 minutes. Exit 0 once clear; exit 1 if it gives up.
Threshold via $GATE_THRESHOLD (default 70).

    python experiments/lib/gate.py
"""
import os
import subprocess
import sys
import time

import psutil

THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "70"))
WAIT_SECONDS = int(os.environ.get("GATE_WAIT_SECONDS", "300"))
MAX_WAITS = int(os.environ.get("GATE_MAX_WAITS", "24"))


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"]
        ).decode().strip()
        return max(int(x) for x in out.splitlines() if x.strip())
    except Exception:
        return 0


for attempt in range(MAX_WAITS + 1):
    cpu = psutil.cpu_percent(interval=1.0)
    gpu = gpu_util()
    if cpu <= THRESHOLD and gpu <= THRESHOLD:
        print(f"[gate] CPU={cpu:.0f}% GPU={gpu}% <= {THRESHOLD:.0f}% -> clear", flush=True)
        sys.exit(0)
    print(f"[gate] CPU={cpu:.0f}% GPU={gpu}% > {THRESHOLD:.0f}% -- waiting "
          f"{WAIT_SECONDS}s (attempt {attempt + 1}/{MAX_WAITS})", flush=True)
    time.sleep(WAIT_SECONDS)

print(f"[gate] gave up after {MAX_WAITS} checks", flush=True)
sys.exit(1)
