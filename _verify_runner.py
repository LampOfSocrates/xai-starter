"""Resource-gated notebook verification for the MLflow wiring.

For each notebook: poll CPU + GPU utilisation; if EITHER is above the threshold
(default 70%) wait 5 minutes and re-check before starting. Once clear, execute
the notebook in place via nbconvert and record pass/fail. The gate protects the
user's own GPU/CPU work -- we never kick off a run while the machine is busy.

    .venv\\Scripts\\python.exe _verify_runner.py
"""
import os
import subprocess
import sys
import time

import psutil

THRESHOLD = 70.0
WAIT_SECONDS = 300          # 5 minutes
MAX_WAITS = 24             # give up after ~2h of waiting on one notebook
PER_NB_TIMEOUT = 1800      # 30 min per notebook

# Lightest compute first so wiring bugs surface early.
NOTEBOOKS = [
    "plm/plm_l1_embeddings_probe.ipynb",
    "plm/plm_l5_model_comparison.ipynb",
    "plm/plm_l11_calibration.ipynb",
    "plm/plm_l9_embedding_retrieval.ipynb",
    "plm/plm_l7_lora_peft.ipynb",
    "plm/plm_l4_token_classification.ipynb",
    "gnn/gnn_l3_graph_classification.ipynb",
    "gnn/gnn_l11_interaction_graphs.ipynb",
    "gnn/gnn_l10_oversmoothing.ipynb",
    "gnn/gnn_l7_edge_features.ipynb",
    "gnn/gnn_l8_contact_prediction.ipynb",
    "gnn/gnn_l4_plm_plus_gnn.ipynb",
    "gnn/gnn_l9_knn_graphs.ipynb",
    "capstones/capstone_l1_end_to_end_plm_gnn.ipynb",
]


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"]
        ).decode().strip()
        return max(int(x) for x in out.splitlines() if x.strip())
    except Exception:
        return 0


def wait_until_clear(tag):
    """Block until CPU and GPU are both <= THRESHOLD, re-checking every 5 min."""
    for attempt in range(MAX_WAITS + 1):
        cpu = psutil.cpu_percent(interval=1.0)
        gpu = gpu_util()
        if cpu <= THRESHOLD and gpu <= THRESHOLD:
            print(f"  [gate] CPU={cpu:.0f}% GPU={gpu}% -> clear, starting {tag}", flush=True)
            return True
        print(f"  [gate] CPU={cpu:.0f}% GPU={gpu}% > {THRESHOLD:.0f}% -- waiting "
              f"5 min (attempt {attempt + 1}/{MAX_WAITS}) before {tag}", flush=True)
        time.sleep(WAIT_SECONDS)
    print(f"  [gate] gave up waiting for {tag} after {MAX_WAITS} checks", flush=True)
    return False


def main():
    env = dict(os.environ)
    try:
        import certifi
        env["SSL_CERT_FILE"] = certifi.where()   # conda's bundle breaks venv downloads
    except Exception:
        pass
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    results = []
    for nb in NOTEBOOKS:
        print(f"\n=== {nb} ===", flush=True)
        if not wait_until_clear(nb):
            results.append((nb, "SKIPPED_BUSY"))
            continue
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", f"--ExecutePreprocessor.timeout={PER_NB_TIMEOUT}", nb],
            env=env, capture_output=True, text=True,
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            results.append((nb, f"OK ({dt:.0f}s)"))
            print(f"  PASS in {dt:.0f}s", flush=True)
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            results.append((nb, f"FAIL ({dt:.0f}s)"))
            print(f"  FAIL in {dt:.0f}s:\n    " + "\n    ".join(tail), flush=True)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for nb, status in results:
        print(f"  {status:18s} {nb}")
    n_ok = sum(1 for _, s in results if s.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} notebooks passed")


if __name__ == "__main__":
    main()
