"""Parameterised runner for the attrib-PINCH PPI explainability experiment.

Run the SAME pipeline with different hyper-parameters and have every run land in
one local MLflow store, so they are directly comparable in the MLflow UI. This
is the scale-up / comparison harness referenced in the repo README.

Examples
--------
    # one configuration
    .venv\\Scripts\\python.exe ig\\run_pinch.py --dropout 0.3 --wd 1e-2

    # the overfit -> regularised -> collapse sweep, as separate comparable runs
    .venv\\Scripts\\python.exe ig\\run_pinch.py --sweep

    # also render a parameterised notebook artifact (needs papermill)
    .venv\\Scripts\\python.exe ig\\run_pinch.py --dropout 0.3 --wd 1e-2 --render pinch_l5

    # view results
    .venv\\Scripts\\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db

Every compute run is gated by ``_resource_check.py``: if CPU or GPU is above the
threshold the runner waits and re-checks (default: 5 minutes, up to 6 times)
before starting, honouring the "don't start heavy work on a busy box" rule.

``--dataset pinder`` is intentionally not wired (option *b*); it will raise.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESOURCE_CHECK = os.path.join(ROOT, "_resource_check.py")
DATA = os.path.join(ROOT, "data")


def wait_for_resources(threshold=70.0, wait_s=300, max_waits=6):
    """Block until CPU and GPU are both <= threshold, re-checking every wait_s.

    Returns True once the box is idle enough, or False if still busy after
    max_waits attempts (so the caller can abort rather than hang forever).
    """
    for attempt in range(max_waits + 1):
        res = subprocess.run([sys.executable, RESOURCE_CHECK, str(threshold)])
        if res.returncode == 0:
            return True
        if attempt < max_waits:
            print(f"[gate] busy; waiting {wait_s}s then re-checking "
                  f"({attempt + 1}/{max_waits})...", flush=True)
            time.sleep(wait_s)
    return False


def _clean_metrics(metrics):
    """Keep only finite numeric metrics (MLflow rejects NaN/strings)."""
    import math
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out[k] = v
    return out


def run_one(hp, experiment, run_name, tag=None, render=None,
            threshold=70.0, wait_s=300, max_waits=6):
    """Gate, run one configuration, and log params + metrics to MLflow.

    ``tag`` becomes the ``report_tag`` MLflow tag — runs sharing it are grouped
    by ``report_pinch.py`` into one tag-level report.
    """
    from pinch_common import run_experiment
    import mlflow
    import mlflow_utils as mu

    if not wait_for_resources(threshold, wait_s, max_waits):
        print(f"[gate] still busy after {max_waits} waits — skipping '{run_name}'.")
        return None

    tags = {"harness": "run_pinch"}
    if tag:
        tags["report_tag"] = tag
    print(f"[run] {run_name}  params={hp}")
    with mu.run(experiment, run_name, params=hp, tags=tags):
        metrics = run_experiment(cache_dir=DATA, **hp)
        mlflow.log_metrics(_clean_metrics(metrics))
        if render:
            art = _render_notebook(render, hp, run_name)
            if art:
                mlflow.log_artifact(art)
    # console summary
    keys = ["train_acc", "pooled_measured", "pooled_hotspots", "ig_delta_max",
            "auroc_attention", "auroc_ig", "auroc_gnnexplainer", "auroc_consensus"]
    print("  " + "  ".join(f"{k}={metrics[k]:.3f}" if isinstance(metrics.get(k), float)
                           else f"{k}={metrics.get(k)}" for k in keys if k in metrics))
    return metrics


def _render_notebook(stem, hp, run_name):
    """Execute a parameterised notebook with papermill; return the output path."""
    try:
        import papermill as pm
    except ImportError:
        print("[render] papermill not installed; skipping notebook artifact. "
              "Install with: pip install papermill")
        return None
    inp = os.path.join(HERE, f"{stem}.ipynb")
    if not os.path.isfile(inp):
        print(f"[render] no such notebook: {inp}")
        return None
    out_dir = os.path.join(HERE, "_runs")
    os.makedirs(out_dir, exist_ok=True)
    safe = run_name.replace(" ", "_").replace("/", "-")
    out = os.path.join(out_dir, f"{stem}__{safe}.ipynb")
    # Pass only params the notebook actually declares (upper-cased names).
    pm_params = {k.upper(): v for k, v in hp.items()}
    print(f"[render] papermill {stem} -> {out}")
    pm.execute_notebook(inp, out, parameters=pm_params, kernel_name="python3")
    return out


SWEEP = [
    dict(dropout=0.0, weight_decay=0.0),    # overfit baseline
    dict(dropout=0.3, weight_decay=1e-3),   # light regularisation
    dict(dropout=0.3, weight_decay=1e-2),   # heavy -> collapse
    dict(dropout=0.5, weight_decay=1e-2),
]


def build_hp(args):
    return dict(dropout=args.dropout, weight_decay=args.wd, hidden=args.hidden,
                epochs=args.epochs, lr=args.lr, augment=args.augment,
                seed=args.seed, threshold=args.threshold, ig_steps=args.ig_steps,
                gnnx_epochs=args.gnnx_epochs, dataset=args.dataset)


def main():
    ap = argparse.ArgumentParser(description="Parameterised attrib-PINCH experiment runner")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", "--weight-decay", dest="wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--augment", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="contact-graph edge threshold in angstroms")
    ap.add_argument("--ig-steps", type=int, default=200)
    ap.add_argument("--gnnx-epochs", type=int, default=80)
    ap.add_argument("--dataset", default="demo", choices=["demo", "pinder"])
    ap.add_argument("--experiment", default="pinch-ppi-xai")
    ap.add_argument("--name", default=None, help="run name (auto if omitted)")
    ap.add_argument("--tag", default=None,
                    help="report group tag; runs sharing it are reported together "
                         "by report_pinch.py --tag <tag>")
    ap.add_argument("--sweep", action="store_true",
                    help="run the overfit->regularised->collapse preset sweep")
    ap.add_argument("--render", default=None, metavar="NOTEBOOK_STEM",
                    help="also papermill a notebook, e.g. pinch_l5_skempi_consensus_benchmark")
    ap.add_argument("--gpu-threshold", dest="gate_threshold", type=float, default=70.0,
                    help="CPU/GPU busy threshold for the resource gate")
    ap.add_argument("--wait", type=int, default=300, help="gate re-check interval (s)")
    ap.add_argument("--max-waits", type=int, default=6)
    args = ap.parse_args()

    if args.sweep:
        for cfg in SWEEP:
            hp = build_hp(args)
            hp.update(cfg)
            name = f"sweep d={cfg['dropout']} wd={cfg['weight_decay']:.0e}"
            run_one(hp, args.experiment, name, tag=args.tag or "reg-sweep",
                    threshold=args.gate_threshold, wait_s=args.wait,
                    max_waits=args.max_waits)
    else:
        hp = build_hp(args)
        name = args.name or f"d={hp['dropout']} wd={hp['weight_decay']:.0e} aug={hp['augment']}"
        run_one(hp, args.experiment, name, tag=args.tag, render=args.render,
                threshold=args.gate_threshold, wait_s=args.wait,
                max_waits=args.max_waits)

    print("\nCompare runs:  .venv\\Scripts\\python.exe -m mlflow ui "
          "--backend-store-uri sqlite:///mlflow.db   (http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()
