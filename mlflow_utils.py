"""
mlflow_utils.py - shared MLflow experiment tracking for the lesson notebooks
===========================================================================
A single import that wires every lesson into ONE local MLflow store, so runs
made from different notebooks (and different folders) are comparable in the
MLflow UI.

Why this exists
---------------
The notebooks train models three different ways - the Hugging Face ``Trainer``,
hand-written PyTorch/PyG loops, and plain scikit-learn - and they launch from
several folders (``plm/``, ``gnn/``, ``capstones/``). This module hides those
differences behind one helper so each notebook needs only a couple of lines.

Backend (local, no server daemon)
---------------------------------
A local **SQLite** tracking DB at the repo root (``mlflow.db``) plus a local
artifact directory (``mlartifacts/``). SQLite is what powers the run-search and
side-by-side compare views in the UI. View runs with::

    .venv\\Scripts\\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

then open http://127.0.0.1:5000.

Usage in a notebook
-------------------
    import mlflow, mlflow_utils as mu          # see the import shim below
    with mu.run("plm-solubility", "l3_esm2_8M_finetune",
                params={"model": MODEL_ID, "epochs": 3, "lr": 2e-5}):
        ...                                     # train
        mlflow.log_metric("test_acc", acc)

For the HF ``Trainer``: inside the ``with mu.run(...)`` block pass
``TrainingArguments(report_to="mlflow", run_name=...)`` and per-step train/eval
metrics are logged automatically into that run.

Importing this module from any lesson folder
--------------------------------------------
The notebooks live one level below the repo root. Mirror the ``gnn_common``
shim so the import works regardless of where Jupyter was launched::

    import os, sys
    _root = os.path.abspath("")
    for _cand in (_root, os.path.dirname(_root)):
        if os.path.isfile(os.path.join(_cand, "mlflow_utils.py")):
            sys.path.insert(0, _cand); break
    import mlflow, mlflow_utils as mu
"""

import contextlib
import os
import platform
import subprocess

# File names live at the repo root so every notebook shares one store.
_DB_NAME = "mlflow.db"
_ART_NAME = "mlartifacts"


def _repo_root():
    """Locate the repo root (the dir holding ``requirements.txt``).

    Notebooks run with a CWD of either the repo root or a lesson subfolder, so
    we walk up a couple of levels looking for the marker file.
    """
    here = os.path.abspath("")
    for cand in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.isfile(os.path.join(cand, "requirements.txt")):
            return cand
    return here


def _as_file_uri(path):
    """Turn an absolute Windows/Posix path into a ``file:///`` URI for MLflow."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def init_tracking():
    """Point MLflow at the repo-root SQLite DB. Idempotent; safe to call often."""
    import mlflow

    root = _repo_root()
    db = os.path.join(root, _DB_NAME).replace("\\", "/")
    os.makedirs(os.path.join(root, _ART_NAME), exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    return mlflow.get_tracking_uri()


def setup_mlflow():
    """One-call notebook setup: configure the local SQLite tracking store and
    return the ready-to-use ``mlflow`` module.

    Replaces the per-notebook boilerplate that imported ``mlflow`` and called the
    tracking setup by hand. Note the *one* line that still has to live in each
    notebook is putting the repo root on ``sys.path`` so this module can be
    imported in the first place (you can't call a helper you can't yet import)::

        import os, sys
        for _cand in (os.path.abspath(""), os.path.dirname(os.path.abspath(""))):
            if os.path.isfile(os.path.join(_cand, "mlflow_utils.py")):
                sys.path.insert(0, _cand); break
        import mlflow_utils as mu
        mlflow = mu.setup_mlflow()              # <- everything else lives here

    Returns:
        the ``mlflow`` module, with its tracking URI already pointed at the
        repo-root ``mlflow.db``.
    """
    import mlflow

    init_tracking()
    return mlflow


def _git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _env_tags():
    """Reproducibility tags attached to every run (device, versions, OS)."""
    tags = {"python": platform.python_version(), "os": platform.system()}
    try:
        import torch

        tags["torch"] = torch.__version__
        tags["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            tags["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        tags["device"] = "cpu"
    return tags


# Tag prefix the experiment driver sets (e.g. MLFLOW_TAG_campaign=overnight-...).
_ENV_TAG_PREFIX = "MLFLOW_TAG_"


def _env_run_tags():
    """Tags injected via environment, e.g. campaign / exp_id / run_uid / config_hash.

    The papermill driver (experiments/run_experiment.sh) exports
    ``MLFLOW_TAG_<name>=<value>`` for each; this lets every ``run(...)`` in a
    notebook pick them up without the notebook itself being aware of them, so
    overnight campaigns and reproducibility ids attach automatically.
    """
    # Lower-case the stripped key: Windows upper-cases env-var names, so
    # MLFLOW_TAG_campaign comes back as MLFLOW_TAG_CAMPAIGN; MLflow tag keys are
    # case-sensitive, so we normalise to a stable lower-case convention.
    return {k[len(_ENV_TAG_PREFIX):].lower(): v
            for k, v in os.environ.items() if k.upper().startswith(_ENV_TAG_PREFIX)}


def set_experiment(name):
    """Select (creating if needed) an experiment whose artifacts live under
    ``mlartifacts/<name>`` at the repo root."""
    import mlflow

    init_tracking()
    root = _repo_root()
    if mlflow.get_experiment_by_name(name) is None:
        art = _as_file_uri(os.path.join(root, _ART_NAME, name))
        mlflow.create_experiment(name, artifact_location=art)
    mlflow.set_experiment(name)


@contextlib.contextmanager
def run(experiment, run_name, params=None, tags=None, nested=False):
    """Start an MLflow run with sensible defaults.

    Args:
        experiment: experiment name (one per task, e.g. ``"plm-solubility"``).
        run_name:   human-readable run label (lesson + key config).
        params:     dict of hyper-parameters to log once at the start.
        tags:       extra tags merged on top of the automatic env/git tags.
        nested:     ``True`` for child runs inside a parent ``run(...)`` block
                    (used by the multi-seed / grid lessons).

    Yields:
        the active :class:`mlflow.ActiveRun`.
    """
    import mlflow

    set_experiment(experiment)
    auto = {"git_commit": _git_commit(_repo_root()), "track": experiment.split("-")[0]}
    auto.update(_env_tags())
    if tags:
        auto.update(tags)
    # Driver-injected campaign / reproducibility ids win last so every run in a
    # papermill-driven notebook carries them, even nested child runs.
    auto.update(_env_run_tags())
    with mlflow.start_run(run_name=run_name, nested=nested) as active:
        if params:
            mlflow.log_params(params)
        mlflow.set_tags(auto)
        yield active


def log_results_dir(path, exts=(".png", ".csv", ".json")):
    """Log every plot/CSV a lesson already wrote to its ``results/`` folder as
    artifacts of the active run. No-op if the folder does not exist."""
    import mlflow

    if not os.path.isdir(path):
        return
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith(tuple(exts)):
            mlflow.log_artifact(os.path.join(path, fn))


# ---------------------------------------------------------------------------
# Run-comparison chart: a lesson's own MLflow history at a glance
# ---------------------------------------------------------------------------
# Best-known "headline number" per task, best first. Anything not here falls
# back to the first metric present (sorted), so new metrics still chart.
_COMPARE_PRIORITY = (
    "test_acc", "test_accuracy", "test_auroc", "test_f1", "test_mcc",
    "accuracy", "acc", "auroc", "auc", "f1", "mcc", "r2",
    "spearman_best", "spearman", "best_prec_at_l", "avg_prec_at_l",
    "precision_at_l", "recovery_pct",
)


def _compact(v):
    """Shorten a param value for an axis label (e.g. a HF model id)."""
    s = str(v)
    if "/" in s:
        s = s.split("/")[-1]
    return s.replace("_UR50D", "").replace("_UR90S_1", "")


def _pick_compare_metric(present):
    for cand in _COMPARE_PRIORITY:
        if cand in present:
            return cand
    return sorted(present)[0] if present else None


def compare_runs(experiment, metric=None, max_runs=40):
    """Return ``(DataFrame, metric)`` of runs in ``experiment`` (oldest first),
    restricted to those that recorded ``metric``. ``metric`` is auto-detected
    when ``None``. Returns an empty frame if the experiment/metric is absent —
    never raises, so it is safe in a fresh checkout."""
    import mlflow
    import pandas as pd

    init_tracking()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return pd.DataFrame(), None
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                            order_by=["attributes.start_time ASC"],
                            max_results=max_runs)
    if df is None or df.empty:
        return pd.DataFrame(), None
    present = [c[len("metrics."):] for c in df.columns
               if c.startswith("metrics.") and df[c].notna().any()]
    metric = metric or _pick_compare_metric(present)
    mcol = f"metrics.{metric}" if metric else None
    if not mcol or mcol not in df.columns:
        return pd.DataFrame(), metric
    return df[df[mcol].notna()].copy(), metric


def plot_run_comparison(experiment, metric=None, max_runs=40, figsize=None):
    """Bar chart comparing the headline ``metric`` across every logged run of an
    MLflow ``experiment`` — a lesson's own run history at a glance.

    Drop it at the end of a lesson: once the current run is logged it shows how
    that run stacks up against prior ones (more epochs, a bigger model, ...).
    Bars are labelled by the param that varies across runs (epochs / model /
    method); the best run is blue, the latest is orange. Degrades to a printed
    message (no exception) when there are no runs yet or the metric is absent,
    so it is safe under papermill and in a fresh checkout.

    Returns the matplotlib ``Figure`` (auto-displayed as the cell's last value),
    or ``None`` when there is nothing to plot.
    """
    df, metric = compare_runs(experiment, metric=metric, max_runs=max_runs)
    if df.empty or not metric:
        print(f"[run-comparison] nothing to plot yet for {experiment!r} "
              f"(need >=1 run with a metric) — re-run to accumulate history.")
        return None

    import matplotlib.pyplot as plt

    vals = df[f"metrics.{metric}"].astype(float).tolist()
    pcols = [c for c in df.columns if c.startswith("params.")]
    varying = [c for c in pcols if df[c].astype(str).nunique(dropna=True) > 1]
    pref = [c for c in ("params.epochs", "params.model", "params.model_id",
                        "params.method", "params.n_train") if c in varying]
    label_cols = pref or varying[:2]

    names = (df["tags.mlflow.runName"].tolist()
             if "tags.mlflow.runName" in df.columns else [None] * len(df))
    ids = df["run_id"].tolist() if "run_id" in df.columns else [""] * len(df)
    labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        parts = [f"{c.split('.', 1)[1]}={_compact(row[c])}"
                 for c in label_cols
                 if row[c] is not None and str(row[c]) != "nan"]
        if not parts:
            parts = [names[i] if isinstance(names[i], str) and names[i]
                     else str(ids[i])[:8]]
        labels.append("\n".join(parts))

    n = len(vals)
    best = max(range(n), key=lambda i: vals[i])
    colors = ["#bcbcbc"] * n
    colors[best] = "#2c7fb8"                      # best run
    if best != n - 1:
        colors[-1] = "#d95f02"                    # latest run (if not the best)

    fig, ax = plt.subplots(figsize=figsize or (max(6, 1.15 * n), 4.4))
    bars = ax.bar(range(n), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(metric)
    if vals and all(0.0 <= v <= 1.0 for v in vals):
        ax.set_ylim(0, 1)
    ax.set_title(f"{experiment} — {metric} across {n} run(s)  "
                 f"(blue = best, orange = latest)")
    plt.tight_layout()
    return fig
