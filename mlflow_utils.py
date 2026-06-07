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
    return {k[len(_ENV_TAG_PREFIX):]: v
            for k, v in os.environ.items() if k.startswith(_ENV_TAG_PREFIX)}


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
