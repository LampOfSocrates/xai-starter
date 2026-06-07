# Experiments runner

Reproducible, resource-gated, parameterised experiment runs on top of the
lesson notebooks. Each run is executed with [papermill](https://papermill.readthedocs.io/)
and logged to the shared MLflow store with campaign + reproducibility tags.

## One-time setup

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Register THIS project's venv as a Jupyter kernel named "plm-starter":
.venv/Scripts/python.exe -m ipykernel install --user --name plm-starter
```

## Layout

```
experiments/
  run_experiment.sh      generic driver: validate -> gate -> ids -> papermill -> MLflow
  exp_C_benchmark.sh     example wrapper (capstone_l2 benchmark suite)
  batch_overnight.sh     run several experiments sequentially under one campaign
  lib/
    gate.py              blocks until CPU & GPU <= threshold (re-checks every 5 min)
    meta.py              mints run_uid (unique) + config_hash (deterministic)
    mkparams.py          key=value -> typed YAML param file for papermill
  runs/                  executed notebooks + manifests (gitignored)
```

## Run one experiment

```bash
# defaults (small smoke run)
experiments/exp_C_benchmark.sh

# scaled-up, tagged to a campaign
experiments/exp_C_benchmark.sh \
  --n-train 5000 --n-val 1000 --n-test 1000 \
  --plm facebook/esm2_t12_35M_UR50D --seeds '[0,1,2,3,4]' \
  --campaign overnight-2026-06-08
```

Or drive any parameterised notebook directly:

```bash
experiments/run_experiment.sh \
  --exp-id C-sol-benchmark \
  --notebook capstones/capstone_l2_benchmark_suite.ipynb \
  --campaign overnight-2026-06-08 \
  -P N_TRAIN=5000 -P 'SEEDS=[0,1,2,3,4]' -P PLM_NAME=facebook/esm2_t12_35M_UR50D
```

## Run a whole overnight batch (sequential, each gated)

```bash
experiments/batch_overnight.sh overnight-2026-06-08
```

## Identifiers & reproducibility

Every run carries four MLflow tags (lower-case, case-normalised across OSes):

| tag | meaning | stable across reruns? |
|---|---|---|
| `campaign` | the batch label you pass with `--campaign` | yes (you choose it) |
| `exp_id` | logical experiment, e.g. `C-sol-benchmark` | yes |
| `config_hash` | sha1 of the sorted scientific params | **yes — identical params = identical hash** |
| `run_uid` | `<exp_id>.<UTC-timestamp>.<rand6>` | **no — unique per execution** |

To check reproducibility, run the same command twice and compare the runs that
share a `config_hash` but differ in `run_uid` — any metric spread is the
run-to-run noise of that exact configuration.

```python
import mlflow
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.config_hash = '4d49ab64284b'")
```

## Making a notebook runnable here

papermill overrides variables in a cell tagged `parameters`. Tag the notebook's
config cell once:

```python
import json
nb = json.load(open("path/to/notebook.ipynb", encoding="utf-8"))
cell = next(c for c in nb["cells"] if c["cell_type"] == "code"
            and "N_TRAIN" in "".join(c["source"]))
cell.setdefault("metadata", {}).setdefault("tags", [])
if "parameters" not in cell["metadata"]["tags"]:
    cell["metadata"]["tags"].append("parameters")
json.dump(nb, open("path/to/notebook.ipynb", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
```

The campaign / id tags attach automatically: the driver exports
`MLFLOW_TAG_*` env vars and `mlflow_utils.run()` merges them into every run,
so the notebook's `mu.run(...)` calls need no changes.

## Gate

`gate.py` blocks until CPU **and** GPU are at/below the threshold (default 70%,
override with `GATE_THRESHOLD`), re-checking every 5 minutes. Each
`run_experiment.sh` invocation gates first, so a sequential batch never starts a
run while the machine is busy with foreground work.
