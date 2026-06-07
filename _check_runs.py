import sys
import mlflow
import mlflow_utils as mu

mu.init_tracking()
name = sys.argv[1] if len(sys.argv) > 1 else None
exps = [mlflow.get_experiment_by_name(name)] if name else mlflow.search_experiments()
for exp in exps:
    if exp is None:
        print("no such experiment:", name); continue
    df = mlflow.search_runs([exp.experiment_id])
    print(f"\n=== experiment: {exp.name} ({len(df)} runs) ===")
    if df.empty:
        continue
    cols = [c for c in df.columns if c in (
        "tags.mlflow.runName", "tags.lesson", "tags.seed", "tags.model",
        "metrics.test_accuracy", "metrics.test_f1", "metrics.test_acc",
        "metrics.eval_f1", "tags.device")]
    print(df[cols].to_string())
