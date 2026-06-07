import mlflow
import mlflow_utils as mu

with mu.run("smoke-test", "helper_check", params={"lr": 0.01, "epochs": 2}):
    for e in range(2):
        mlflow.log_metric("val_acc", 0.5 + 0.1 * e, step=e)
    mlflow.log_metric("test_acc", 0.71)

print("tracking_uri:", mlflow.get_tracking_uri())
exp = mlflow.get_experiment_by_name("smoke-test")
df = mlflow.search_runs([exp.experiment_id])
cols = ["tags.mlflow.runName", "params.lr", "metrics.test_acc", "tags.device", "tags.git_commit"]
print(df[cols].to_string())
