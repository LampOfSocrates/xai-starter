"""One-shot MLflow wiring for the GNN-track notebooks.

For each notebook: insert an MLflow import-shim cell right after the imports
cell, then splice an `mlflow`-logging block into main() at a unique anchor.
Idempotent: re-running is a no-op once the shim/anchor markers are present.
gnn_l5 is intentionally absent -- it is an equivariance demonstration with no
trained metric to log.
"""
import json
import sys

SHIM_OS = '''# --- MLflow experiment tracking -------------------------------------------
# Import the shared helper regardless of where Jupyter launched from.
# See mlflow_utils.py for the repo-root SQLite backend.
import sys
_root = os.path.abspath("")
for _cand in (_root, os.path.dirname(_root)):
    if os.path.isfile(os.path.join(_cand, "mlflow_utils.py")):
        sys.path.insert(0, _cand); break
import mlflow
import mlflow_utils as mu
'''

SHIM_NOOS = '''# --- MLflow experiment tracking -------------------------------------------
# Import the shared helper regardless of where Jupyter launched from.
# See mlflow_utils.py for the repo-root SQLite backend.
import os, sys
for _cand in (os.path.abspath(""), os.path.dirname(os.path.abspath(""))):
    if os.path.isfile(os.path.join(_cand, "mlflow_utils.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import mlflow
import mlflow_utils as mu
'''

# (path, main_cell_id, [(old, new), ...] pre-replacements, anchor, block)
SPECS = [
    ("gnn/gnn_l3_graph_classification.ipynb", "a953bace", [],
     '    print(f"\\nFinal test accuracy: {final_acc:.3f}")\n',
     '\n    with mu.run("gnn-graph-classification", "l3_gat_onehot",\n'
     '                params={"dataset": DATASET_NAME, "model": "GAT", "features": "one-hot",\n'
     '                        "n_train": N_TRAIN, "n_test": N_TEST, "window": WINDOW,\n'
     '                        "hidden": HIDDEN, "heads": HEADS, "epochs": EPOCHS,\n'
     '                        "batch_size": BATCH_SIZE, "lr": LR},\n'
     '                tags={"lesson": "gnn_l3", "model": "GAT", "features": "one-hot"}):\n'
     '        mlflow.log_metrics({"test_acc": float(final_acc),\n'
     '                            "majority_acc": float(max(train_pos, 1 - train_pos))})\n'),

    ("gnn/gnn_l4_plm_plus_gnn.ipynb", "8591d685",
     [('    print(f"\\nFinal test accuracy: {evaluate(model, test_loader, device):.3f}")\n',
       '    final_acc = evaluate(model, test_loader, device)\n'
       '    print(f"\\nFinal test accuracy: {final_acc:.3f}")\n')],
     '    # Hand the trained model + data back so the analysis section below can\n',
     '    with mu.run("gnn-graph-classification", "l4_gat_esm2feats",\n'
     '                params={"dataset": DATASET_NAME, "plm": PLM_NAME, "model": "GAT",\n'
     '                        "features": "ESM-2", "n_train": N_TRAIN, "n_test": N_TEST,\n'
     '                        "hidden": HIDDEN, "heads": HEADS, "epochs": EPOCHS,\n'
     '                        "batch_size": BATCH_SIZE, "lr": LR},\n'
     '                tags={"lesson": "gnn_l4", "model": "GAT", "features": "ESM-2"}):\n'
     '        mlflow.log_metric("test_acc", float(final_acc))\n\n'),

    ("gnn/gnn_l7_edge_features.ipynb", "3f772b42", [],
     '    print(f"\\nSaved accuracy plot to {savepath}")\n',
     '\n    with mu.run("gnn-edge-features", "l7_gcn_vs_edgegnn",\n'
     '                params={"dataset": "synthetic_helix_hairpin", "n_train": N_TRAIN,\n'
     '                        "n_test": N_TEST, "hidden": HIDDEN, "epochs": EPOCHS,\n'
     '                        "lr": LR, "batch_size": BATCH_SIZE, "n_rbf": N_RBF,\n'
     '                        "edge_dim": int(edge_dim)},\n'
     '                tags={"lesson": "gnn_l7"}):\n'
     '        mlflow.log_metrics({"gcn_test_acc": float(gcn_acc),\n'
     '                            "edge_test_acc": float(edge_acc),\n'
     '                            "edge_minus_gcn": float(delta)})\n'
     '        if os.path.isfile(savepath):\n'
     '            mlflow.log_artifact(savepath)\n'),

    ("gnn/gnn_l8_contact_prediction.ipynb", "7acd8e71", [],
     '    print(f"\\nContact map figure saved to: {fig_path}")\n',
     '\n    with mu.run("gnn-contact-prediction", "l8_contact_predictor",\n'
     '                params={"dataset": "synthetic_miniproteins", "n_train": N_TRAIN_PROTEINS,\n'
     '                        "n_test": N_TEST_PROTEINS, "hidden": HIDDEN,\n'
     '                        "gnn_layers": GNN_LAYERS, "epochs": EPOCHS, "lr": LR},\n'
     '                tags={"lesson": "gnn_l8"}):\n'
     '        mlflow.log_metrics({"precision_at_L": float(final_prec),\n'
     '                            "mean_auc": float(final_auc),\n'
     '                            "random_precision": float(mean_contact_rate),\n'
     '                            "lift": float(lift)})\n'
     '        if os.path.isfile(fig_path):\n'
     '            mlflow.log_artifact(fig_path)\n'),

    ("gnn/gnn_l9_knn_graphs.ipynb", "1ff34da2", [],
     '    print(f"\\nBest graph: {best[0]}  ({best[1]:.3f})")\n',
     '\n    with mu.run("gnn-knn-graphs", "l9_graph_construction",\n'
     '                params={"dataset": DATASET_NAME, "plm": PLM_NAME, "n_train": N_TRAIN,\n'
     '                        "n_test": N_TEST, "knn_k": KNN_K, "model": "GAT"},\n'
     '                tags={"lesson": "gnn_l9"}):\n'
     '        mlflow.log_metrics({"acc_sequence_window": float(acc_seq),\n'
     '                            "acc_embedding_knn": float(acc_knn),\n'
     '                            "acc_random_control": float(acc_rand),\n'
     '                            "avg_edges_knn": float(avg_edges_knn),\n'
     '                            "avg_edges_seq": float(avg_edges_seq)})\n'
     '        _viz = "./results/gnn_l9_edge_comparison.png"\n'
     '        if os.path.isfile(_viz):\n'
     '            mlflow.log_artifact(_viz)\n'),

    ("gnn/gnn_l10_oversmoothing.ipynb", "b3e4cd28", [],
     '    print(f"\\nSaved plot to {fig_path}")\n',
     '\n    with mu.run("gnn-oversmoothing", "l10_depth_sweep",\n'
     '                params={"depths": str(DEPTHS), "hidden": HIDDEN, "n_train": N_TRAIN,\n'
     '                        "n_test": N_TEST, "batch_size": BATCH_SIZE},\n'
     '                tags={"lesson": "gnn_l10"}):\n'
     '        for _i, _d in enumerate(DEPTHS):\n'
     '            mlflow.log_metrics({"vanilla_acc": float(vanilla_acc[_i]),\n'
     '                                "vanilla_mpcs": float(vanilla_mpcs[_i]),\n'
     '                                "vanilla_dirichlet": float(vanilla_de[_i]),\n'
     '                                "residual_acc": float(residual_acc[_i]),\n'
     '                                "residual_mpcs": float(residual_mpcs[_i])}, step=_d)\n'
     '        if os.path.isfile(fig_path):\n'
     '            mlflow.log_artifact(fig_path)\n'),

    ("gnn/gnn_l11_interaction_graphs.ipynb", "38baf5d9", [],
     '    print(f"  Accuracy : {acc:.4f}  (majority baseline = {majority_acc:.3f})")\n',
     '\n    with mu.run("gnn-interaction-prediction", "l11_hetero_gnn",\n'
     '                params={"n_proteins": N_PROTEINS, "n_drugs": N_DRUGS,\n'
     '                        "protein_dim": PROTEIN_FEAT_DIM, "drug_dim": DRUG_FEAT_DIM,\n'
     '                        "hidden": HIDDEN, "model": "HeteroGNN"},\n'
     '                tags={"lesson": "gnn_l11"}):\n'
     '        mlflow.log_metrics({"test_auc": float(auc), "test_acc": float(acc),\n'
     '                            "majority_acc": float(majority_acc),\n'
     '                            "pos_rate": float(pos_rate)})\n'),
]


def find_import_cell(cells):
    for idx, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "import torch" in src or src.startswith("import os"):
            return idx
    raise RuntimeError("no import cell found")


def main():
    for path, main_id, pre, anchor, block in SPECS:
        nb = json.load(open(path, encoding="utf-8"))
        cells = nb["cells"]
        full = "\n".join("".join(c["source"]) for c in cells)

        # 1. Insert shim after the imports cell (skip if already wired).
        if "mlflow_utils as mu" not in full:
            imp = find_import_cell(cells)
            has_os = "import os" in "".join(cells[imp]["source"])
            shim = SHIM_OS if has_os else SHIM_NOOS
            cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "id": "mlflow-shim", "source": shim.splitlines(keepends=True)}
            cells.insert(imp + 1, cell)

        # 2. Edit the main cell.
        mc = next(c for c in cells if c.get("id") == main_id)
        src = "".join(mc["source"])
        for old, new in pre:
            assert src.count(old) == 1, f"{path}: pre-replace anchor count != 1"
            src = src.replace(old, new)
        if block.strip() not in src:
            assert src.count(anchor) == 1, f"{path}: anchor count != 1 ({anchor!r})"
            src = src.replace(anchor, anchor + block)
        mc["source"] = src.splitlines(keepends=True)

        json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"wired {path}")


if __name__ == "__main__":
    main()
