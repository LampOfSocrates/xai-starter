"""Generate a tag-level report from MLflow runs of the attrib-PINCH experiment.

Every run launched by ``run_pinch.py --tag <T>`` is stamped with the MLflow tag
``report_tag=<T>``. This script pulls all runs sharing a tag, ranks them, and
writes a self-contained markdown report (+ a chart) summarising:

  * each run's config and per-method hotspot-recovery AUROC,
  * the mean AUROC per xAI method across the group (does consensus win?),
  * the best run and best single method,
  * an honest note on label counts (so a lucky AUROC on 14 residues is not
    over-read).

Usage
-----
    .venv\\Scripts\\python.exe ig\\report_pinch.py --tag overnight-reg-sweep
    # -> ig/reports/overnight-reg-sweep.md  (+ .png)
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

METHODS = ["attention", "ig", "gnnexplainer", "consensus"]
PARAM_COLS = ["dropout", "weight_decay", "augment", "epochs", "ig_steps",
              "gnnx_epochs", "threshold", "seed", "dataset"]


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—" if v is None or v == "" else str(v)


def main():
    ap = argparse.ArgumentParser(description="Tag-level MLflow report for attrib-PINCH")
    ap.add_argument("--tag", required=True, help="report_tag shared by the runs")
    ap.add_argument("--experiment", default="pinch-ppi-xai")
    ap.add_argument("--out", default=None, help="output .md path (auto if omitted)")
    args = ap.parse_args()

    import mlflow
    import mlflow_utils as mu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mu.init_tracking()
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        sys.exit(f"No experiment '{args.experiment}' in the MLflow store yet.")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                            filter_string=f"tags.report_tag = '{args.tag}'")
    if df.empty:
        sys.exit(f"No runs tagged report_tag='{args.tag}' in '{args.experiment}'.")

    # reset_index so positional [i] access lines up names, params, and metrics
    # (after sort_values the labels no longer equal 0..n-1).
    df = df.sort_values("start_time").reset_index(drop=True)
    out_dir = os.path.join(HERE, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_md = args.out or os.path.join(out_dir, f"{args.tag}.md")
    out_png = os.path.splitext(out_md)[0] + ".png"

    def col(name):
        return df[name] if name in df.columns else [None] * len(df)

    names = list(col("tags.mlflow.runName"))
    auroc = {m: [the if the is not None else np.nan
                 for the in col(f"metrics.auroc_{m}")] for m in METHODS}
    mean_auroc = {m: float(np.nanmean(auroc[m])) if not all(np.isnan(auroc[m])) else float("nan")
                  for m in METHODS}

    # --- chart: mean AUROC per method across the tag group ---
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [mean_auroc[m] for m in METHODS]
    bars = ax.bar(METHODS, vals, color=["goldenrod", "slateblue", "purple", "crimson"])
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean hotspot-recovery AUROC")
    ax.set_title(f"Tag '{args.tag}': mean AUROC per method ({len(df)} runs)")
    ax.bar_label(bars, fmt="%.3f")
    plt.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # --- best run / best method ---
    best_method = max((m for m in METHODS if not np.isnan(mean_auroc[m])),
                      key=lambda m: mean_auroc[m], default=None)
    cons = np.array(auroc["consensus"], dtype=float)
    best_idx = int(np.nanargmax(cons)) if not all(np.isnan(cons)) else 0

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# attrib-PINCH — report for tag `{args.tag}`\n")
    lines.append(f"_Experiment_ `{args.experiment}` · _{len(df)} runs_ · generated {now}\n")

    lines.append("## Mean AUROC per method (across the tag group)\n")
    lines.append(f"![mean AUROC]({os.path.basename(out_png)})\n")
    lines.append("| Method | mean AUROC |")
    lines.append("|---|---|")
    for m in METHODS:
        star = " ⬅ best" if m == best_method else ""
        lines.append(f"| {m} | {_fmt(mean_auroc[m])}{star} |")
    lines.append("")

    lines.append("## Per-run detail\n")
    header = ["run", "train_acc", "measured", "hotspots", "ig_delta_max"] + \
             [f"AUROC_{m}" for m in METHODS] + ["dropout", "wd", "augment", "ig_steps"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i in range(len(df)):
        row = [
            str(names[i]),
            _fmt(col("metrics.train_acc")[i], 2),
            _fmt(col("metrics.pooled_measured")[i], 0),
            _fmt(col("metrics.pooled_hotspots")[i], 0),
            _fmt(col("metrics.ig_delta_max")[i], 2),
        ]
        row += [_fmt(auroc[m][i]) for m in METHODS]
        row += [
            _fmt(col("params.dropout")[i], 2),
            _fmt(col("params.weight_decay")[i]),
            _fmt(col("params.augment")[i], 0),
            _fmt(col("params.ig_steps")[i], 0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- honest read ---
    measured = [m for m in col("metrics.pooled_measured") if m is not None]
    max_measured = int(max(measured)) if measured else 0
    lines.append("## Read\n")
    if best_method:
        lines.append(f"- Best method on average for this tag: **{best_method}** "
                     f"(mean AUROC {_fmt(mean_auroc[best_method])}).")
        lines.append(f"- Consensus mean AUROC: **{_fmt(mean_auroc['consensus'])}** "
                     f"(vs random 0.5).")
        lead = "leads" if best_method == "consensus" else "does NOT lead"
        lines.append(f"- The **consensus hypothesis** {lead} here.")
    lines.append(f"- Largest labelled set in any run: **{max_measured} residues**. "
                 "At this scale AUROC is high-variance — treat differences of "
                 "<0.1 as noise, and prefer tags with more runs / more complexes.")
    if max_measured < 100:
        lines.append("- ⚠️ Fewer than 100 labelled residues: this is still the "
                     "toy/demo regime. For a real verdict, scale the dataset "
                     "(more complexes, or PINDER — option *b*).")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")
    print(f"runs: {len(df)}  best method: {best_method}  "
          f"consensus mean AUROC: {_fmt(mean_auroc['consensus'])}")


if __name__ == "__main__":
    main()
