"""Generate a complete campaign-level report from the MLflow store.

Every run launched by ``experiments/run_experiment.sh`` is stamped with the tags
``campaign`` / ``exp_id`` / ``config_hash`` / ``run_uid`` / ``reason``. This
script pulls all runs sharing a ``campaign`` (across every MLflow experiment),
groups them by ``exp_id`` and then by ``config_hash`` (so reruns of an identical
config are averaged with a std), auto-detects each experiment's primary metric,
and writes a self-contained markdown report with one bar chart per experiment:

  reports/report_<campaign>.md          the report (links the charts below)
  reports/figs_<campaign>/<exp_id>.png  primary-metric chart per experiment
  reports/summary_<campaign>.json       machine-readable summary

The report is deterministic. With ``--interpret`` it additionally shells out to
the ``claude`` CLI (headless ``claude -p``), hands it the JSON summary, and
splices the model's plain-English interpretation into an ``## Interpretation``
section. The AI step is opt-in so the factual report never depends on it.

Usage
-----
    .venv\\Scripts\\python.exe mlflow_report.py --campaign gnn-smoke-2026-06-08
    .venv\\Scripts\\python.exe mlflow_report.py --campaign overnight-2026-06-08 --interpret
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Names we treat as "the headline number", best first. Anything matching the
# fuzzy fallback (contains acc/auroc/f1/...) is used only if none of these hit.
PRIMARY_PRIORITY = [
    "test_acc", "test_accuracy", "test_auroc", "test_f1", "test_mcc",
    "accuracy", "acc", "auroc", "auc", "f1", "mcc", "r2",
]
SCORE_HINTS = ("acc", "auroc", "auc", "f1", "mcc", "score", "r2")
# Metrics that are diagnostics, not the thing being optimised — never primary.
NON_PRIMARY_HINTS = ("majority", "baseline", "avg_edges", "n_", "count",
                     "loss", "edges", "delta", "measured", "hotspots")


def fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—" if v is None or v == "" else str(v)
    if f != f:  # NaN
        return "—"
    return f"{f:.{nd}f}"


def pick_primary(metric_names):
    """Choose the headline metric for an experiment from the metrics present."""
    present = set(metric_names)
    for cand in PRIMARY_PRIORITY:
        if cand in present:
            return cand
    scored = [m for m in metric_names
              if any(h in m for h in SCORE_HINTS)
              and not any(h in m for h in NON_PRIMARY_HINTS)]
    if scored:
        return sorted(scored)[0]
    plain = [m for m in metric_names if not any(h in m for h in NON_PRIMARY_HINTS)]
    return sorted(plain)[0] if plain else (sorted(metric_names)[0] if metric_names else None)


def load_campaign(campaign):
    import mlflow
    import mlflow_utils as mu
    mu.init_tracking()
    flt = f"tags.campaign = '{campaign}'"
    try:
        df = mlflow.search_runs(filter_string=flt, search_all_experiments=True)
    except TypeError:
        # Older MLflow without search_all_experiments: enumerate experiments.
        ids = [e.experiment_id for e in mlflow.search_experiments()]
        df = mlflow.search_runs(experiment_ids=ids, filter_string=flt)
    return df


def varying_params(group, param_cols):
    """Param columns whose value actually differs across the runs in a group
    (the swept axes worth showing); falls back to all if nothing varies."""
    varying = []
    for c in param_cols:
        vals = set(str(v) for v in group[c].tolist() if v is not None and str(v) != "nan")
        if len(vals) > 1:
            varying.append(c)
    return varying


def summarise(df):
    """Build the structured per-experiment / per-config summary."""
    import numpy as np
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    param_cols = [c for c in df.columns if c.startswith("params.")]

    experiments = []
    for eid in sorted(x for x in df["tags.exp_id"].dropna().unique()):
        g = df[df["tags.exp_id"] == eid]
        # metrics that have at least one value in this experiment
        present = [c.split(".", 1)[1] for c in metric_cols
                   if g[c].notna().any()]
        primary = pick_primary(present)
        swept = varying_params(g, param_cols)
        reason = next((r for r in g["tags.reason"].tolist()
                       if isinstance(r, str) and r.strip()), "") \
            if "tags.reason" in g.columns else ""

        configs = []
        hcol = "tags.config_hash" if "tags.config_hash" in g.columns else None
        keys = sorted(g[hcol].dropna().unique()) if hcol else ["(all)"]
        for h in keys:
            cg = g[g[hcol] == h] if hcol else g
            mvals = {}
            for m in present:
                arr = cg[f"metrics.{m}"].astype(float).to_numpy()
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    mvals[m] = {"mean": float(arr.mean()),
                                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                                "n": int(arr.size)}
            params = {}
            for c in (swept or param_cols):
                v = next((x for x in cg[c].tolist() if x is not None and str(x) != "nan"), None)
                if v is not None:
                    params[c.split(".", 1)[1]] = v
            configs.append({"config_hash": h, "n_runs": len(cg),
                            "params": params, "metrics": mvals})

        # best config by primary metric mean
        best = None
        if primary:
            ranked = [c for c in configs if primary in c["metrics"]]
            if ranked:
                best = max(ranked, key=lambda c: c["metrics"][primary]["mean"])
        experiments.append({
            "exp_id": eid, "reason": reason, "n_runs": len(g),
            "primary_metric": primary, "metrics_present": present,
            "swept_params": [c.split(".", 1)[1] for c in swept],
            "configs": configs,
            "best_config_hash": best["config_hash"] if best else None,
        })
    return experiments


def _score_metrics(exp):
    """Accuracy-like metrics: every value present is within [0, 1]. This keeps
    test_acc / majority_acc / acc_embedding_knn etc. and naturally drops count
    diagnostics like avg_edges_knn (which are >1), so within-run comparisons
    (e.g. l9's three graph constructions) all show up as bars."""
    out = []
    for m in exp["metrics_present"]:
        vals = [c["metrics"][m]["mean"] for c in exp["configs"] if m in c["metrics"]]
        if vals and all(0.0 <= v <= 1.0 for v in vals):
            out.append(m)
    return out or ([exp["primary_metric"]] if exp["primary_metric"] else [])


def chart(exp, fig_path):
    """Grouped bar chart: every accuracy-like metric, one group of bars per
    config, seed error bars when n>1. For a single config this becomes one bar
    per metric — exactly the within-run comparison (l9 graph types, l3 vs
    majority baseline). The primary metric's bar is outlined so it stands out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = _score_metrics(exp)
    cfgs = exp["configs"]
    if not metrics or not cfgs:
        return False

    def clabel(c):
        if c["params"]:
            return ", ".join(f"{k}={v}" for k, v in list(c["params"].items())[:2])
        return c["config_hash"][:6]

    n_m, n_c = len(metrics), len(cfgs)
    x = np.arange(n_m)
    width = 0.8 / n_c
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * n_m * n_c), 4.4))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(cfgs):
        means = [c["metrics"].get(m, {}).get("mean", np.nan) for m in metrics]
        errs = [c["metrics"].get(m, {}).get("std", 0.0) for m in metrics]
        off = (ci - (n_c - 1) / 2) * width
        bars = ax.bar(x + off, means, width, yerr=errs, capsize=3,
                      label=clabel(c) if n_c > 1 else None, color=cmap(ci % 10),
                      edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    # mark the primary metric on the x tick so the headline number is obvious
    ticks = [(m + "  ★" if m == exp["primary_metric"] else m) for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", c="k", lw=0.7)
    ax.set_title(f"{exp['exp_id']} — {exp['n_runs']} run(s) · ★ = primary metric")
    if n_c > 1:
        ax.legend(fontsize=7, title="config")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    return True


def render_md(campaign, experiments, fig_dir, generated):
    total_runs = sum(e["n_runs"] for e in experiments)
    L = []
    L.append(f"# Campaign report — `{campaign}`\n")
    L.append(f"_{total_runs} runs across {len(experiments)} experiment(s)_ · "
             f"generated {generated}\n")
    L.append("## Experiments\n")
    for e in experiments:
        rw = "run" if e["n_runs"] == 1 else "runs"
        L.append(f"- [`{e['exp_id']}`](#{e['exp_id'].lower().replace('_','-')}) — "
                 f"{e['n_runs']} {rw}, primary `{e['primary_metric']}`")
    L.append("")

    for e in experiments:
        L.append(f"## {e['exp_id']}\n")
        L.append(f"**Why:** {e['reason'] or '_(no reason recorded)_'}\n")
        rw = "run" if e["n_runs"] == 1 else "runs"
        L.append(f"_{e['n_runs']} {rw} · primary metric_ `{e['primary_metric']}` · "
                 f"_swept_ {', '.join('`'+p+'`' for p in e['swept_params']) or '(none)'}\n")
        # forward slashes so the link renders on GitHub / any markdown viewer
        fig_rel = f"{os.path.basename(fig_dir)}/{e['exp_id']}.png"
        if e.get("_has_chart"):
            L.append(f"![{e['exp_id']} chart]({fig_rel})\n")

        # config table: swept params + every metric (mean±std)
        mnames = e["metrics_present"]
        head = ["config", "n"] + e["swept_params"] + mnames
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "|".join(["---"] * len(head)) + "|")
        for c in e["configs"]:
            star = " ⬅" if c["config_hash"] == e["best_config_hash"] else ""
            row = [f"`{c['config_hash'][:8]}`{star}", str(c["n_runs"])]
            row += [str(c["params"].get(p, "—")) for p in e["swept_params"]]
            for m in mnames:
                if m in c["metrics"]:
                    mm = c["metrics"][m]
                    row.append(fmt(mm["mean"]) + (f" ±{fmt(mm['std'])}" if mm["n"] > 1 else ""))
                else:
                    row.append("—")
            L.append("| " + " | ".join(row) + " |")
        L.append("")

        if e["best_config_hash"]:
            best = next(c for c in e["configs"] if c["config_hash"] == e["best_config_hash"])
            bm = best["metrics"][e["primary_metric"]]
            cfg_desc = ", ".join(f"{k}={v}" for k, v in best["params"].items()) or best["config_hash"][:8]
            L.append(f"**Best config:** `{cfg_desc}` → {e['primary_metric']} "
                     f"= {fmt(bm['mean'])}" + (f" ±{fmt(bm['std'])}" if bm["n"] > 1 else "") + "\n")
    return "\n".join(L)


def interpret(summary, md_path):
    """Shell out to the claude CLI to write the Interpretation section."""
    claude = shutil.which("claude")
    if not claude:
        return "_`--interpret` requested but the `claude` CLI was not found on PATH; skipped._"
    prompt = (
        "You are a senior ML scientist reading the results of an experiment "
        "campaign. Below is a JSON summary (per experiment: the plain-English "
        "reason it was run, the primary metric, each config's params and "
        "metric mean±std over seeds).\n\n"
        "Write a concise interpretation in GitHub markdown (no preamble, no "
        "headline — start straight at bullet points). For EACH experiment: "
        "state what the numbers say, name the winning config, and judge "
        "honestly whether the gap is real given the run count / std / N (small "
        "N and overlapping error bars = 'no signal'). End with one overall "
        "takeaway. Keep it under ~250 words.\n\n"
        "JSON summary:\n```json\n" + json.dumps(summary, indent=1) + "\n```\n"
    )
    try:
        # Force UTF-8 decode: claude emits em-dashes / ± and the Windows locale
        # codec (cp1252) mangles them into mojibake otherwise.
        out = subprocess.run([claude, "-p", prompt], capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=300)
    except Exception as exc:  # noqa: BLE001
        return f"_`--interpret` failed to run the claude CLI: {exc}_"
    if out.returncode != 0:
        return f"_`--interpret` claude CLI exited {out.returncode}: {out.stderr.strip()[:300]}_"
    return out.stdout.strip() or "_(claude returned no output)_"


def main():
    ap = argparse.ArgumentParser(description="Campaign-level MLflow report")
    ap.add_argument("--campaign", required=True, help="campaign tag to report on")
    ap.add_argument("--interpret", action="store_true",
                    help="also write an AI interpretation via the claude CLI")
    ap.add_argument("--out", default=None, help="output .md path (auto if omitted)")
    args = ap.parse_args()

    import datetime
    df = load_campaign(args.campaign)
    if df is None or df.empty:
        sys.exit(f"No runs with tags.campaign = '{args.campaign}' in the MLflow store.")
    if "tags.exp_id" not in df.columns:
        sys.exit("Runs found but none carry a tags.exp_id — were they run via the driver?")

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out_dir = os.path.join(HERE, "reports")
    fig_dir = os.path.join(out_dir, f"figs_{args.campaign}")
    os.makedirs(fig_dir, exist_ok=True)
    out_md = args.out or os.path.join(out_dir, f"report_{args.campaign}.md")
    out_json = os.path.join(out_dir, f"summary_{args.campaign}.json")

    experiments = summarise(df)
    for e in experiments:
        e["_has_chart"] = chart(e, os.path.join(fig_dir, f"{e['exp_id']}.png"))

    summary = {"campaign": args.campaign, "generated": generated,
               "n_runs": int(len(df)),
               "experiments": [{k: v for k, v in e.items() if not k.startswith("_")}
                               for e in experiments]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    md = render_md(args.campaign, experiments, fig_dir, generated)
    if args.interpret:
        md += "\n## Interpretation\n\n" + interpret(summary, out_md) + "\n"
    else:
        md += ("\n## Interpretation\n\n_Run with `--interpret` to have the "
               "`claude` CLI read the summary and write this section._\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print(f"charts -> {fig_dir}")
    print(f"runs: {len(df)}  experiments: {len(experiments)}  "
          f"({', '.join(e['exp_id'] for e in experiments)})")


if __name__ == "__main__":
    main()
