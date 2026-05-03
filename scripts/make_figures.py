"""Generate the headline figures for the presentation:

  - F1-macro comparison bar chart (per dataset)
  - Confusion matrix heatmaps (one per run, plus an aggregated key set)
  - Coverage chart for LLM runs

Outputs to presentation/key_figures/ as PNGs at 150dpi.

Usage: .venv/Scripts/python.exe scripts/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")  # no display server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "presentation", "key_figures")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_ORDER = ["negative", "neutral", "positive"]


def _load_table() -> pd.DataFrame | None:
    path = os.path.join(ROOT, "results", "summary", "final_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def f1_comparison(df: pd.DataFrame) -> None:
    """Headline figure: F1-macro per (model, dataset). Best config per model."""
    # Pick best (template, shots) per (model, dataset) by f1_macro.
    best = df.sort_values("f1_macro", ascending=False).groupby(
        ["model", "dataset"], as_index=False
    ).first()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=best, x="model", y="f1_macro", hue="dataset",
        order=sorted(best["model"].unique()), ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-macro (best config per model)")
    ax.set_xlabel("")
    ax.set_title("Sentiment classification — F1-macro by model")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=9, padding=2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "f1_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def coverage_chart(df: pd.DataFrame) -> None:
    """LLM-only coverage (parsing success rate)."""
    llm = df[df["model"].isin(["plutus8b", "mistral7b", "qwen25_7b", "finllama", "llama31"])]
    if llm.empty:
        return
    pivot = llm.pivot_table(index=["model", "dataset"], columns=["template", "shots"],
                            values="coverage", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("LLM parsing coverage")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "coverage_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def confusion_grid(df: pd.DataFrame) -> None:
    """Confusion matrices for the best run of each model on each dataset."""
    best = df.sort_values("f1_macro", ascending=False).groupby(
        ["model", "dataset"], as_index=False
    ).first()
    conf_dir = os.path.join(ROOT, "results", "summary", "confusions")
    if not os.path.isdir(conf_dir):
        return

    # Pull predictions back from per-run dirs to render confusions in fixed label order
    pred_root = os.path.join(ROOT, "results", "predictions")

    models = sorted(best["model"].unique())
    datasets = sorted(best["dataset"].unique())
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(3.4 * len(datasets), 3.4 * len(models)),
                             squeeze=False)

    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            row = best[(best["model"] == model) & (best["dataset"] == ds)]
            ax = axes[i][j]
            if row.empty:
                ax.set_axis_off()
                continue
            tpl = row.iloc[0]["template"]
            shots = int(row.iloc[0]["shots"]) if pd.notna(row.iloc[0]["shots"]) else 0
            seed = int(row.iloc[0]["seed"])
            if tpl == "-":
                run_id = f"{model}__{ds}__seed{seed}"
            else:
                run_id = f"{model}__{ds}__{tpl}__{shots}shot__seed{seed}"

            cf_path = os.path.join(conf_dir, f"{run_id}.json")
            if not os.path.exists(cf_path):
                ax.set_axis_off()
                continue
            with open(cf_path) as f:
                cf = json.load(f)
            labels = cf.get("labels", [])
            mat = np.array(cf["matrix"])

            # Reorder to negative/neutral/positive
            idx = [labels.index(l) if l in labels else -1 for l in LABEL_ORDER]
            ordered = np.zeros((3, 3), dtype=int)
            for ii, src in enumerate(idx):
                for jj, dst in enumerate(idx):
                    if src >= 0 and dst >= 0:
                        ordered[ii][jj] = mat[src][dst]

            sns.heatmap(ordered, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["neg", "neu", "pos"],
                        yticklabels=["neg", "neu", "pos"],
                        cbar=False, ax=ax)
            ax.set_title(f"{model} · {ds}\nf1m={row.iloc[0]['f1_macro']:.2f}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "confusion_grid.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def per_class_f1(df: pd.DataFrame) -> None:
    """Heatmap of per-class F1 — best config per (model, dataset)."""
    conf_dir = os.path.join(ROOT, "results", "summary", "confusions")
    if not os.path.isdir(conf_dir):
        return
    best = df.sort_values("f1_macro", ascending=False).groupby(
        ["model", "dataset"], as_index=False
    ).first()

    rows = []
    for _, r in best.iterrows():
        if r["template"] == "-":
            run_id = f"{r['model']}__{r['dataset']}__seed{int(r['seed'])}"
        else:
            run_id = f"{r['model']}__{r['dataset']}__{r['template']}__{int(r['shots'])}shot__seed{int(r['seed'])}"
        cf_path = os.path.join(conf_dir, f"{run_id}.json")
        if not os.path.exists(cf_path):
            continue
        with open(cf_path) as f:
            cf = json.load(f)
        for cls, m in cf.get("per_class", {}).items():
            rows.append({
                "model": r["model"], "dataset": r["dataset"],
                "class": cls, "f1": m["f1"],
            })
    if not rows:
        return

    pcdf = pd.DataFrame(rows)
    for ds in pcdf["dataset"].unique():
        sub = pcdf[pcdf["dataset"] == ds]
        pivot = sub.pivot(index="model", columns="class", values="f1")
        col_order = [c for c in LABEL_ORDER if c in pivot.columns]
        pivot = pivot[col_order]
        fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.5 * len(pivot))))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, cbar_kws={"label": "F1"}, ax=ax,
        )
        ax.set_title(f"Per-class F1 (best config) — {ds}")
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"per_class_f1_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


def fewshot_effect(df: pd.DataFrame) -> None:
    """Bar chart of Δ F1-macro between 0-shot and 3-shot per (model, dataset, template)."""
    llm = df[df["model"].isin(["plutus8b", "mistral7b", "qwen25_7b", "finllama"])]
    if llm.empty:
        return
    pivot = llm.pivot_table(
        index=["model", "dataset", "template"],
        columns="shots",
        values="f1_macro",
    ).dropna(how="any")
    if not {0, 3}.issubset(pivot.columns):
        return
    pivot["delta"] = pivot[3] - pivot[0]
    pivot = pivot.reset_index()
    pivot["row_label"] = pivot.apply(
        lambda r: f"{r['model']} · {r['dataset']} · tpl {r['template']}", axis=1
    )

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    colors = ["#3a8c4f" if d > 0 else "#c44a4a" for d in pivot["delta"]]
    ax.barh(pivot["row_label"], pivot["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔF1-macro (3-shot − 0-shot)")
    ax.set_title("Few-shot effect: gain vs. loss when adding 3 examples")
    for i, (lbl, d) in enumerate(zip(pivot["row_label"], pivot["delta"])):
        ax.text(d, i, f"  {d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fewshot_effect.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    df = _load_table()
    if df is None:
        print("no final_table.csv yet; run aggregate.py first")
        return
    sns.set_theme(style="whitegrid")
    f1_comparison(df)
    coverage_chart(df)
    confusion_grid(df)
    per_class_f1(df)
    fewshot_effect(df)


if __name__ == "__main__":
    main()
