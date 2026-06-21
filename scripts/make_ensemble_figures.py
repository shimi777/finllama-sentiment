"""Figures for the prompt-ensemble improvement (reads results/summary/ensemble_table.csv).

  - ensemble_vs_single.png : per (dataset, model), single-prompt mean F1 with a
    whisker spanning the worst..best single prompt (the "prompt lottery"), next to
    the unweighted and cv-weighted ensemble. Shows the ensemble killing variance.
  - ensemble_coverage_tradeoff.png : mean F1 vs mean coverage for each aggregation
    policy. Shows cv-weighted soft voting is the best *full-coverage* aggregator.

Outputs to presentation/key_figures/ at 150dpi.

Usage: .venv/Scripts/python.exe scripts/make_ensemble_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "presentation", "key_figures")
os.makedirs(OUT_DIR, exist_ok=True)

ABSTAIN = "abstain"


def _load() -> pd.DataFrame | None:
    path = os.path.join(ROOT, "results", "summary", "ensemble_table.csv")
    if not os.path.exists(path):
        print("no ensemble_table.csv; run aggregate_ensemble.py first")
        return None
    return pd.read_csv(path)


def ensemble_vs_single(df: pd.DataFrame) -> None:
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8), squeeze=False)

    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        cell = df[(df["dataset"] == ds) & (df["tie_break"] == ABSTAIN)]
        models = sorted(cell["model"].unique())
        x = np.arange(len(models))
        w = 0.26

        mean_f1, worst_f1, best_f1, unw, cvw = [], [], [], [], []
        for mdl in models:
            sub = cell[cell["model"] == mdl]
            r0 = sub.iloc[0]
            mean_f1.append(r0["single_mean_f1"])
            worst_f1.append(r0["single_worst_f1"])
            best_f1.append(r0["single_best_f1"])
            unw.append(sub[sub["method"] == "unweighted"]["ens_f1"].iloc[0])
            cvw.append(sub[sub["method"] == "cv_weighted"]["ens_f1"].iloc[0])

        mean_f1 = np.array(mean_f1); worst_f1 = np.array(worst_f1); best_f1 = np.array(best_f1)
        yerr = np.vstack([mean_f1 - worst_f1, best_f1 - mean_f1])

        ax.bar(x - w, mean_f1, w, color="#bdbdbd", label="single prompt (mean)")
        ax.errorbar(x - w, mean_f1, yerr=yerr, fmt="none", ecolor="#404040",
                    capsize=5, lw=1.2, label="single worst..best")
        ax.bar(x, unw, w, color="#4c78a8", label="ensemble (majority)")
        ax.bar(x + w, cvw, w, color="#2e8b57", label="ensemble (cv-weighted)")
        ax.scatter(x - w, best_f1, marker="*", s=90, color="#d62728", zorder=5,
                   label="best single (oracle)")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-macro")
        ax.set_title(f"{ds}: ensemble vs the prompt lottery")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "ensemble_vs_single.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def coverage_tradeoff(df: pd.DataFrame) -> None:
    """Mean F1 vs mean coverage per aggregation policy (averaged over the 6 cells)."""
    policies = [
        ("unweighted", "abstain", "majority / abstain", "#4c78a8"),
        ("unweighted", "order", "majority / order", "#9ecae1"),
        ("unweighted", "neutral", "majority / neutral", "#c6dbef"),
        ("cv_weighted", "abstain", "cv-weighted (leakage-free)", "#2e8b57"),
        ("oracle_weighted", "abstain", "oracle-weighted (ceiling)", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, tb, label, color in policies:
        sub = df[(df["method"] == method) & (df["tie_break"] == tb)]
        if sub.empty:
            continue
        f1, cov = sub["ens_f1"].mean(), sub["ens_cov"].mean()
        marker = "D" if method == "cv_weighted" else ("*" if method == "oracle_weighted" else "o")
        size = 220 if method in ("cv_weighted", "oracle_weighted") else 120
        ax.scatter(cov, f1, s=size, color=color, marker=marker, edgecolor="black",
                   linewidth=0.6, zorder=5, label=label)
        ax.annotate(f"  {label}\n  F1={f1:.3f}, cov={cov:.3f}", (cov, f1),
                    fontsize=8, va="center")

    ax.set_xlabel("coverage (fraction of examples labeled)")
    ax.set_ylabel("mean F1-macro (over 6 cells)")
    ax.set_title("Aggregation policies: F1 vs coverage\ncv-weighted = best full-coverage aggregator")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.25)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ensemble_coverage_tradeoff.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    df = _load()
    if df is None:
        return
    sns.set_theme(style="whitegrid")
    ensemble_vs_single(df)
    coverage_tradeoff(df)


if __name__ == "__main__":
    main()
