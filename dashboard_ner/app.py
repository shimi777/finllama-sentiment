"""Streamlit dashboard for the NER comparison benchmark — separate app.

Reads from:
    results/summary_ner/final_table_ner.csv
    results/summary_ner/confusions/{run_id}.json
    results/predictions/{run_id}/predictions.jsonl
    results/_ner_spend.json

Run:
    .venv/Scripts/python.exe -m streamlit run dashboard_ner/app.py

Designed to be entirely independent of dashboard/app.py — different data files,
different metrics, different schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENTITY_TYPES = ["PER", "LOC", "ORG"]


# --- Pretty model names + family classification ---
MODEL_LABEL = {
    # Local NER baselines
    "gliner-large":          "GLiNER-large-v2.1 (local)",
    "gliner-medium":         "GLiNER-medium-v2.1 (local)",
    "gliner-small":          "GLiNER-small-v2.1 (local)",
    "nuner-zero":            "NuNER-Zero (local)",
    # NER-via-API LLMs
    "gpt-4.1-nano":          "GPT-4.1-nano",
    "gpt-4.1-mini":          "GPT-4.1-mini",
    "gpt-4o-mini":           "GPT-4o-mini",
    "gpt-4o":                "GPT-4o",
    "claude-haiku-4-5":      "Claude Haiku 4.5",
    "claude-sonnet-4-6":     "Claude Sonnet 4.6",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "gemini-2.5-flash":      "Gemini 2.5 Flash",
    # Modal-hosted 7-9B LLMs (article + modern)
    "qwen25_7b":             "Qwen2.5-7B-Instruct (article)",
    "mistral7b":             "Mistral-7B-Instruct-v0.3 (article)",
    "plutus8b":              "Plutus-8B-instruct (article, financial)",
    "qwen3_8b":              "Qwen3-8B (modern)",
    "qwen3_4b":              "Qwen3-4B-Instruct (modern)",
    "gemma2_9b":             "Gemma-2-9B-it (modern)",
    "gemma2_2b":             "Gemma-2-2B-it (modern)",
    "llama31_8b":            "Llama-3.1-8B-Instruct (article)",
    "finma7b":               "FinMA-7B-full (article, financial)",
}

# Tier classification — drives the color/grouping in charts.
MODEL_TIER = {
    "gliner-large": "local", "gliner-medium": "local", "gliner-small": "local", "nuner-zero": "local",
    "gpt-4.1-nano": "api", "gpt-4.1-mini": "api", "gpt-4o-mini": "api", "gpt-4o": "api",
    "claude-haiku-4-5": "api", "claude-sonnet-4-6": "api",
    "gemini-2.5-flash-lite": "api", "gemini-2.5-flash": "api",
    "qwen25_7b": "article-llm", "mistral7b": "article-llm", "plutus8b": "article-llm",
    "llama31_8b": "article-llm", "finma7b": "article-llm",
    "qwen3_8b": "modern-llm", "qwen3_4b": "modern-llm",
    "gemma2_9b": "modern-llm", "gemma2_2b": "modern-llm",
}
TIER_COLOR = {
    "local":       "#7fb3d5",   # blue
    "api":         "#f5b041",   # orange
    "article-llm": "#a569bd",   # purple
    "modern-llm":  "#52be80",   # green
}
# Readable tier names — used in legends, the leaderboard, and verdict cards
# so the UI never shows raw slugs like "article-llm".
TIER_LABEL = {
    "local":       "Local NER",
    "api":         "API LLM",
    "article-llm": "Article LLM (7-9B)",
    "modern-llm":  "Modern LLM (7-9B)",
}


# ---------- data loaders (cached) ----------

@st.cache_data
def load_table() -> pd.DataFrame | None:
    p = ROOT / "results" / "summary_ner" / "final_table_ner.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model_label"] = df["model"].map(MODEL_LABEL).fillna(df["model"])
    df["tier"] = df["model"].map(MODEL_TIER).fillna("api")
    df["cost_per_100"] = df.apply(
        lambda r: (r["cost_usd"] / r["n_samples"] * 100.0) if r["n_samples"] else 0.0,
        axis=1,
    )
    return df


@st.cache_data
def load_confusions() -> dict[str, dict]:
    cdir = ROOT / "results" / "summary_ner" / "confusions"
    out: dict[str, dict] = {}
    if not cdir.exists():
        return out
    for f in cdir.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return out


@st.cache_data
def load_spend() -> dict | None:
    p = ROOT / "results" / "_ner_spend.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data
def load_predictions(run_id: str) -> pd.DataFrame | None:
    p = ROOT / "results" / "predictions" / run_id / "predictions.jsonl"
    if not p.exists():
        return None
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


# ---------- main app ----------

st.set_page_config(
    page_title="Financial NER Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial NER — Modern Models Comparison")
st.caption(
    "FiNER-ORD test set · entity types: PER / LOC / ORG · "
    "strict span-F1 via seqeval IOB2 · partial-F1 with overlap+type · "
    "cost tracked per call."
)

df = load_table()
if df is None or df.empty:
    st.warning(
        "No NER results found yet. To populate:\n\n"
        "```\n"
        "py scripts/run_ner.py --dry-run   # see plan + cost estimate\n"
        "py scripts/run_ner.py             # local + cheap APIs (if keys set)\n"
        "py scripts/aggregate_ner.py       # build final_table_ner.csv\n"
        "```"
    )
    st.stop()

confusions = load_confusions()
spend = load_spend()


# ---------- sidebar ----------

st.sidebar.header("Filters")
all_models = sorted(df["model"].unique().tolist())
all_templates = sorted(df["template"].unique().tolist())
sel_models = st.sidebar.multiselect("Models", all_models, default=all_models)
sel_templates = st.sidebar.multiselect("Templates", all_templates, default=all_templates)

filt = df[df["model"].isin(sel_models) & df["template"].isin(sel_templates)].copy()

st.sidebar.divider()
st.sidebar.subheader("API spend")
if spend:
    cap = spend.get("cap_usd", 0.0)
    cum = spend.get("cumulative_usd", 0.0)
    rem = max(0.0, cap - cum)
    st.sidebar.metric("Cumulative", f"${cum:.4f}", delta=f"−${cum:.4f} of ${cap:.2f} cap" if cap else None)
    st.sidebar.metric("Remaining", f"${rem:.4f}")
    st.sidebar.metric("Total API calls", spend.get("n_calls", 0))
    if spend.get("by_model"):
        sp = pd.DataFrame([
            {"model": k, "spend_usd": v} for k, v in spend["by_model"].items()
        ]).sort_values("spend_usd", ascending=False)
        st.sidebar.dataframe(sp, hide_index=True, use_container_width=True)
else:
    st.sidebar.info("No API calls recorded yet.")


# ---------- top metrics ----------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runs", len(filt))
if not filt.empty:
    top = filt.loc[filt["strict_f1"].idxmax()]
    c2.metric("Best strict-F1", f"{top['strict_f1']:.3f}", top["model_label"])
    c3.metric("Total cost (filtered)", f"${filt['cost_usd'].sum():.4f}")
    cheap = filt[filt["strict_f1"] > 0].copy()
    if not cheap.empty:
        # Best "bang per buck" — strict_f1 per $0.01 spent (or just F1 if free).
        cheap["bang"] = cheap.apply(
            lambda r: r["strict_f1"] if r["cost_usd"] == 0 else r["strict_f1"] / (r["cost_usd"] + 1e-9),
            axis=1,
        )
        best_value = cheap.loc[cheap["bang"].idxmax()]
        c4.metric(
            "Best value",
            best_value["model_label"],
            f"F1={best_value['strict_f1']:.2f} · ${best_value['cost_usd']:.4f}",
        )


# ---------- leaderboard: head-to-head model comparison ----------

st.subheader("🏆 Model leaderboard")
st.caption(
    "One row per model (its single best run), ranked by **strict span-F1** — the "
    "strictest metric: exact boundary *and* correct type. Green = better. "
    "Per-type columns (PER / LOC / ORG) show where each model is strong or weak; "
    "cost is per 100 examples."
)
if not filt.empty:
    # Collapse to the best-scoring run per model so the ranking is unambiguous.
    board = (
        filt.sort_values("strict_f1", ascending=False)
        .groupby("model", as_index=False)
        .first()
        .sort_values("strict_f1", ascending=False)
        .reset_index(drop=True)
    )
    board.insert(0, "rank", board.index + 1)
    board["type"] = board["tier"].map(TIER_LABEL).fillna(board["tier"])
    board["config"] = board.apply(
        lambda r: "—" if r["backend"] == "local"
        else f"tmpl {r['template']} · {int(r['shots'])}-shot",
        axis=1,
    )
    lb_rename = {
        "rank": "#",
        "model_label": "Model",
        "type": "Type",
        "config": "Config",
        "strict_f1": "Strict-F1",
        "partial_f1": "Partial-F1",
        "f1_PER": "PER",
        "f1_LOC": "LOC",
        "f1_ORG": "ORG",
        "coverage": "Coverage",
        "cost_per_100": "Cost/100",
    }
    lb_present = [c for c in lb_rename if c in board.columns]
    show = board[lb_present].rename(columns=lb_rename)
    grad_cols = [c for c in ["Strict-F1", "Partial-F1", "PER", "LOC", "ORG"] if c in show.columns]
    styler = (
        show.style
        .format({
            "Strict-F1": "{:.3f}", "Partial-F1": "{:.3f}",
            "PER": "{:.3f}", "LOC": "{:.3f}", "ORG": "{:.3f}",
            "Coverage": "{:.0%}", "Cost/100": "${:.4f}",
        })
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, subset=grad_cols)
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


# ---------- Article vs Modern verdict card ----------

st.subheader("Article-models vs Modern-models on FiNER-ORD")
st.caption(
    "Best strict-F1 within each tier — answers the research question: "
    "did the financial fine-tuning generation hold up against newer general models?"
)
verdict = filt.copy()
if not verdict.empty:
    best_by_tier = verdict.sort_values("strict_f1", ascending=False).groupby("tier").head(1)
    tier_order = ["article-llm", "modern-llm", "local", "api"]
    cols = st.columns(min(4, len(best_by_tier)))
    i = 0
    for tier in tier_order:
        row = best_by_tier[best_by_tier["tier"] == tier]
        if row.empty:
            continue
        row = row.iloc[0]
        with cols[i % len(cols)]:
            tier_pretty = {
                "article-llm": "Article LLMs",
                "modern-llm":  "Modern LLMs",
                "local":       "Local NER",
                "api":         "API LLMs",
            }.get(tier, tier)
            st.metric(
                f"Best {tier_pretty}",
                f"F1 {row['strict_f1']:.3f}",
                row["model_label"],
            )
        i += 1


# ---------- F1 bar chart ----------

st.subheader("Strict span-F1 by model")
if not filt.empty:
    chart_df = filt.copy()
    chart_df["cfg"] = chart_df.apply(
        lambda r: "" if r["backend"] == "local"
        else f" · tmpl {r['template']} · {int(r['shots'])}-shot",
        axis=1,
    )
    chart_df["run_label"] = chart_df["model_label"] + chart_df["cfg"]
    chart_df = chart_df.sort_values("strict_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(chart_df))))
    colors = [TIER_COLOR.get(t, "#bdc3c7") for t in chart_df["tier"]]
    ax.barh(chart_df["run_label"], chart_df["strict_f1"], color=colors)
    ax.set_xlabel("Strict span-F1")
    ax.set_xlim(0, 1)
    for i, v in enumerate(chart_df["strict_f1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=TIER_LABEL.get(t, t)) for t, c in TIER_COLOR.items()
                      if t in chart_df["tier"].unique()]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    st.pyplot(fig, clear_figure=True)


# ---------- cost vs F1 scatter ----------

st.subheader("Cost vs accuracy frontier")
st.caption("Cheaper-and-better is better — bottom-right is the Pareto front.")
if not filt.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    marker_for_tier = {"local": "s", "api": "o", "article-llm": "^", "modern-llm": "D"}
    for _, row in filt.iterrows():
        color = TIER_COLOR.get(row["tier"], "#bdc3c7")
        marker = marker_for_tier.get(row["tier"], "o")
        ax.scatter(row["cost_per_100"], row["strict_f1"], c=color, marker=marker, s=80, alpha=0.85)
        ax.annotate(
            row["model_label"], (row["cost_per_100"], row["strict_f1"]),
            xytext=(5, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Cost per 100 examples (USD)")
    ax.set_ylabel("Strict span-F1")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    if filt["cost_per_100"].max() > 0:
        ax.set_xscale("symlog", linthresh=0.001)
    st.pyplot(fig, clear_figure=True)


# ---------- per-type breakdown ----------

st.subheader("Per-entity-type F1 (strict)")
if not filt.empty:
    long = filt.melt(
        id_vars=["model_label", "template", "shots", "backend"],
        value_vars=["f1_PER", "f1_LOC", "f1_ORG"],
        var_name="entity_type", value_name="f1",
    )
    long["entity_type"] = long["entity_type"].str.replace("f1_", "")
    pivot = long.pivot_table(
        index="model_label", columns="entity_type", values="f1", aggfunc="mean",
    ).reindex(columns=ENTITY_TYPES)
    st.dataframe(pivot.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                 use_container_width=True)


# ---------- full results table ----------

st.subheader("All runs")
show_cols = [
    "model_label", "backend", "template", "shots", "n_samples", "coverage",
    "strict_f1", "partial_f1", "type_only_f1",
    "f1_PER", "f1_LOC", "f1_ORG",
    "avg_latency_ms", "input_tokens", "output_tokens",
    "cost_usd", "cost_per_100",
]
present = [c for c in show_cols if c in filt.columns]
st.dataframe(
    filt[present].sort_values("strict_f1", ascending=False),
    hide_index=True, use_container_width=True,
)


# ---------- cross-type confusion ----------

st.subheader("Cross-type confusion (true → predicted)")
st.caption("Counts where predicted span overlaps a gold span. MISS = gold span the model didn't tag at all.")

run_ids_in_filter = filt["model"].astype(str) + "__FiNER-ORD"
available_confs = [k for k in confusions if any(k.startswith(rid) for rid in run_ids_in_filter)]
if not available_confs:
    st.info("No confusion data for the selected filter.")
else:
    n_per_row = 3
    rows_cm = [available_confs[i : i + n_per_row] for i in range(0, len(available_confs), n_per_row)]
    for row in rows_cm:
        cols = st.columns(len(row))
        for col, rid in zip(cols, row):
            cm_dict = confusions[rid].get("confusion_by_type", {}).get("matrix", {})
            if not cm_dict:
                col.write(rid + " (no confusion)")
                continue
            cols_order = ENTITY_TYPES + ["MISS"]
            mat = np.array([
                [cm_dict.get(t, {}).get(p, 0) for p in cols_order] for t in ENTITY_TYPES
            ])
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                        xticklabels=cols_order, yticklabels=ENTITY_TYPES,
                        cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(rid, fontsize=8)
            col.pyplot(fig, clear_figure=True)


# ---------- per-example drill-down ----------

st.subheader("Per-example drill-down")
# List run dirs directly from disk — robust to varying run-id schemas
# (local runs have no template/shots in the id; Modal/API runs do).
pred_root_dir = ROOT / "results" / "predictions"
all_run_dirs = sorted([
    p.name for p in pred_root_dir.glob("*FiNER-ORD*")
    if (p / "predictions.jsonl").exists()
]) if pred_root_dir.exists() else []

sel_run = st.selectbox("Pick a run to inspect", all_run_dirs) if all_run_dirs else None
preds = load_predictions(sel_run) if sel_run else None
if preds is not None and not preds.empty:
    st.caption(f"{len(preds)} predictions in this run.")
    show_only = st.radio("Filter", ["All", "Parse failures only", "Has entities"], horizontal=True)
    view = preds.copy()
    if show_only == "Parse failures only":
        view = view[view["parse_ok"] == False]  # noqa: E712
    elif show_only == "Has entities":
        view = view[view["pred_entities"].apply(lambda x: bool(x))]
    cols_to_show = [c for c in ["id", "parse_ok", "raw_output", "latency_ms", "cost_usd"] if c in view.columns]
    st.dataframe(view[cols_to_show].head(200), hide_index=True, use_container_width=True)
else:
    st.info("No predictions found for that run id.")


st.divider()
st.caption(
    "Strict F1 = exact span+type match · Partial F1 = same type with token overlap · "
    "MISS = gold span the model missed entirely. Backed by seqeval (strict IOB2)."
)
