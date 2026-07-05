"""Streamlit dashboard for the FinLLaMA-vs-baselines sentiment benchmark + NER analysis.

Run:
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py

Top-level structure: two big tabs — Sentiment Analysis (LLMs vs baselines on FPB+FiQA)
and Named Entity Recognition (entity extraction on the same texts). Each tab has its
own sub-tabs and scoped selectors so the views stay focused.

Reads from results/summary/final_table.csv, results/summary/confusions/,
results/summary/focal_error_sample.csv, results/summary/ner_summary.json,
results/summary/ner_top_entities.csv, and per-run dirs under results/predictions/.
Re-run scripts/aggregate.py + scripts/run_finbert_ner.py to refresh the data, then
hit "Reload predictions" or hard-refresh the browser.
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

LABEL_ORDER = ["negative", "neutral", "positive"]

# --- Pretty model names ---
MODEL_LABEL = {
    "vader": "VADER (lexicon)",
    "finbert": "FinBERT (ProsusAI, 110M)",
    "finbert_tone": "FinBERT-tone (analyst tone, 110M)",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral-7B-Instruct-v0.3",
    "plutus8b": "plutus-8B-instruct (TheFinAI)",
    "finllama": "FinLLaMA-instruct",
}

# Model family colors for consistent plotting
MODEL_FAMILY = {
    "vader": "Lexicon",
    "finbert": "BERT-class",
    "finbert_tone": "BERT-class",
    "qwen25_7b": "LLM (general)",
    "mistral7b": "LLM (general)",
    "plutus8b": "LLM (financial)",
    "finllama": "LLM (financial)",
}
FAMILY_PALETTE = {
    "Lexicon": "#9aa0a6",
    "BERT-class": "#1976d2",
    "LLM (general)": "#7b4fb3",
    "LLM (financial)": "#ef6c00",
}


# ----------------- Data loaders (cached) -----------------

@st.cache_data
def load_table() -> pd.DataFrame | None:
    p = ROOT / "results" / "summary" / "final_table.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model_label"] = df["model"].map(MODEL_LABEL).fillna(df["model"])
    df["family"] = df["model"].map(MODEL_FAMILY).fillna("Other")
    return df


@st.cache_data
def load_confusions() -> dict[str, dict]:
    cdir = ROOT / "results" / "summary" / "confusions"
    out: dict[str, dict] = {}
    if not cdir.exists():
        return out
    for f in cdir.glob("*.json"):
        with open(f) as fh:
            out[f.stem] = json.load(fh)
    return out


@st.cache_data
def load_errors() -> pd.DataFrame | None:
    p = ROOT / "results" / "summary" / "focal_error_sample.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_modal_spend() -> dict | None:
    p = ROOT / "results" / "_modal_spend.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_predictions(run_id: str) -> pd.DataFrame | None:
    p = ROOT / "results" / "predictions" / run_id / "predictions.jsonl"
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data
def load_all_predictions() -> pd.DataFrame:
    """Long-format: one row per (run_id, sample_id) with prediction + run metadata."""
    pred_root = ROOT / "results" / "predictions"
    if not pred_root.exists():
        return pd.DataFrame()
    rows = []
    for run_dir in pred_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_p = run_dir / "meta.json"
        pred_p = run_dir / "predictions.jsonl"
        if not (meta_p.exists() and pred_p.exists()):
            continue
        with open(meta_p) as fh:
            meta = json.load(fh)
        model_short = run_dir.name.split("__")[0]
        with open(pred_p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                if "pred_label" not in p:
                    # NER-benchmark runs (e.g. FiNER-ORD) use a pred_tags/
                    # pred_entities schema, not sentiment pred_label. They have
                    # their own loaders (load_ner_*) — skip them here.
                    continue
                rows.append({
                    "run_id": run_dir.name,
                    "model": model_short,
                    "dataset": meta.get("dataset", ""),
                    "template": meta.get("template") or "-",
                    "shots": int(meta.get("shots") or 0),
                    "id": p["id"],
                    "pred": p["pred_label"],
                    "raw_output": p.get("raw_output", ""),
                    "parse_ok": p.get("parse_ok", False),
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_ner_summary() -> dict | None:
    p = ROOT / "results" / "summary" / "ner_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_ner_top_entities() -> pd.DataFrame | None:
    p = ROOT / "results" / "summary" / "ner_top_entities.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# --- NER benchmark: FiNER-ORD has gold spans, so strict span-F1 is comparable
#     across models (unlike the exploratory FPB/FiQA entity-count view). ---
NER_MODEL_LABEL = {
    "gliner-large": "GLiNER-large-v2.1",
    "gliner-medium": "GLiNER-medium-v2.1",
    "gliner-small": "GLiNER-small-v2.1",
    "nuner-zero": "NuNER-Zero",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral-7B-Instruct-v0.3",
    "plutus8b": "plutus-8B-instruct (financial)",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_4b": "Qwen3-4B-Instruct",
    "gemma2_9b": "Gemma-2-9B-it",
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "finma7b": "FinMA-7B-full (financial)",
}
NER_MODEL_FAMILY = {
    "gliner-large": "Specialised NER", "gliner-medium": "Specialised NER",
    "gliner-small": "Specialised NER", "nuner-zero": "Specialised NER",
    "qwen25_7b": "General LLM", "mistral7b": "General LLM",
    "qwen3_8b": "General LLM", "qwen3_4b": "General LLM",
    "gemma2_9b": "General LLM", "llama31_8b": "General LLM",
    "plutus8b": "Financial LLM", "finma7b": "Financial LLM",
}


@st.cache_data
def load_ner_benchmark() -> pd.DataFrame | None:
    """FiNER-ORD multi-model benchmark table (gold-labelled → real F1)."""
    p = ROOT / "results" / "summary_ner" / "final_table_ner.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model_label"] = df["model"].map(NER_MODEL_LABEL).fillna(df["model"])
    df["family"] = df["model"].map(NER_MODEL_FAMILY).fillna("General LLM")
    df["cost_per_100"] = df.apply(
        lambda r: (r["cost_usd"] / r["n_samples"] * 100.0) if r["n_samples"] else 0.0,
        axis=1,
    )
    return df


@st.cache_data
def load_ner_entities(dataset: str, seed: int = 42) -> pd.DataFrame:
    """Load per-sample NER entities for a dataset."""
    p = ROOT / "results" / "predictions" / f"finbert_ner__{dataset}__seed{seed}" / "entities.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data
def load_gold() -> dict:
    """Gold labels for every sample id, keyed by id."""
    from src.data_loader import load_fpb, load_fiqa
    _, fpb_test = load_fpb(seed=42)
    fiqa_test = load_fiqa()
    return {s["id"]: s for s in (fpb_test + fiqa_test)}


def reorder_confusion(matrix: list, labels: list) -> np.ndarray:
    """Reorder a confusion matrix to LABEL_ORDER (negative/neutral/positive)."""
    mat = np.array(matrix)
    n = len(LABEL_ORDER)
    out = np.zeros((n, n), dtype=int)
    idx = {l: i for i, l in enumerate(labels)}
    for i, lbl_i in enumerate(LABEL_ORDER):
        for j, lbl_j in enumerate(LABEL_ORDER):
            if lbl_i in idx and lbl_j in idx:
                out[i][j] = mat[idx[lbl_i]][idx[lbl_j]]
    return out


# ===========================================================================
# Page setup + styling
# ===========================================================================

st.set_page_config(
    page_title="Financial NLP Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid")

# Custom CSS — tab pills, hero header, metric cards, section spacing.
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

.hero {
  background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 50%, #4a7ba7 100%);
  color: #ffffff;
  padding: 1.6rem 2rem;
  border-radius: 14px;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 14px rgba(30,58,95,0.18);
}
.hero h1 { color: #fff; margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { color: #d6e4f0; margin: 0.4rem 0 0; font-size: 0.95rem; max-width: 950px; line-height: 1.5; }
.hero .badges { margin-top: 0.7rem; }
.hero .badge {
  display: inline-block; background: rgba(255,255,255,0.16); color: #fff;
  padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.78rem; margin-right: 0.4rem;
  border: 1px solid rgba(255,255,255,0.22);
}

div[data-testid="stTabs"] > div[role="tablist"] {
  gap: 0.4rem;
  background: #f5f6f8;
  padding: 0.4rem;
  border-radius: 12px;
  border: 1px solid #e1e4ea;
}
div[data-testid="stTabs"] button[role="tab"] {
  padding: 0.55rem 1.1rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.92rem;
  color: #444;
  background: transparent;
  border: 1px solid transparent;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ffffff;
  color: #1e3a5f;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  border-color: #d0d5dd;
}

div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fb 100%);
  padding: 0.8rem 1rem;
  border-radius: 10px;
  border: 1px solid #e8eaf0;
}

.verdict-box {
  border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
  border-left: 5px solid #ccc; background: #fafbfc;
}
.verdict-green { border-left-color: #2e7d32; background: #f1f8f3; }
.verdict-red   { border-left-color: #c62828; background: #fdf3f3; }
.verdict-gray  { border-left-color: #6c757d; background: #f3f4f6; }

.sub-header {
  font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
  margin: 0.6rem 0 0.4rem; padding-bottom: 0.3rem;
  border-bottom: 2px solid #e1e4ea;
}
</style>
""", unsafe_allow_html=True)


# Hero header
st.markdown("""
<div class="hero">
  <h1>📊 Financial NLP Benchmark</h1>
  <p>
    Comparing <b>TheFinAI's plutus-8B-instruct</b> (the published successor to FinLLaMA-instruct)
    against general 7-8B LLMs and classical baselines on Financial PhraseBank and FiQA-SA —
    plus entity extraction on the same texts.
  </p>
  <div class="badges">
    <span class="badge">FPB · 690 test</span>
    <span class="badge">FiQA · 1,173 test</span>
    <span class="badge">6 sentiment models</span>
    <span class="badge">NER · ORG/PER/LOC/MISC</span>
  </div>
</div>
""", unsafe_allow_html=True)


df = load_table()
if df is None or df.empty:
    st.warning(
        "No `results/summary/final_table.csv` yet. "
        "Run `scripts/aggregate.py` after the matrix completes."
    )
    st.stop()


# ===========================================================================
# Sidebar (global)
# ===========================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Filters here apply to the **Sentiment** tab. The **NER** tab has its own scoped selectors.")

    all_datasets = sorted(df["dataset"].unique().tolist())
    all_models = sorted(df["model"].unique().tolist(), key=lambda m: MODEL_LABEL.get(m, m))
    sel_datasets = st.multiselect("Datasets", all_datasets, default=all_datasets)
    sel_models = st.multiselect(
        "Models",
        all_models,
        default=all_models,
        format_func=lambda m: MODEL_LABEL.get(m, m),
    )

    st.divider()
    st.subheader("💰 Modal cost")
    spend = load_modal_spend()
    if spend:
        st.metric("Total spend (T4)", f"${spend['total_usd']:.4f}")
        st.metric("Total seconds", f"{spend['total_seconds']:.0f}")
        st.caption(f"{len(spend['runs'])} runs recorded")
    else:
        st.caption("No Modal runs recorded yet")

    st.divider()
    st.caption(
        "💡 Re-run `scripts/aggregate.py` to refresh sentiment results, "
        "`scripts/run_finbert_ner.py` for NER."
    )

filtered = df[df["dataset"].isin(sel_datasets) & df["model"].isin(sel_models)]


# ===========================================================================
# Top-level tabs: Sentiment vs NER
# ===========================================================================

tab_sentiment, tab_ner = st.tabs([
    "📊  Sentiment Analysis",
    "🏷️  Named Entity Recognition",
])


# ---------------------------------------------------------------------------
# TAB 1: SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------

with tab_sentiment:
    if filtered.empty:
        st.info("No runs match the current sidebar filters. Widen your selection on the left.")
    else:
        sub_overview, sub_runs, sub_class, sub_highlights, sub_example, sub_prompt, sub_conf, sub_errors = st.tabs([
            "🏁 Overview",
            "📋 All runs",
            "📐 Per-class",
            "✨ Highlights",
            "🔍 Per-example",
            "🎛️ Prompt sensitivity",
            "🔲 Confusion",
            "❌ Errors & drill-down",
        ])

        # ----- Sub-tab: Overview -----
        with sub_overview:
            st.markdown('<div class="sub-header">At a glance</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Runs in view", len(filtered))
            c2.metric("Best F1-macro", f"{filtered['f1_macro'].max():.3f}")
            c3.metric("Best Accuracy", f"{filtered['accuracy'].max():.3f}")
            c4.metric("Mean coverage", f"{filtered['coverage'].mean():.2%}")

            st.markdown('<div class="sub-header">Does financial instruction-tuning help?</div>', unsafe_allow_html=True)
            st.caption(
                "Compares the focal model (plutus-8B-instruct, the published successor to FinLLaMA-instruct) "
                "to the strongest non-financial 7-8B LLM on the same dataset."
            )

            ans_cols = st.columns(2)
            focal_short = "plutus8b"
            fallback_focal = "finllama"
            non_fin_models = ("qwen25_7b", "mistral7b")

            for col, ds in zip(ans_cols, ["FPB", "FiQA"]):
                with col:
                    ds_df = filtered[filtered["dataset"] == ds]
                    focal_runs = ds_df[ds_df["model"].isin([focal_short, fallback_focal])]
                    non_fin_runs = ds_df[ds_df["model"].isin(non_fin_models)]
                    finbert_runs = ds_df[ds_df["model"] == "finbert"]

                    if focal_runs.empty:
                        st.info(f"**{ds}** — focal model hasn't run yet.")
                    elif non_fin_runs.empty:
                        st.info(f"**{ds}** — need a non-financial LLM run to compare.")
                    else:
                        focal_best = focal_runs.loc[focal_runs["f1_macro"].idxmax()]
                        non_fin_best = non_fin_runs.loc[non_fin_runs["f1_macro"].idxmax()]
                        delta = focal_best["f1_macro"] - non_fin_best["f1_macro"]
                        if delta > 0.02:
                            verdict_class = "verdict-green"
                            verdict = "✅ Yes — financial tuning helps"
                        elif delta < -0.02:
                            verdict_class = "verdict-red"
                            verdict = "❌ No — general LLM is better"
                        else:
                            verdict_class = "verdict-gray"
                            verdict = "➖ Roughly equivalent"

                        fb_line = ""
                        if not finbert_runs.empty:
                            fb_best = finbert_runs.loc[finbert_runs["f1_macro"].idxmax()]
                            fb_line = (
                                f"<br/><small><em>FinBERT (specialised classifier): "
                                f"F1m {fb_best['f1_macro']:.3f}</em></small>"
                            )

                        st.markdown(
                            f"""
                            <div class="verdict-box {verdict_class}">
                              <h4 style="margin:0 0 0.4rem;">{ds}</h4>
                              <b>{verdict}</b><br/>
                              <small>Focal: <code>{MODEL_LABEL.get(focal_best['model'], focal_best['model'])}</code>
                              ({focal_best['template']}, {int(focal_best['shots'])}-shot) →
                              <b>F1m {focal_best['f1_macro']:.3f}</b></small><br/>
                              <small>Best general LLM: <code>{MODEL_LABEL.get(non_fin_best['model'], non_fin_best['model'])}</code>
                              ({non_fin_best['template']}, {int(non_fin_best['shots'])}-shot) →
                              <b>F1m {non_fin_best['f1_macro']:.3f}</b></small><br/>
                              <small>Δ (focal − general): <b>{delta:+.3f}</b></small>
                              {fb_line}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown('<div class="sub-header">F1-macro — best config per model × dataset</div>', unsafe_allow_html=True)
            best = (
                filtered.sort_values("f1_macro", ascending=False)
                .groupby(["model", "model_label", "family", "dataset"], as_index=False)
                .first()
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            sns.barplot(data=best, x="model_label", y="f1_macro", hue="dataset", ax=ax, palette=["#1976d2", "#ef6c00"])
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel("F1-macro")
            ax.axhline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.6)
            plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
            for c in ax.containers:
                ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
            ax.legend(title="Dataset", loc="lower right", frameon=True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ----- Sub-tab: All runs -----
        with sub_runs:
            st.markdown('<div class="sub-header">All run rows</div>', unsafe_allow_html=True)
            st.caption(
                f"Showing **{len(filtered)} runs** matching the sidebar filters. "
                "Sort any column. Use the search box to find specific runs."
            )
            display = filtered[[
                "model_label", "family", "dataset", "template", "shots", "n_samples",
                "accuracy", "f1_macro", "f1_weighted", "coverage", "runtime_s",
            ]].rename(columns={"model_label": "model"})
            st.dataframe(
                display.sort_values(["dataset", "model", "template", "shots"]).reset_index(drop=True),
                width="stretch",
                height=min(60 + 35 * len(display), 600),
                column_config={
                    "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0.0, max_value=1.0, format="%.3f"),
                    "f1_macro": st.column_config.ProgressColumn("f1_macro", min_value=0.0, max_value=1.0, format="%.3f"),
                    "f1_weighted": st.column_config.ProgressColumn("f1_w", min_value=0.0, max_value=1.0, format="%.3f"),
                    "coverage": st.column_config.ProgressColumn("coverage", min_value=0.0, max_value=1.0, format="%.2f"),
                    "runtime_s": st.column_config.NumberColumn("runtime (s)", format="%.1f"),
                },
            )

        # ----- Sub-tab: Per-class -----
        with sub_class:
            st.markdown('<div class="sub-header">Per-class precision / recall / F1</div>', unsafe_allow_html=True)
            st.caption(
                "Class-level numbers tell you *how* a model fails. "
                "A model with F1m 0.80 on FPB might be perfect on positive but useless on neutral — "
                "this view exposes that."
            )

            @st.cache_data
            def per_class_table(_hash: int) -> pd.DataFrame:
                cdir = ROOT / "results" / "summary" / "confusions"
                rows = []
                if not cdir.exists():
                    return pd.DataFrame()
                for f in cdir.glob("*.json"):
                    with open(f) as fh:
                        cf = json.load(fh)
                    run_id = cf.get("run_id", f.stem)
                    per = cf.get("per_class", {})
                    parts = run_id.split("__")
                    model_short = parts[0]
                    dataset = parts[1] if len(parts) > 1 else ""
                    if len(parts) >= 4:
                        tpl = parts[2]
                        shots = int(parts[3].replace("shot", "")) if "shot" in parts[3] else 0
                    else:
                        tpl, shots = "-", 0
                    for cls, m in per.items():
                        rows.append({
                            "model": model_short,
                            "model_label": MODEL_LABEL.get(model_short, model_short),
                            "dataset": dataset,
                            "template": tpl,
                            "shots": shots,
                            "class": cls,
                            "precision": round(m["precision"], 3),
                            "recall": round(m["recall"], 3),
                            "f1": round(m["f1"], 3),
                            "support": m["support"],
                        })
                return pd.DataFrame(rows)

            pc = per_class_table(hash(tuple(filtered.index)))
            if pc.empty:
                st.info("No confusion data yet.")
            else:
                pc_view = pc[pc["model"].isin(filtered["model"].unique()) & pc["dataset"].isin(filtered["dataset"].unique())]
                pc_best_keys = (
                    filtered.sort_values("f1_macro", ascending=False)
                    .groupby(["model", "dataset"], as_index=False)
                    .first()[["model", "dataset", "template", "shots"]]
                )
                pc_best = pc_view.merge(pc_best_keys, on=["model", "dataset", "template", "shots"], how="inner")

                if pc_best.empty:
                    st.info("Per-class data not joinable to runs in view.")
                else:
                    ds_pick_pc = st.selectbox(
                        "Dataset",
                        sorted(pc_best["dataset"].unique()),
                        key="pc_ds",
                    )
                    pc_show = pc_best[pc_best["dataset"] == ds_pick_pc].sort_values(["model_label", "class"])
                    col_l, col_r = st.columns([3, 2])
                    with col_l:
                        st.dataframe(
                            pc_show[["model_label", "class", "precision", "recall", "f1", "support"]]
                            .rename(columns={"model_label": "model"})
                            .reset_index(drop=True),
                            width="stretch",
                            height=min(60 + 35 * len(pc_show), 500),
                        )
                    with col_r:
                        pivot_f1 = pc_show.pivot(index="model_label", columns="class", values="f1")
                        col_order = [c for c in LABEL_ORDER if c in pivot_f1.columns]
                        pivot_f1 = pivot_f1[col_order]
                        fig, ax = plt.subplots(figsize=(5, max(2.5, 0.55 * len(pivot_f1))))
                        sns.heatmap(
                            pivot_f1, annot=True, fmt=".2f", cmap="RdYlGn",
                            vmin=0, vmax=1, cbar_kws={"label": "F1"}, ax=ax,
                        )
                        ax.set_title(f"Per-class F1 — {ds_pick_pc}", fontsize=10)
                        ax.set_ylabel("")
                        ax.set_xlabel("")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        # ----- Sub-tab: Highlights -----
        with sub_highlights:
            st.markdown('<div class="sub-header">Highlights for slides</div>', unsafe_allow_html=True)
            st.caption(
                "Auto-curated examples worth screenshotting for your deck. "
                "Sorted by 'tellability': clear focal-model wins, prompt flips, unanimous misses."
            )
            highlights_preds = load_all_predictions()
            if highlights_preds.empty:
                st.info("No predictions yet.")
            else:
                gold_for_highlights = load_gold()
                h1, h2 = st.columns([2, 3])
                h_dataset = h1.selectbox(
                    "Dataset",
                    sorted(highlights_preds["dataset"].dropna().unique()),
                    key="hl_ds",
                )
                n_show = h2.slider("How many highlights", 3, 20, 8, key="hl_n")

                pool = highlights_preds[highlights_preds["dataset"] == h_dataset]
                by_id = pool.groupby("id")
                rows = []
                for sid, grp in by_id:
                    gs = gold_for_highlights.get(sid)
                    if gs is None:
                        continue
                    gold_lbl = gs["label"]
                    preds_by_model = (
                        grp.groupby("model")["pred"]
                        .agg(lambda s: s.dropna().tolist())
                        .to_dict()
                    )
                    focal_preds = preds_by_model.get("plutus8b", []) + preds_by_model.get("finllama", [])
                    general_models = ("qwen25_7b", "mistral7b")
                    general_preds = sum((preds_by_model.get(m, []) for m in general_models), [])
                    any_focal_correct = any(p == gold_lbl for p in focal_preds)
                    any_general_correct = any(p == gold_lbl for p in general_preds)
                    any_focal_wrong = any(p != gold_lbl for p in focal_preds if p is not None)
                    any_general_wrong = any(p != gold_lbl for p in general_preds if p is not None)

                    score = 0
                    tag = ""
                    if any_focal_correct and any_general_wrong and not any_general_correct:
                        score = 4
                        tag = "Focal wins, generals miss"
                    elif any_general_correct and any_focal_wrong and not any_focal_correct:
                        score = 3
                        tag = "Generals win, focal misses"
                    else:
                        for m, plist in preds_by_model.items():
                            if len(set(plist)) > 1:
                                score = max(score, 2)
                                tag = tag or f"{MODEL_LABEL.get(m, m)} flips between prompts"
                        all_preds_list = sum(preds_by_model.values(), [])
                        non_null_preds = [p for p in all_preds_list if p is not None]
                        if non_null_preds and all(p != gold_lbl for p in non_null_preds):
                            score = max(score, 2)
                            tag = tag or "Everyone misses"

                    if score > 0:
                        rows.append({
                            "score": score, "tag": tag, "id": sid,
                            "text": gs["text"], "gold": gold_lbl,
                            "preds": preds_by_model,
                        })

                rows.sort(key=lambda r: -r["score"])
                rows = rows[:n_show]

                if not rows:
                    st.info("No highlight cases yet — wait for more LLM runs to complete.")
                else:
                    for i, r in enumerate(rows, 1):
                        with st.container(border=True):
                            st.markdown(f"**#{i} · {r['tag']}** · `{r['id']}`  →  gold = `{r['gold']}`")
                            st.markdown(f"> {r['text']}")
                            pred_lines = []
                            for m_short, preds in sorted(r["preds"].items()):
                                label = MODEL_LABEL.get(m_short, m_short)
                                if not preds:
                                    continue
                                if all(p == preds[0] for p in preds):
                                    pred_lines.append(f"- **{label}:** `{preds[0]}` ({len(preds)} runs)")
                                else:
                                    pred_lines.append(f"- **{label}:** {', '.join(f'`{p}`' for p in preds)}")
                            st.markdown("\n".join(pred_lines))

        # ----- Sub-tab: Per-example -----
        with sub_example:
            st.markdown('<div class="sub-header">Per-example breakdown</div>', unsafe_allow_html=True)
            st.caption(
                "Pick a single example. See every model's prediction next to the gold label — "
                "useful for spotting disagreements and surfacing concrete cases for the presentation."
            )

            c_reload, _ = st.columns([1, 6])
            if c_reload.button("🔄 Reload predictions", key="reload_pe"):
                load_all_predictions.clear()
                load_gold.clear()

            all_preds = load_all_predictions()
            if all_preds.empty:
                st.info("No predictions yet — wait for the matrix to write at least one run.")
            else:
                gold_map = load_gold()

                f1c, f2c, f3c = st.columns([2, 3, 2])
                ds_pick = f1c.selectbox(
                    "Dataset",
                    sorted(all_preds["dataset"].dropna().unique()),
                    key="example_ds",
                )
                text_search = f2c.text_input("Search text contains", key="example_search")
                only_disagree = f3c.checkbox("Only disagreements", value=False, key="example_disagree")

                pool = all_preds[all_preds["dataset"] == ds_pick]
                candidate_ids = sorted(pool["id"].unique().tolist())

                if text_search:
                    s_lower = text_search.lower()
                    candidate_ids = [
                        i for i in candidate_ids
                        if i in gold_map and s_lower in gold_map[i]["text"].lower()
                    ]

                if only_disagree:
                    kept: list[str] = []
                    for i in candidate_ids:
                        sub = pool[pool["id"] == i]
                        preds_set = set(sub["pred"].dropna().unique())
                        if len(preds_set) > 1:
                            kept.append(i)
                    candidate_ids = kept

                if not candidate_ids:
                    st.info("No examples match the filter.")
                else:
                    sel_id = st.selectbox(
                        f"Pick an example  ({len(candidate_ids)} available)",
                        candidate_ids,
                        key="example_id",
                    )
                    gold_sample = gold_map.get(sel_id)
                    if gold_sample is None:
                        st.warning(f"Sample {sel_id} not in gold map.")
                    else:
                        with st.container(border=True):
                            st.markdown(f"**Sample id:** `{sel_id}` &nbsp; · &nbsp; **Gold:** `{gold_sample['label']}`")
                            st.markdown(f"**Text:** {gold_sample['text']}")

                        sub = pool[pool["id"] == sel_id].copy()
                        sub["model"] = sub["model"].map(MODEL_LABEL).fillna(sub["model"])
                        sub["correct"] = sub["pred"] == gold_sample["label"]
                        view = sub[[
                            "model", "template", "shots", "pred", "correct", "parse_ok", "raw_output",
                        ]].rename(columns={"raw_output": "raw"}).reset_index(drop=True)

                        def _row_style(row):
                            if row["pred"] is None or (isinstance(row["pred"], float) and pd.isna(row["pred"])):
                                return ["background-color: #ffe9c2"] * len(row)
                            if row["correct"]:
                                return ["background-color: #d6f5d6"] * len(row)
                            return ["background-color: #fde0e0"] * len(row)

                        try:
                            styled = view.style.apply(_row_style, axis=1)
                            st.dataframe(styled, width="stretch")
                        except Exception:
                            st.dataframe(view, width="stretch")

                        n_correct = int(view["correct"].sum())
                        n_total = len(view)
                        n_parse_fail = int((~view["parse_ok"]).sum())
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Models correct", f"{n_correct} / {n_total}")
                        sc2.metric("Models wrong", f"{n_total - n_correct - n_parse_fail}")
                        sc3.metric("Parse failures", n_parse_fail)

        # ----- Sub-tab: Prompt sensitivity + Few-shot -----
        with sub_prompt:
            llm_models = ("plutus8b", "mistral7b", "qwen25_7b", "finllama")
            llm = filtered[filtered["model"].isin(llm_models)]
            if llm.empty:
                st.info("Need at least one LLM run to show prompt-sensitivity / few-shot effects.")
            else:
                st.markdown('<div class="sub-header">Prompt sensitivity (Template A vs B)</div>', unsafe_allow_html=True)
                st.caption(
                    "ΔF1-macro between Template A and Template B per (model, shots, dataset). "
                    "Large gap = result depends on phrasing more than on the model."
                )
                pivot = (
                    llm.pivot_table(
                        index=["model_label", "dataset", "shots"],
                        columns="template",
                        values="f1_macro",
                    )
                    .dropna(how="any")
                )
                if {"A", "B"}.issubset(pivot.columns):
                    pivot["Δ (B-A)"] = (pivot["B"] - pivot["A"]).round(3)
                    st.dataframe(pivot, width="stretch")
                else:
                    st.info("Need both Template A and B runs for a model to show the gap.")

                st.markdown('<div class="sub-header">Few-shot effect (0-shot vs 3-shot)</div>', unsafe_allow_html=True)
                st.caption(
                    "ΔF1-macro between 0-shot and 3-shot per (model, template, dataset). "
                    "If a financial-instruction-tuned model gains *less* from few-shot, "
                    "that's evidence the tuning already encodes the task."
                )
                pivot2 = (
                    llm.pivot_table(
                        index=["model_label", "dataset", "template"],
                        columns="shots",
                        values="f1_macro",
                    )
                    .dropna(how="any")
                )
                if {0, 3}.issubset(pivot2.columns):
                    pivot2["Δ (3-0)"] = (pivot2[3] - pivot2[0]).round(3)
                    st.dataframe(pivot2, width="stretch")
                else:
                    st.info("Need both 0-shot and 3-shot runs to show the gap.")

        # ----- Sub-tab: Confusion -----
        with sub_conf:
            st.markdown('<div class="sub-header">Confusion matrices — best config per model × dataset</div>', unsafe_allow_html=True)
            confs = load_confusions()
            best = (
                filtered.sort_values("f1_macro", ascending=False)
                .groupby(["model", "model_label", "dataset"], as_index=False)
                .first()
            )
            models_in_view = sorted(best["model"].unique(), key=lambda m: MODEL_LABEL.get(m, m))
            datasets_in_view = sorted(best["dataset"].unique())

            for ds in datasets_in_view:
                st.markdown(f"#### {ds}")
                cols = st.columns(len(models_in_view) or 1)
                for i, m in enumerate(models_in_view):
                    row = best[(best["model"] == m) & (best["dataset"] == ds)]
                    with cols[i]:
                        if row.empty:
                            st.write(f"_{MODEL_LABEL.get(m, m)}: no run_")
                            continue
                        r = row.iloc[0]
                        tpl = r["template"]
                        shots = int(r["shots"]) if pd.notna(r["shots"]) else 0
                        seed = int(r["seed"])
                        run_id = (
                            f"{m}__{ds}__seed{seed}" if tpl == "-"
                            else f"{m}__{ds}__{tpl}__{shots}shot__seed{seed}"
                        )
                        cf = confs.get(run_id)
                        if not cf:
                            st.write(f"_{MODEL_LABEL.get(m, m)}: no confusion data_")
                            continue
                        mat = reorder_confusion(cf["matrix"], cf.get("labels", []))
                        fig, ax = plt.subplots(figsize=(2.6, 2.4))
                        sns.heatmap(
                            mat, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["neg", "neu", "pos"],
                            yticklabels=["neg", "neu", "pos"],
                            cbar=False, ax=ax,
                        )
                        title = MODEL_LABEL.get(m, m).split(" ")[0]
                        ax.set_title(
                            f"{title}\nf1m={r['f1_macro']:.2f}  acc={r['accuracy']:.2f}",
                            fontsize=8,
                        )
                        ax.set_xlabel("predicted", fontsize=7)
                        ax.set_ylabel("true", fontsize=7)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        # ----- Sub-tab: Errors & per-run drill-down -----
        with sub_errors:
            st.markdown('<div class="sub-header">Error explorer — focal-model misses</div>', unsafe_allow_html=True)
            errors = load_errors()
            if errors is None:
                st.info("Run `scripts/error_analysis.py` after the LLM matrix to populate the error sample.")
            else:
                st.caption(
                    f"{len(errors)} sampled misclassifications from the focal-model run(s). "
                    "Use these for hand-categorization of error types."
                )
                err_dataset = st.selectbox(
                    "Filter by dataset", ["all"] + sorted(errors["dataset"].unique().tolist()),
                    key="err_ds_pick",
                )
                err_view = errors if err_dataset == "all" else errors[errors["dataset"] == err_dataset]
                st.dataframe(err_view, width="stretch", height=380)

                if len(err_view):
                    st.markdown("#### Drill into a specific row")
                    idx = st.slider("Row", 0, len(err_view) - 1, 0, key="err_row")
                    row = err_view.iloc[idx]
                    with st.container(border=True):
                        st.markdown(f"**Dataset:** {row['dataset']}  |  **id:** `{row['id']}`")
                        st.markdown(f"**Text:** {row['text']}")
                        cc1, cc2 = st.columns(2)
                        cc1.markdown(f"**Gold:** `{row['gold']}`")
                        cc2.markdown(f"**Predicted:** `{row['pred']}`")
                        st.markdown(f"**Raw model output:** `{row['raw_output']}`")

            st.markdown('<div class="sub-header">Per-run drill-down</div>', unsafe_allow_html=True)
            best = (
                filtered.sort_values("f1_macro", ascending=False)
                .groupby(["model", "model_label", "dataset"], as_index=False)
                .first()
            )
            run_ids = sorted(filtered.apply(
                lambda r: (
                    f"{r['model']}__{r['dataset']}__seed{int(r['seed'])}"
                    if r["template"] == "-"
                    else f"{r['model']}__{r['dataset']}__{r['template']}__{int(r['shots'])}shot__seed{int(r['seed'])}"
                ),
                axis=1,
            ).unique().tolist())

            sel_run = st.selectbox("Pick a run", run_ids, key="run_drill_pick") if run_ids else None
            if sel_run:
                preds = load_predictions(sel_run)
                if preds is None:
                    st.warning(f"No predictions.jsonl for {sel_run}")
                else:
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Predictions", len(preds))
                    cc2.metric("Parse OK", f"{preds['parse_ok'].mean():.2%}")
                    cc3.metric(
                        "Mean latency",
                        f"{preds['latency_ms'].mean():.0f} ms"
                        if "latency_ms" in preds
                        else "—",
                    )
                    st.dataframe(preds.head(50), width="stretch", height=300)
                    st.caption("Showing first 50 rows of predictions.jsonl")


# ---------------------------------------------------------------------------
# TAB 2: NAMED ENTITY RECOGNITION
# ---------------------------------------------------------------------------

with tab_ner:
    ner_summary = load_ner_summary()
    ner_top = load_ner_top_entities()

    if ner_summary is None:
        st.info(
            "🛈 No NER results yet. Run **`scripts/run_finbert_ner.py`** (via Modal) to populate "
            "`results/summary/ner_summary.json`."
        )
    else:
        sub_n_perf, sub_n_overview, sub_n_types, sub_n_cross, sub_n_top, sub_n_drill = st.tabs([
            "🏆 Model performance",
            "🏁 Overview",
            "📊 Entity types",
            "🎯 Sentiment cross-tab",
            "🔎 Top entities",
            "🔍 Per-sample drill-down",
        ])

        # ----- Sub-tab: Model performance (FiNER-ORD benchmark) -----
        with sub_n_perf:
            st.markdown('<div class="sub-header">Model performance — FiNER-ORD benchmark</div>', unsafe_allow_html=True)
            ner_bench = load_ner_benchmark()
            if ner_bench is None or ner_bench.empty:
                st.info(
                    "No NER benchmark results found. Run **`scripts/run_ner.py`** then "
                    "**`scripts/aggregate_ner.py`** to populate "
                    "`results/summary_ner/final_table_ner.csv`."
                )
            else:
                st.caption(
                    "The other NER sub-tabs are **exploratory** entity counts on FPB/FiQA (no gold "
                    "labels). This view is the **FiNER-ORD** benchmark, which ships gold entity spans — "
                    "so strict span-F1 is directly comparable across models. "
                    "Strict = exact boundary **and** type (seqeval IOB2); higher is better."
                )

                # Collapse to the single best run per model (by strict-F1).
                board = (
                    ner_bench.sort_values("strict_f1", ascending=False)
                    .groupby("model", as_index=False)
                    .first()
                    .sort_values("strict_f1", ascending=False)
                    .reset_index(drop=True)
                )

                # Headline metrics — overall best, best specialised NER, best LLM.
                k1, k2, k3 = st.columns(3)
                top_row = board.iloc[0]
                k1.metric("Best overall", top_row["model_label"], f"strict-F1 {top_row['strict_f1']:.3f}")
                best_spec = board[board["family"] == "Specialised NER"]
                best_llm = board[board["family"].isin(["General LLM", "Financial LLM"])]
                if not best_spec.empty:
                    r = best_spec.iloc[0]
                    k2.metric("Best specialised NER", r["model_label"], f"strict-F1 {r['strict_f1']:.3f}")
                if not best_llm.empty:
                    r = best_llm.iloc[0]
                    k3.metric("Best LLM", r["model_label"], f"strict-F1 {r['strict_f1']:.3f}")

                # Strict-F1 comparison bar chart, coloured by model family.
                chart = board.sort_values("strict_f1", ascending=True)
                fam_color = {
                    "Specialised NER": "#1976d2",
                    "General LLM": "#7b4fb3",
                    "Financial LLM": "#ef6c00",
                }
                fig, ax = plt.subplots(figsize=(9, max(2.5, 0.55 * len(chart))))
                bars = ax.barh(
                    chart["model_label"], chart["strict_f1"],
                    color=[fam_color.get(f, "#9aa0a6") for f in chart["family"]],
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel("Strict span-F1")
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
                ax.set_axisbelow(True)
                ax.grid(axis="x", alpha=0.3)
                from matplotlib.patches import Patch
                handles = [Patch(color=c, label=f) for f, c in fam_color.items()
                           if f in chart["family"].unique()]
                ax.legend(handles=handles, loc="lower right", fontsize=8, title="Model family")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Leaderboard table with per-type F1.
                st.markdown("**Leaderboard — best config per model**")
                board.insert(0, "rank", board.index + 1)
                board["config"] = board.apply(
                    lambda r: "—" if r["backend"] == "local"
                    else f"tmpl {r['template']} · {int(r['shots'])}-shot",
                    axis=1,
                )
                lb_rename = {
                    "rank": "#", "model_label": "Model", "family": "Family", "config": "Config",
                    "strict_f1": "Strict-F1", "partial_f1": "Partial-F1",
                    "f1_PER": "PER", "f1_LOC": "LOC", "f1_ORG": "ORG",
                    "coverage": "Coverage", "cost_per_100": "Cost/100",
                }
                lb_cols = [c for c in lb_rename if c in board.columns]
                show = board[lb_cols].rename(columns=lb_rename)
                grad = [c for c in ["Strict-F1", "Partial-F1", "PER", "LOC", "ORG"] if c in show.columns]
                styler = (
                    show.style
                    .format({
                        "Strict-F1": "{:.3f}", "Partial-F1": "{:.3f}",
                        "PER": "{:.3f}", "LOC": "{:.3f}", "ORG": "{:.3f}",
                        "Coverage": "{:.0%}", "Cost/100": "${:.4f}",
                    })
                    .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, subset=grad)
                )
                st.dataframe(styler, width="stretch", hide_index=True)
                st.caption(
                    "Per-type columns (PER / LOC / ORG) show where each model is strong or weak. "
                    "Cost is USD per 100 examples (local models are free)."
                )

        # ----- Sub-tab: Overview -----
        with sub_n_overview:
            st.markdown('<div class="sub-header">NER overview</div>', unsafe_allow_html=True)
            st.caption(
                f"Model: `{ner_summary.get('model_hf_id', '?')}` · "
                f"aggregation: `{ner_summary.get('aggregation_strategy', '?')}` · "
                f"seed: {ner_summary.get('seed', '?')}. "
                "FPB/FiQA have no NER gold labels, so this is **exploratory** — counts, "
                "top entities, and cross-tabs against the sentiment gold label."
            )
            ner_cols = st.columns(len(ner_summary["datasets"]) or 1)
            for col, ds_s in zip(ner_cols, ner_summary["datasets"]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {ds_s['dataset']}")
                        m1, m2 = st.columns(2)
                        m1.metric("Samples", ds_s["n_samples"])
                        m2.metric("Total entities", ds_s["total_entities"])
                        m3, m4 = st.columns(2)
                        m3.metric("Avg / sample", f"{ds_s['avg_entities_per_sample']:.2f}")
                        m4.metric("With ≥1 entity", f"{ds_s['pct_samples_with_entity']:.0f}%")
                        st.markdown(
                            "**By type:** "
                            + ", ".join(
                                f"`{k}` {v}"
                                for k, v in sorted(
                                    ds_s["entities_by_type"].items(), key=lambda kv: -kv[1]
                                )
                            )
                        )

        # ----- Sub-tab: Entity types bar chart -----
        with sub_n_types:
            st.markdown('<div class="sub-header">Entity types per dataset</div>', unsafe_allow_html=True)
            st.caption("Which entity categories show up most in each dataset's text?")
            bar_rows = []
            for ds_s in ner_summary["datasets"]:
                for etype, count in ds_s["entities_by_type"].items():
                    bar_rows.append({"dataset": ds_s["dataset"], "entity_type": etype, "count": count})
            bar_df = pd.DataFrame(bar_rows)
            if not bar_df.empty:
                fig, ax = plt.subplots(figsize=(9, 4))
                sns.barplot(data=bar_df, x="entity_type", y="count", hue="dataset",
                            palette=["#1976d2", "#ef6c00"], ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("# entities")
                for c in ax.containers:
                    ax.bar_label(c, fmt="%d", fontsize=8, padding=2)
                ax.legend(title="Dataset", loc="upper right", frameon=True)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ----- Sub-tab: Entity × sentiment cross-tab -----
        with sub_n_cross:
            st.markdown('<div class="sub-header">Entity types × gold sentiment label</div>', unsafe_allow_html=True)
            st.caption(
                "Are positive / neutral / negative texts mentioning different kinds of entities? "
                "Counts of detected entity types, split by the gold sentiment label of the source text."
            )
            cross_ds = st.selectbox(
                "Dataset for cross-tab",
                [d["dataset"] for d in ner_summary["datasets"]],
                key="ner_cross_ds",
            )
            cross_data = next(
                (d for d in ner_summary["datasets"] if d["dataset"] == cross_ds), None
            )
            if cross_data and cross_data.get("entities_by_sentiment_then_type"):
                cross_df = pd.DataFrame(cross_data["entities_by_sentiment_then_type"]).fillna(0).astype(int)
                col_order = [c for c in LABEL_ORDER if c in cross_df.columns]
                cross_df = cross_df[col_order]
                cross_df = cross_df.loc[cross_df.sum(axis=1).sort_values(ascending=False).index]

                col_l, col_r = st.columns([3, 2])
                with col_l:
                    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.6 * len(cross_df))))
                    sns.heatmap(
                        cross_df, annot=True, fmt="d", cmap="Purples",
                        cbar_kws={"label": "count"}, ax=ax
                    )
                    ax.set_title(f"Entity counts by gold sentiment — {cross_ds}", fontsize=10)
                    ax.set_xlabel("gold sentiment")
                    ax.set_ylabel("entity type")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                with col_r:
                    st.markdown("**Raw counts**")
                    st.dataframe(cross_df, width="stretch")
            else:
                st.info("No entity × sentiment data for this dataset.")

        # ----- Sub-tab: Top entities -----
        with sub_n_top:
            st.markdown('<div class="sub-header">Top entities per type</div>', unsafe_allow_html=True)
            if ner_top is None or ner_top.empty:
                st.info("No `ner_top_entities.csv` found.")
            else:
                t1, t2, t3 = st.columns([2, 2, 1])
                top_ds = t1.selectbox(
                    "Dataset",
                    sorted(ner_top["dataset"].unique()),
                    key="ner_top_ds",
                )
                avail_types = sorted(ner_top[ner_top["dataset"] == top_ds]["entity_group"].unique())
                top_type = t2.selectbox("Entity type", avail_types, key="ner_top_type")
                n_top = t3.slider("Top N", 5, 20, 10, key="ner_top_n")

                top_view = (
                    ner_top[(ner_top["dataset"] == top_ds) & (ner_top["entity_group"] == top_type)]
                    .sort_values("count", ascending=False)
                    .head(n_top)
                )
                if top_view.empty:
                    st.info("No entities for this combination.")
                else:
                    col_l, col_r = st.columns([3, 2])
                    with col_l:
                        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(top_view))))
                        sns.barplot(data=top_view, y="word", x="count", color="#7b4fb3", ax=ax)
                        ax.set_ylabel("")
                        ax.set_xlabel("# mentions")
                        ax.set_title(f"Top {top_type} entities in {top_ds}")
                        for c in ax.containers:
                            ax.bar_label(c, fmt="%d", fontsize=8, padding=2)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    with col_r:
                        st.markdown("**Table**")
                        st.dataframe(
                            top_view[["rank", "word", "count"]].reset_index(drop=True),
                            width="stretch",
                            height=min(60 + 35 * len(top_view), 500),
                        )

        # ----- Sub-tab: Per-sample drill-down -----
        with sub_n_drill:
            st.markdown('<div class="sub-header">Per-sample entity drill-down</div>', unsafe_allow_html=True)
            st.caption("Pick a sample and see the extracted entities highlighted in the text alongside its gold sentiment.")

            d1, d2, d3 = st.columns([2, 3, 2])
            drill_ds = d1.selectbox(
                "Dataset",
                [d["dataset"] for d in ner_summary["datasets"]],
                key="ner_drill_ds",
            )
            drill_search = d2.text_input(
                "Search text contains", key="ner_drill_search",
                placeholder="e.g. Apple, EUR, profit…",
            )
            drill_sentiment = d3.selectbox(
                "Filter by gold sentiment",
                ["all"] + LABEL_ORDER,
                key="ner_drill_sentiment",
            )

            ents_df = load_ner_entities(drill_ds)
            if ents_df.empty:
                st.info(f"No NER entities file for {drill_ds}.")
            else:
                if drill_search:
                    s = drill_search.lower()
                    ents_df = ents_df[ents_df["text"].str.lower().str.contains(s, na=False)]
                if drill_sentiment != "all":
                    ents_df = ents_df[ents_df["gold_label"] == drill_sentiment]
                if ents_df.empty:
                    st.info("No matches with current filters.")
                else:
                    # Legend for entity-type colors
                    st.markdown(
                        "**Legend:** "
                        "<span style='background:#fde6a8;padding:2px 6px;border-radius:3px;'>ORG</span> "
                        "<span style='background:#c8e6c9;padding:2px 6px;border-radius:3px;'>PER</span> "
                        "<span style='background:#bbdefb;padding:2px 6px;border-radius:3px;'>LOC</span> "
                        "<span style='background:#f8bbd0;padding:2px 6px;border-radius:3px;'>MISC</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Showing first 30 of **{len(ents_df)}** matches.")

                    ents_df = ents_df.head(30)
                    color_map = {"ORG": "#fde6a8", "PER": "#c8e6c9", "LOC": "#bbdefb", "MISC": "#f8bbd0"}
                    sentiment_color = {"positive": "#2e7d32", "neutral": "#6c757d", "negative": "#c62828"}

                    for _, r in ents_df.iterrows():
                        with st.container(border=True):
                            s_color = sentiment_color.get(r["gold_label"], "#444")
                            st.markdown(
                                f"`{r['id']}` &nbsp;·&nbsp; gold = "
                                f"<span style='color:{s_color};font-weight:600;'>{r['gold_label']}</span> "
                                f"&nbsp;·&nbsp; {len(r['entities'])} entities",
                                unsafe_allow_html=True,
                            )
                            text = r["text"]
                            ents = sorted(r["entities"], key=lambda e: e.get("start", -1))
                            out_parts = []
                            cursor = 0
                            for e in ents:
                                start = e.get("start", -1)
                                end = e.get("end", -1)
                                if start < 0 or end < 0 or start < cursor:
                                    continue
                                out_parts.append(text[cursor:start])
                                bg = color_map.get(e["entity_group"], "#eee")
                                out_parts.append(
                                    f"<span style='background:{bg};padding:1px 5px;border-radius:3px;'>"
                                    f"{text[start:end]}<sub style='font-size:0.7em;color:#555;'>"
                                    f" {e['entity_group']}</sub></span>"
                                )
                                cursor = end
                            out_parts.append(text[cursor:])
                            st.markdown("".join(out_parts), unsafe_allow_html=True)


# ===========================================================================
# Footer
# ===========================================================================

st.divider()
st.caption(
    "Built for the LLMs in Finance seminar. "
    "Re-run `scripts/aggregate.py` + `scripts/run_finbert_ner.py` and refresh to update."
)
