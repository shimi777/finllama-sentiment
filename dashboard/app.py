"""Streamlit dashboard for the FinLLaMA-vs-baselines sentiment benchmark.

Run:
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py

Reads from results/summary/final_table.csv, results/summary/confusions/, and
results/summary/focal_error_sample.csv. Re-runs of scripts/aggregate.py update
the data shown here automatically (just refresh the browser).
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
    "finbert": "FinBERT (110M)",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral-7B-Instruct-v0.3",
    "plutus8b": "plutus-8B-instruct (TheFinAI)",
    "finllama": "FinLLaMA-instruct",
}


# ----------------- Data loaders (cached) -----------------

@st.cache_data
def load_table() -> pd.DataFrame | None:
    p = ROOT / "results" / "summary" / "final_table.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model_label"] = df["model"].map(MODEL_LABEL).fillna(df["model"])
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


# ----------------- Page setup -----------------

st.set_page_config(
    page_title="FinLLaMA Sentiment Benchmark",
    page_icon="📊",
    layout="wide",
)
sns.set_theme(style="whitegrid")

st.title("Financial Sentiment Classification — Results Dashboard")
st.caption(
    "Comparing TheFinAI's plutus-8B-instruct (the published successor to FinLLaMA-instruct) "
    "against general 7-8B instruction-tuned LLMs (Mistral, Qwen2.5) and classical baselines "
    "(FinBERT, VADER) on Financial PhraseBank and FiQA-SA."
)

df = load_table()

if df is None or df.empty:
    st.warning(
        "No `results/summary/final_table.csv` yet. "
        "Run `scripts/aggregate.py` after the matrix completes."
    )
    st.stop()

# ----------------- Sidebar filters -----------------

with st.sidebar:
    st.header("Filters")
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
    st.subheader("Modal cost")
    spend = load_modal_spend()
    if spend:
        st.metric("Total spend (T4)", f"${spend['total_usd']:.4f}")
        st.metric("Total seconds", f"{spend['total_seconds']:.0f}")
        st.caption(f"{len(spend['runs'])} runs recorded")
    else:
        st.caption("No Modal runs recorded yet")

filtered = df[df["dataset"].isin(sel_datasets) & df["model"].isin(sel_models)]

if filtered.empty:
    st.info("No runs match the current filters.")
    st.stop()

# ----------------- Top metrics -----------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runs in view", len(filtered))
c2.metric("Best F1-macro", f"{filtered['f1_macro'].max():.3f}")
c3.metric("Best Accuracy", f"{filtered['accuracy'].max():.3f}")
c4.metric("Mean coverage", f"{filtered['coverage'].mean():.2%}")

# ----------------- Headline answer card -----------------

st.header("Research-question answer")
st.caption(
    "**Does TheFinAI's financial instruction tuning help?** "
    "Compares the focal model (plutus-8B-instruct, the published successor to FinLLaMA-instruct) "
    "to the strongest non-financial 7-8B LLM on the same dataset."
)
ans_cols = st.columns(2)
focal_short = "plutus8b"
fallback_focal = "finllama"
non_fin_models = ("qwen25_7b", "mistral7b")

for col, ds in zip(ans_cols, ["FPB", "FiQA"]):
    with col:
        with st.container(border=True):
            st.markdown(f"### {ds}")
            ds_df = filtered[filtered["dataset"] == ds]
            focal_runs = ds_df[ds_df["model"].isin([focal_short, fallback_focal])]
            non_fin_runs = ds_df[ds_df["model"].isin(non_fin_models)]
            finbert_runs = ds_df[ds_df["model"] == "finbert"]

            if focal_runs.empty:
                st.info("Focal model (plutus / FinLLaMA) hasn't run yet on this dataset.")
            elif non_fin_runs.empty:
                st.info("Need at least one non-financial LLM run to compare.")
            else:
                focal_best = focal_runs.loc[focal_runs["f1_macro"].idxmax()]
                non_fin_best = non_fin_runs.loc[non_fin_runs["f1_macro"].idxmax()]
                delta = focal_best["f1_macro"] - non_fin_best["f1_macro"]
                if delta > 0.02:
                    verdict = "✅ Yes — financial tuning helps"
                    color = "green"
                elif delta < -0.02:
                    verdict = "❌ No — general LLM is better"
                    color = "red"
                else:
                    verdict = "➖ Roughly equivalent"
                    color = "gray"
                st.markdown(f"**Verdict:** :{color}[{verdict}]")
                st.markdown(
                    f"**Best focal:** {MODEL_LABEL.get(focal_best['model'], focal_best['model'])} "
                    f"(template {focal_best['template']}, {int(focal_best['shots'])}-shot) → "
                    f"**F1m {focal_best['f1_macro']:.3f}**"
                )
                st.markdown(
                    f"**Best general LLM:** {MODEL_LABEL.get(non_fin_best['model'], non_fin_best['model'])} "
                    f"(template {non_fin_best['template']}, {int(non_fin_best['shots'])}-shot) → "
                    f"**F1m {non_fin_best['f1_macro']:.3f}**"
                )
                st.markdown(f"**Δ (focal − general):** {delta:+.3f} F1-macro points")
                if not finbert_runs.empty:
                    fb_best = finbert_runs.loc[finbert_runs["f1_macro"].idxmax()]
                    st.markdown(
                        f"_For reference, the small specialised classifier FinBERT scores "
                        f"**F1m {fb_best['f1_macro']:.3f}** on the same dataset._"
                    )

# ----------------- Headline figure -----------------

st.header("F1-macro — best config per model × dataset")
best = (
    filtered.sort_values("f1_macro", ascending=False)
    .groupby(["model", "model_label", "dataset"], as_index=False)
    .first()
)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=best, x="model_label", y="f1_macro", hue="dataset", ax=ax)
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.set_ylabel("F1-macro")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ----------------- Full table -----------------

st.header("All runs")
display = filtered[[
    "model_label", "dataset", "template", "shots", "n_samples",
    "accuracy", "f1_macro", "f1_weighted", "coverage", "runtime_s",
]].rename(columns={"model_label": "model"})
st.dataframe(
    display.sort_values(["dataset", "model", "template", "shots"]).reset_index(drop=True),
    use_container_width=True,
    height=min(60 + 35 * len(display), 600),
)

# ----------------- Per-class breakdown -----------------

st.header("Per-class precision / recall / F1")
st.caption(
    "Class-level numbers tell you *how* a model fails. "
    "A model with F1m 0.80 on FPB might be perfect on positive but useless on neutral — "
    "this view exposes that."
)


@st.cache_data
def per_class_table(_filtered_hash: int) -> pd.DataFrame:
    """Pull per-class metrics from results/summary/confusions/*.json and join with final_table."""
    cdir = ROOT / "results" / "summary" / "confusions"
    rows = []
    if not cdir.exists():
        return pd.DataFrame()
    for f in cdir.glob("*.json"):
        with open(f) as fh:
            cf = json.load(fh)
        run_id = cf.get("run_id", f.stem)
        per = cf.get("per_class", {})
        # parse run_id to get model/dataset/template/shots
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
    # Best config per (model, dataset)
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
            "Dataset for per-class view",
            sorted(pc_best["dataset"].unique()),
            key="pc_ds",
        )
        pc_show = pc_best[pc_best["dataset"] == ds_pick_pc]
        # Pivot for nice tabular display: rows = model × class, columns = P/R/F1
        pc_show = pc_show.sort_values(["model_label", "class"])
        st.dataframe(
            pc_show[["model_label", "class", "precision", "recall", "f1", "support"]]
            .rename(columns={"model_label": "model"})
            .reset_index(drop=True),
            use_container_width=True,
            height=min(60 + 35 * len(pc_show), 500),
        )

        # Heatmap of per-class F1
        pivot_f1 = pc_show.pivot(index="model_label", columns="class", values="f1")
        col_order = [c for c in LABEL_ORDER if c in pivot_f1.columns]
        pivot_f1 = pivot_f1[col_order]
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * len(pivot_f1))))
        sns.heatmap(
            pivot_f1, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, cbar_kws={"label": "F1"}, ax=ax,
        )
        ax.set_title(f"Per-class F1 — best config per model on {ds_pick_pc}")
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ----------------- Highlights for slides -----------------

st.header("Highlights for slides")
st.caption(
    "Auto-curated examples worth screenshotting for your deck. "
    "Sorted by how 'tellable' the case is: clear focal-model wins, prompt flips, and unanimous misses."
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

        # Score this case for "interestingness"
        focal_preds = preds_by_model.get("plutus8b", []) + preds_by_model.get("finllama", [])
        general_models = ("qwen25_7b", "mistral7b")
        general_preds = sum((preds_by_model.get(m, []) for m in general_models), [])
        any_focal_correct = any(p == gold_lbl for p in focal_preds)
        any_general_correct = any(p == gold_lbl for p in general_preds)
        any_focal_wrong = any(p != gold_lbl for p in focal_preds if p is not None)
        any_general_wrong = any(p != gold_lbl for p in general_preds if p is not None)

        score = 0
        tag = ""
        # 1) Focal wins where general fails — strongest case for financial tuning
        if any_focal_correct and any_general_wrong and not any_general_correct:
            score = 4
            tag = "Focal wins, generals miss"
        # 2) General wins where focal fails — counter-evidence
        elif any_general_correct and any_focal_wrong and not any_focal_correct:
            score = 3
            tag = "Generals win, focal misses"
        # 3) Prompt flip in same model
        else:
            for m, plist in preds_by_model.items():
                if len(set(plist)) > 1:
                    score = max(score, 2)
                    tag = tag or f"{MODEL_LABEL.get(m, m)} flips between prompts"
            # 4) Unanimous miss (every model wrong)
            all_preds_list = sum(preds_by_model.values(), [])
            non_null_preds = [p for p in all_preds_list if p is not None]
            if non_null_preds and all(p != gold_lbl for p in non_null_preds):
                score = max(score, 2)
                tag = tag or "Everyone misses"

        if score > 0:
            rows.append({
                "score": score,
                "tag": tag,
                "id": sid,
                "text": gs["text"],
                "gold": gold_lbl,
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


# ----------------- Per-example breakdown -----------------

st.header("Per-example breakdown")
st.caption(
    "Pick a single example. See every model's prediction next to the gold label — "
    "useful for spotting disagreements and surfacing concrete cases for the presentation."
)

c_reload, _ = st.columns([1, 6])
if c_reload.button("Reload predictions"):
    load_all_predictions.clear()
    load_gold.clear()

all_preds = load_all_predictions()
if all_preds.empty:
    st.info("No predictions yet — wait for the matrix to write at least one run.")
else:
    gold_map = load_gold()

    f1, f2, f3 = st.columns([2, 3, 2])
    ds_pick = f1.selectbox(
        "Dataset",
        sorted(all_preds["dataset"].dropna().unique()),
        key="example_ds",
    )
    text_search = f2.text_input("Search text contains", key="example_search")
    only_disagree = f3.checkbox("Only disagreements", value=False, key="example_disagree")

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
                st.markdown(f"**Sample id:** `{sel_id}`")
                st.markdown(f"**Text:** {gold_sample['text']}")
                gc1, gc2 = st.columns([1, 6])
                gc1.markdown(f"**Gold:** `{gold_sample['label']}`")

            sub = pool[pool["id"] == sel_id].copy()
            sub["model"] = sub["model"].map(MODEL_LABEL).fillna(sub["model"])
            sub["correct"] = sub["pred"] == gold_sample["label"]

            # Add baselines (FinBERT, VADER) explicitly even if they appear under different patterns
            view = sub[[
                "model", "template", "shots", "pred", "correct", "parse_ok", "raw_output",
            ]].rename(columns={"raw_output": "raw"}).reset_index(drop=True)

            # Style: highlight correct rows green, wrong red
            def _row_style(row):
                if row["pred"] is None or (isinstance(row["pred"], float) and pd.isna(row["pred"])):
                    return ["background-color: #ffe9c2"] * len(row)
                if row["correct"]:
                    return ["background-color: #d6f5d6"] * len(row)
                return ["background-color: #fde0e0"] * len(row)

            try:
                styled = view.style.apply(_row_style, axis=1)
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(view, use_container_width=True)

            # Quick disagreement summary
            n_correct = int(view["correct"].sum())
            n_total = len(view)
            n_parse_fail = int((~view["parse_ok"]).sum())
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Models correct", f"{n_correct} / {n_total}")
            sc2.metric("Models wrong", f"{n_total - n_correct - n_parse_fail}")
            sc3.metric("Parse failures", n_parse_fail)


# ----------------- Prompt sensitivity (LLMs only) -----------------

llm_models = ("plutus8b", "mistral7b", "qwen25_7b", "finllama")
llm = filtered[filtered["model"].isin(llm_models)]
if not llm.empty:
    st.header("Prompt sensitivity")
    st.caption(
        "ΔF1-macro between Template A and Template B per (model, shots, dataset). "
        "If the gap is large, the result depends on phrasing more than on the model."
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
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("Need both Template A and B runs for a model to show the gap.")

# ----------------- Few-shot effect (LLMs only) -----------------

if not llm.empty:
    st.header("Few-shot effect")
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
        st.dataframe(pivot2, use_container_width=True)
    else:
        st.info("Need both 0-shot and 3-shot runs to show the gap.")

# ----------------- Confusion matrix grid -----------------

st.header("Confusion matrices — best config per model × dataset")
confs = load_confusions()

models_in_view = sorted(best["model"].unique(), key=lambda m: MODEL_LABEL.get(m, m))
datasets_in_view = sorted(best["dataset"].unique())

for ds in datasets_in_view:
    st.subheader(f"Dataset: {ds}")
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
            ax.set_title(
                f"{MODEL_LABEL.get(m, m).split(' ')[0]}\n"
                f"f1m={r['f1_macro']:.2f}  acc={r['accuracy']:.2f}",
                fontsize=8,
            )
            ax.set_xlabel("predicted", fontsize=7)
            ax.set_ylabel("true", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ----------------- Error explorer -----------------

st.header("Error explorer — focal-model misses")
errors = load_errors()
if errors is None:
    st.info(
        "Run `scripts/error_analysis.py` after the LLM matrix to populate the error sample."
    )
else:
    st.caption(
        f"{len(errors)} sampled misclassifications from the focal-model run(s). "
        "Use these for hand-categorization of error types."
    )
    err_dataset = st.selectbox(
        "Filter by dataset", ["all"] + sorted(errors["dataset"].unique().tolist())
    )
    err_view = errors if err_dataset == "all" else errors[errors["dataset"] == err_dataset]
    st.dataframe(err_view, use_container_width=True, height=400)

    if len(err_view):
        st.subheader("Sample drill-down")
        idx = st.slider("Row", 0, len(err_view) - 1, 0)
        row = err_view.iloc[idx]
        with st.container(border=True):
            st.markdown(f"**Dataset:** {row['dataset']}  |  **id:** `{row['id']}`")
            st.markdown(f"**Text:** {row['text']}")
            cc1, cc2 = st.columns(2)
            cc1.markdown(f"**Gold:** `{row['gold']}`")
            cc2.markdown(f"**Predicted:** `{row['pred']}`")
            st.markdown(f"**Raw model output:** `{row['raw_output']}`")

# ----------------- Per-run drill-down -----------------

st.header("Per-run drill-down")
run_ids = sorted(filtered.apply(
    lambda r: (
        f"{r['model']}__{r['dataset']}__seed{int(r['seed'])}"
        if r["template"] == "-"
        else f"{r['model']}__{r['dataset']}__{r['template']}__{int(r['shots'])}shot__seed{int(r['seed'])}"
    ),
    axis=1,
).unique().tolist())

sel_run = st.selectbox("Pick a run", run_ids) if run_ids else None
if sel_run:
    preds = load_predictions(sel_run)
    if preds is None:
        st.warning(f"No predictions.jsonl for {sel_run}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions", len(preds))
        c2.metric("Parse OK", f"{preds['parse_ok'].mean():.2%}")
        c3.metric(
            "Mean latency",
            f"{preds['latency_ms'].mean():.0f} ms"
            if "latency_ms" in preds
            else "—",
        )
        st.dataframe(
            preds.head(50),
            use_container_width=True,
            height=300,
        )
        st.caption("Showing first 50 rows of predictions.jsonl")

st.divider()
st.caption(
    "Built for the LLMs in Finance seminar. "
    "Re-run `scripts/aggregate.py` and refresh this page to update the dashboard."
)
