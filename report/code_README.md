# FinLLaMA / Open-FinLLMs reproduction — code

Inference-only reproduction of the *Open-FinLLMs* sentiment + NER results,
benchmarking a financial-tuned 8B (plutus-8B, substituted for the unpublished
FinLLaMA-instruct) against general LLMs, FinBERT/VADER, and GLiNER.

**This archive contains code only — no data, no model weights, no predictions.**
The committed summary tables (`results/summary/*.csv`) are included so every
figure and table in the report regenerates without a GPU.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Environment notes:
- `transformers >= 4.50` and the chat template are required for plutus-8B; Qwen3 needs `>= 4.51`.
- Set `HF_TOKEN` for gated weights (only needed to re-run inference).
- Seed = 42 everywhere.

## Reproduce the report's tables and figures (no GPU)

Rebuilds everything from the committed per-run summaries / predictions:

```bash
python scripts/aggregate.py            # results/summary/final_table.csv   (sentiment)
python scripts/aggregate_ensemble.py   # results/summary/ensemble_table.csv
python scripts/aggregate_ner.py        # results/summary_ner/final_table_ner.csv  (FiNER-ORD)
python scripts/summarize_ner.py        # results/summary/ner_table.csv     (FIN/Alvarado)

python scripts/make_figures.py         # f1_comparison, coverage_heatmap, confusion_grid, per_class_f1, fewshot_effect
python scripts/make_ensemble_figures.py# ensemble_vs_single, ensemble_coverage_tradeoff
python scripts/make_ner_figure.py      # ner_comparison
python scripts/tag_errors.py           # error taxonomy in focal_error_sample.csv
```

> Note: the aggregation scripts read `results/predictions/<run_id>/`, which is
> **not shipped** (it is the raw model output / data). With only the summary CSVs
> present, the figure scripts still run from those CSVs; full re-aggregation
> requires re-running inference first (below).

## Re-run inference (needs a GPU / Modal, ~$1 total)

```bash
python scripts/run_baselines.py        # FinBERT, FinBERT-tone, VADER (CPU/GPU)
python scripts/run_llm_matrix.py       # sentiment LLM matrix on Modal T4 (budget-guarded)
python scripts/run_ner.py              # GLiNER + FiNER-ORD
python scripts/run_ner_matrix.py       # FIN/Alvarado NER on Modal T4
```

## Tests

```bash
pytest -q
```

(Known environment wrinkle: the full run shows ~11 spurious failures in
`tests/test_evaluation.py` from a `scipy`/`array_api_compat` caching quirk under
test ordering — every test passes in isolation, e.g. `pytest tests/test_evaluation.py`.)

## Layout

```
src/                      pipeline library
  data_loader.py          FPB / FiQA loaders -> unified Sample dicts
  prompts.py              templates A/B/C + few-shot sampling
  parser.py               LLM-output -> canonical label (coverage-aware)
  evaluation.py           accuracy / macro-F1 / coverage / confusion
  ensemble.py             majority + weighted prompt-vote
  models/                 LLMRunner (4-bit), FinBERTRunner, VADERRunner
  ner/                    FiNER-ORD NER pipeline (package)
  ner_loader.py ...       FIN/Alvarado NER pipeline (flat modules)
scripts/                  aggregation, figure, inference, Modal drivers
configs/                  experiment.yaml, ner.yaml
results/summary/          committed CSVs + confusion matrices (the report's source of truth)
dashboard/ , dashboard_ner/   Streamlit result explorers
tests/                    pytest suite
```

## Dashboards (optional)

```bash
streamlit run dashboard/app.py          # sentiment results explorer
streamlit run dashboard_ner/app.py      # NER results explorer
```
