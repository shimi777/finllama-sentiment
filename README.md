# FinLLaMA-Instruct Sentiment Analysis

Seminar project ("LLMs in Finance"): does financial instruction tuning add
measurable value on sentiment classification vs. general-purpose and
domain-specific baselines? Inference only — no fine-tuning.

**Note on scope:** `TheFinAI/FinLLaMA-instruct` (the model under test) was
unpublished by its authors partway through this project (404 on the Hub). It
was substituted with `TheFinAI/plutus-8B-instruct`, the same group's current
8B financial-instruct model. `meta-llama/Llama-3.1-8B-Instruct` was skipped
(gated, no access this session). See `report/report.md` Appendix A/B for the
full reproducibility notes and substitution rationale.

## Models actually evaluated

| Role | Model |
|---|---|
| Financial-tuned focal model | `TheFinAI/plutus-8B-instruct` (substitute for FinLLaMA-instruct, 404) |
| General 7-8B LLM | `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct` |
| In-domain classifier | `ProsusAI/finbert`, `yiyanghkust/finbert-tone` |
| Lexicon baseline | VADER |

(`Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-Instruct-2507`, and GLiNER small/large were
also run for the separate NER track — see Appendix B of the report.)

## Datasets

- **Financial PhraseBank (FPB)**, 75%-agreement split, 690-sentence test set.
- **FiQA-SA** (`TheFinAI/fiqa-sentiment-classification`), 1,173-sentence test
  set, continuous sentiment score bucketed into positive/neutral/negative with
  a ±0.10 neutral band.

**Subsample parity:** the LLM matrix runs on a fixed 300-sample subset per
dataset (same sample ids across all LLM cells, for cost/time reasons). FinBERT
and VADER were scored on the *full* test sets **and** re-scored on the same
300-id subsets so the comparison is apples-to-apples — see
`review/evidence/parity_table.csv`. Conclusions are unchanged between full-set
and subsample scoring.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # includes the scipy<1.17 pin below
.venv\Scripts\python.exe -m pytest                             # ~146 tests, should be all green
```

> **scipy pin:** `requirements.txt` pins `scipy>=1.11,<1.17`. scipy 1.17's
> array-api probe raises a `TypeError` inside `array_api_compat` when
> `torch>=2.11` is importable in the same environment, which breaks
> `sklearn` imports and fails most of `tests/test_evaluation.py`. If you ever
> see that failure mode, `pip install "scipy<1.17"` fixes it.

Rebuild the committed summary tables from cached predictions (no GPU needed,
requires the full repo checkout — `results/predictions/` is not in the code
zip):

```bash
.venv\Scripts\python.exe scripts/aggregate.py           # -> results/summary/final_table.csv (FPB, FiQA)
.venv\Scripts\python.exe scripts/aggregate_ensemble.py  # -> results/summary/ensemble_table.csv
.venv\Scripts\python.exe scripts/aggregate_ner.py       # -> results/summary_ner/final_table_ner.csv (FiNER-ORD)
.venv\Scripts\python.exe scripts/bootstrap_ci.py        # -> results/summary/bootstrap_ci.csv
```

`scripts/aggregate.py` only aggregates `FPB`/`FiQA` runs; other datasets
under `results/predictions/` (e.g. `FPBall`, NER tracks) are intentionally
skipped with a printed pointer to the script that owns them
(`scripts/eval_allagree.py` / `run_allagree.py` for FPBall,
`scripts/aggregate_ner.py` for the FiNER-ORD NER benchmark).

### Dashboards

Two Streamlit dashboards, launched via their `run.bat`:

```bash
dashboard\run.bat        # sentiment dashboard, http://localhost:8502
dashboard_ner\run.bat    # NER dashboard, http://localhost:8503
```

### Reproducing the LLM inference (Modal)

Re-running the 8B-class models needs a GPU; the project targets Modal (T4)
with a hard budget guard:

```bash
.venv\Scripts\python.exe scripts/run_llm_matrix.py   # sentiment matrix; aborts before exceeding CAP_USD=$7
```

`CAP_USD` is set in `scripts/run_llm_matrix.py`; actual spend for the whole
study was ~$1.3 (well under Modal's $30/month free credit). Requires
`HF_TOKEN` set for gated weights (`meta-llama/Llama-3.1-8B-Instruct`, if
attempted).

## Results layout

All committed outputs live under `results/summary/` (raw per-run predictions
in `results/predictions/` are gitignored):

| File | Contents |
|---|---|
| `final_table.csv` | Per (model × dataset × template × shots) accuracy/F1/coverage, FPB + FiQA |
| `ensemble_table.csv` | Prompt-ensemble (majority vote across templates/shots) results |
| `fpb_agreement_comparison.csv` | 75%-agreement vs. 100%-agreement (unanimous) gold comparison |
| `bootstrap_ci.csv` | 95% bootstrap confidence intervals per F1 number |
| `summary_ner/final_table_ner.csv` | FiNER-ORD NER benchmark (GLiNER vs. LLMs), strict/partial/type-only F1 |
| `ner_table.csv` | FIN/Alvarado-2015 NER track (separate, exploratory) |

`review/evidence/parity_table.csv` documents the full-set-vs-subsample parity
check referenced above.

## Deck / submission bundle

- **`report/_bundle/`** is the authoritative submission bundle (presentations,
  implementation report, code zip).
- **`presentation/implementation_deck*.pptx`** / `Implementation_deck_rev*.pptx`
  are generated working snapshots from `presentation/build_deck.js`, useful
  for regenerating slides but not the submitted artifact.

## More detail

- `project_plan.md` — scope, module interfaces, run-naming/checkpoint format,
  Colab setup.
- `report/report.md` — full write-up, findings, and Appendix A
  (reproducibility) / Appendix B (datasets, models, artifacts).
- `STATUS.md` / `STATUS_NER.md` — hand-off notes from the runs that produced
  the committed results.
