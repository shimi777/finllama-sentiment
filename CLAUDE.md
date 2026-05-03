# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hebrew-language seminar project ("LLMs in Finance") testing whether the financial instruction tuning of FinLLaMA-Instruct adds measurable value on sentiment classification vs. baselines (LLaMA-3.1-8B-Instruct, FinBERT, VADER) on FPB and FiQA-SA. Inference only — no fine-tuning. Designed to run on Google Colab T4 (16GB VRAM); local dev is fine for everything except the 8B LLM runs.

`project_plan.md` is the source of truth for scope, design decisions, module interfaces (§13), run-naming/checkpoint format (§14), and Colab setup (§15). Read the relevant section before changing module signatures, the run directory layout, or the experiment matrix.

## Commands

- Install: `pip install -r requirements.txt`
- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_parser.py`
- Run a single test: `pytest tests/test_parser.py::test_exact_positive`

`pytest.ini` sets `addopts = -p no:dash` to disable a noisy plugin — keep it.

There is no linter, type checker, or build step configured.

## Architecture

The codebase is a small library in `src/` plus four notebooks in `notebooks/` that drive experiments. All modules import as `from src...` (the `src` package is on the path because tests/notebooks run from repo root).

**Data flow:** `data_loader` → unified `Sample` dicts → `prompts.build_prompt` (LLMs only) → runner (`LLMRunner` / `FinBERTRunner` / `VADERRunner`) → `parser.parse` (LLMs only) → `Prediction` dicts → `evaluation.compute_metrics`.

**Two key invariants — do not break:**

1. **Unified `Sample` schema** (`src/data_loader.py`): every dataset loader returns `{id, text, label, dataset, split}` with label in `{"positive","neutral","negative"}`. FPB labels come straight from the file; FiQA's continuous score is bucketed via `neutral_band` (default ±0.10).
2. **Parse-failure handling** (`src/parser.py` + `src/evaluation.py`): when `parse()` returns `None`, the prediction has `parse_ok=False` and `pred_label=None`. `compute_metrics` *excludes* those from accuracy/F1 and reports them only in `coverage`. This avoids the "force-to-neutral" bias — never paper over a parse failure by defaulting to a label.

**Runner interface split:** `LLMRunner.generate()` returns raw `(text, latency_ms)` tuples — parsing happens outside. `FinBERTRunner.predict()` and `VADERRunner.predict()` return canonical labels directly (no parser needed). Treat `LLMRunner.unload()` as mandatory between LLM swaps on Colab — two 4-bit 8B models do not coexist on a 16GB T4.

**Run artifacts** live under `results/predictions/{run_id}/` with `meta.json`, `predictions.jsonl`, `progress.json`. Run ID schema and resume/checkpoint rules are in `project_plan.md` §14. `results/predictions/` is gitignored; only `results/summary/` is committed.

## Things that bite

- **FPB loader bypasses `datasets`**: it pulls the raw zip via `huggingface_hub.hf_hub_download` and parses `*75Agree*.txt` directly (latin-1 encoding, lines split on the last `@`). Recent commits (`b056044`, `a050987`) chased loader-script breakage — don't reintroduce `load_dataset("takala/financial_phrasebank", ...)`.
- **FiQA loader still uses `datasets.load_dataset`** with `TheFinAI/fiqa-sentiment-classification`. The `configs/experiment.yaml` `hf_id` for fiqa (`ChanceFocus/...`) is stale relative to the loader code — the loader's hardcoded id is what runs.
- **`HF_TOKEN` env var** is read by `load_fpb`; required for gated `meta-llama/Llama-3.1-8B-Instruct` access.
- **Tokenizer left-padding** is set in `LLMRunner.__init__` because batched generation requires it — don't switch to right-padding.
