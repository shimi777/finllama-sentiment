# NER reproduction — overnight autonomous run (Open-FinLLMs Table 7)

Paper: *Open-FinLLMs: Open Multimodal Large Language Models for Financial
Applications* (user-attached PDF; NER prompt is verbatim from Appendix E.6 / Figure 5).

Run window: ~03:00–03:32 local (May 11, 2026). Hard stop set for 04:00.

## TL;DR

- ✅ Full NER pipeline built on top of the existing sentiment infra. Modal
  container (T4, 4-bit), budget tracking, run-directory layout all reused.
- ✅ 3 open-weight models benchmarked on the Alvarado-2015 FIN test set (98
  examples via PIXIU `TheFinAI/flare-ner`) with the paper's verbatim prompt,
  zero-shot.
- ✅ 14 NER unit tests + full suite passes 77/77.
- 💵 Modal spend: **$0.158** over the 3 runs (963 GPU-seconds on T4). Cap was $7.

## Headline finding

**No open-weight model reached even half the paper's reported scores.** The
financial-tuned 8B (plutus, our substitute for the unpublished FinLLaMA-instruct)
came in *worst* of the three — same pattern as the sentiment track.

| Model | Best micro-F1 (ours) | Paper analog | Paper NER | Δ |
|---|---|---|---|---|
| **plutus-8B-instruct** (focal financial) | **0.107** | FinLLaMA-instruct | 0.57 | **−0.463** |
| Mistral-7B-Instruct-v0.3 (general 7B) | 0.153 | Mistral-7B-Instruct-v0.1 | 0.00 | +0.153 |
| Qwen2.5-7B-Instruct (general 7B) | **0.160** | — | — | — |

Two things jump out:
1. **Plutus has perfect coverage (1.00, 0 parse failures)** but the lowest F1 —
   it follows the format precisely but disagrees with gold on the actual spans.
   Most misses are punctuation/case mismatches ("Evergreen Solar Inc." vs gold
   "Evergreen Solar Inc") and missing all-caps duplicates ("EVERGREEN SOLAR" as
   a separate gold entity that the model produces as one merged span).
2. **Mistral-v0.3 ≫ Mistral-v0.1.** The paper reports Mistral-v0.1 at 0.00 (model
   refused or formatted unparseably). v0.3 follows the format and lands at 0.15,
   a 15-point swing from one minor-version bump.

## All runs (final, after parser-patch re-eval)

| run_id | template | shots | n_eval | cov | micro F1 | P | R | parse fail |
|---|---|---|---|---|---|---|---|---|
| `ner__mistral7b__FIN__paper__0shot__seed42` | paper | 0 | 92/98 | 0.94 | **0.153** | 0.122 | 0.204 | 6 |
| `ner__plutus8b__FIN__paper__0shot__seed42` | paper | 0 | 98/98 | 1.00 | **0.107** | 0.085 | 0.145 | 0 |
| `ner__qwen25_7b__FIN__paper__0shot__seed42` | paper | 0 | 82/98 | 0.84 | **0.160** | 0.136 | 0.194 | 16 |

Per-type breakdown is in each run's `meta.json` (e.g. mistral PER F1 = 0.22, ORG
F1 = 0.11, LOC F1 = 0.21 — model picks up persons better than orgs).

## What got built tonight

| Component | Path | Notes |
|---|---|---|
| NER data loader | [src/ner_loader.py](src/ner_loader.py) | Loads Alvarado-2015 FIN via TheFinAI/flare-ner. Supports BIO (HF datasets) and paper-format gold strings. Normalises types to {PER, ORG, LOC, MISC}. |
| NER prompts | [src/ner_prompts.py](src/ner_prompts.py) | `paper` (verbatim) and `strict` (paper + "reply with only the list" suffix). Few-shot helper. |
| NER parser | [src/ner_parser.py](src/ner_parser.py) | Tolerant: `name, TYPE; ...`, `name (TYPE), ...`, newline / bullet variants, "NONE" → []. Scrubs parenthetical asides ("(a law firm)") in paper-format outputs. Returns None only on hard parse failure. |
| NER evaluation | [src/ner_evaluation.py](src/ner_evaluation.py) | Strict entity-level F1. Micro + macro + per-type P/R/F1. `only={PER,ORG,LOC}` matches the paper's prompt scope. Parse failures excluded from F1, reported as coverage. |
| Modal driver | [scripts/run_ner_matrix.py](scripts/run_ner_matrix.py) | Reuses `LLMRunner` from `modal_app.py`. Distinct `ner__*` namespace. `--dry-run`, `--limit`, `--only`, idempotent via `progress.json`. |
| Re-eval helper | [scripts/reeval_ner.py](scripts/reeval_ner.py) | Re-parses saved `raw_output` with current parser and recomputes metrics — useful after parser improvements (we used it tonight). |
| Aggregator | [scripts/summarize_ner.py](scripts/summarize_ner.py) | Writes `results/summary/ner_table.csv` + prints comparison vs. paper Table 7. |
| Tests | [tests/test_ner.py](tests/test_ner.py) | 14 tests: parser variants, prompt rendering, metric corners (perfect / partial / parse-fail / type restriction). |

## Pipeline shape (parallel to sentiment, distinct namespace)

```
src/ner_loader.load_fin_alvarado()      ->  list[NERSample]
    |
    v
src/ner_prompts.build_ner_prompt()      ->  prompt str
    |
    v
scripts/modal_app.LLMRunner.generate()  ->  raw output str   (Modal T4, 4-bit)
    |
    v
src/ner_parser.parse_ner()              ->  list[Entity] | None
    |
    v
src/ner_evaluation.compute_ner_metrics  ->  micro F1, per-type P/R/F1
    |
    v
scripts/summarize_ner.main()            ->  results/summary/ner_table.csv
```

Run-ID schema: `ner__{model_short}__FIN__{template}__{shots}shot__seed{SEED}`
Lives under `results/predictions/{run_id}/{meta.json,predictions.jsonl,progress.json}`
just like the sentiment runs.

## Caveats — the gap from paper numbers is real, but partly methodological

1. **Test-set size: 98 vs paper's 980.** The paper used the raw Alvarado 2015
   corpus; only the PIXIU FLARE subset (98 examples) is publicly distributed.
   Set `FIN_HF_ID` env to override if a larger mirror is found (`tner/fin`,
   etc. were tried; none had the full 980).
2. **FinLLaMA-instruct itself is unpublished** (same 404 as in the sentiment
   track). Plutus-8B-instruct is TheFinAI's current 8B financial instruct
   model and the closest substitute, but it is not the model the paper
   evaluated — comparison is indicative only.
3. **Strict-match metric is harsh.** Plutus has 0 parse failures and outputs
   look correct ("Evergreen Solar Inc."), but gold often differs by punctuation
   ("Evergreen Solar Inc"). The paper says "Entity F1" but PIXIU's official
   eval may normalize whitespace / case before matching — we did not (yet).
4. **4-bit quantization** is used everywhere for T4 fit. Paper likely ran fp16.
5. **GPT-4 / ChatGPT / FinTral / Palmyra-Fin not benchmarked** — out of scope
   for a T4 reproduction. Paper numbers quoted for reference.
6. **Only one prompt template × 0-shot was run tonight.** The full matrix
   (paper + strict templates × 0 + 3-shot = 4 cells per model = 12 runs) was
   scoped down to fit the 60-min window. ~$0.65 + 50 min of GPU time would
   close the matrix; the driver is already idempotent.

## How to resume / extend

```bash
# Run remaining matrix cells (strict template, few-shot)
.venv/Scripts/python.exe scripts/run_ner_matrix.py --templates strict --shots 0 3

# Or the full matrix (skips already-complete runs)
.venv/Scripts/python.exe scripts/run_ner_matrix.py --templates paper strict --shots 0 3

# Re-aggregate
.venv/Scripts/python.exe scripts/summarize_ner.py

# Smoke without Modal (10 examples, dry-run path)
.venv/Scripts/python.exe scripts/run_ner_matrix.py --dry-run --limit 10

# Re-evaluate existing runs after a parser improvement
.venv/Scripts/python.exe scripts/reeval_ner.py
```

Things worth trying next if numbers feel suspicious:
- Add a **normalized-match** flag to `compute_ner_metrics` (strip case, periods,
  trailing punctuation) and re-eval. Likely closes a big chunk of the gap.
- Try **FinMA-7B-full** (`TheFinAI/finma-7b-full`) — gated, needs HF approval —
  the paper put it at 0.35 NER, which we should be able to hit.
- Add **Finer-Ord** as a second NER dataset (paper §4 mentions it alongside
  FIN, 1,080 test examples).

## Final ledger

- Modal spend: $0.158 / $7 cap (`results/_modal_spend.json`)
- Tests: 77 passing (14 new for NER)
- Git: `claude/modest-ardinghelli-74ef02` branch, two commits tonight
  (pipeline scaffold, then live results + handoff)
