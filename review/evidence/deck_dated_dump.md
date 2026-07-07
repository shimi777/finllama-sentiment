# Deck dump: implementation_deck_2026-05-04-07-48.pptx


## Slide 1

- Does Financial Instruction Tuning Help LLMs?
- Replicating the sentiment-classification slice of Open-FinLLMs (Huang et al., 2024)
- Implementation Presentation  ·  30 min
- [Authors]   ·   LLMs in Finance Seminar

## Slide 2

- LLMs in Finance Seminar  ·  Implementation Presentation
- What we'll cover
- 1
- Objective and setup
- Research question, scope, reproducibility note
- 5-7 min
- 2
- Method and experimental design
- Datasets, models, prompts, matrix, defenses
- 8-10 min
- 3
- Results and error analysis
- Headline numbers, per-class behavior, concrete cases
- 10-12 min
- 4
- Lessons learned
- Reproducibility, prompt sensitivity, what we'd change
- 3-5 min

## Slide 3

- 1
- Objective and setup
- What we set out to test, and why it matters.

## Slide 4

- LLMs in Finance Seminar  ·  Implementation Presentation
- The question
- Open-FinLLMs (Huang et al., 2024) claims financial instruction tuning improves financial NLP.
- Does TheFinAI's financial instruction tuning beat the LLaMA-family base it was built on?
- Specifically, on sentiment classification — and how does it compare to classical baselines (FinBERT, VADER) and other 7-8B general-purpose LLMs (Mistral, Qwen2.5)?
- Reproduced
- FPB / FiQA-SA sentiment task; 4-bit inference of an 8B financial LLM vs. 7-8B general LLMs vs. classical baselines; F1-macro / accuracy reporting.
- Adapted
- Substituted plutus-8B-instruct (TheFinAI's 2025 successor) when FinLLaMA-instruct was unpublished; prompt templates A/B specifically chosen to measure sensitivity.
- Simplified
- Inference only (no fine-tuning, no multimodal, no trading sim). 300-sample subsample per dataset for the LLM matrix to fit Modal T4 budget. Two prompts, two shot counts.

## Slide 5

- LLMs in Finance Seminar  ·  Implementation Presentation
- Reproducibility note
- What changed since the paper, and what we did about it.
- The model in the paper, TheFinAI/FinLLaMA-instruct, has been unpublished by the authors.
- We substitute TheFinAI/plutus-8B-instruct (Feb 2025) — the same group's current 8B financial-instruction-tuned model
Same architecture (LLaMA-3 8B), same research direction, same goal: beat the base on financial NLP
This itself is a finding: LLM benchmarks are fragile when the artifact under test can disappear
We document the swap in our slides and report and re-run will be possible if FinLLaMA-instruct returns

## Slide 6

- 2
- Method and experimental design
- Pipeline, datasets, models, prompts, defenses against leakage.

## Slide 7

- LLMs in Finance Seminar  ·  Implementation Presentation
- Pipeline
- Single unified Sample schema; every downstream stage is dataset-agnostic.
- data_loader
- FPB + FiQA → unified dicts
- prompts
- Templates A/B + few-shot
- model runner
- Modal T4 (LLMs) or CPU (baselines)
- parser
- Canonical label or null
- evaluation
- F1m / accuracy / coverage
- Per-run directory: meta.json + predictions.jsonl + progress.json — checkpoint/resume built in
Run ID schema: {model}__{dataset}__{template}__{shots}shot__seed{seed}
Modal Volume caches HF weights — multi-GB download paid only once across the project
Coverage tracked separately so parse failures aren't force-mapped to 'neutral'

## Slide 8

- LLMs in Finance Seminar  ·  Implementation Presentation
- Datasets
- Two financial-sentiment benchmarks with different difficulty profiles.

**[Table — Slide 8]**

| Dataset | Source | Size | Labels | Note |
|---|---|---|---|---|
| Financial PhraseBank | Malo et al. 2014 | 690 test (sentences_75agree) | neg / neu / pos (3-class) | Annotator agreement ≥ 75% — clean test signal |
| FiQA-SA | FiQA 2018 | 1,173 test | score ∈ [-1, 1] → 3-class with ±0.10 band | Headlines + posts; noisier text, harder |
- LLM matrix uses a 300-sample subsample per dataset (Modal time budget); baselines use full test sets
Few-shot pool: only FPB train (FiQA has no train split) — preserves test purity
Leakage caveat: both datasets predate every LLM here by 7-12 years and were almost certainly seen at pretraining time

## Slide 9

- LLMs in Finance Seminar  ·  Implementation Presentation
- Models
- Five models across three families: financial-tuned, general 7-8B LLM, and classical.

**[Table — Slide 9]**

| Family | Model | Size / Type | Why included |
|---|---|---|---|
| Financial-tuned | TheFinAI/plutus-8B-instruct | 8B · LLaMA-3, 4-bit | Focal model — successor to FinLLaMA-instruct |
| General-purpose LLM | mistralai/Mistral-7B-Instruct-v0.3 | 7B · 4-bit | Open LLaMA-family-like comparator |
| General-purpose LLM | Qwen/Qwen2.5-7B-Instruct | 7B · 4-bit | Strong open instruct baseline |
| Classical specialised | ProsusAI/finbert | 110M · BERT classifier | Domain-specific small model — strong on FPB |
| Classical lexicon | VADER | Rule-based | Cheap floor — what does no learning give you? |
- All LLMs run in 4-bit on a single Modal T4 (16GB VRAM). LLaMA-3.1-8B-Instruct skipped — gated, no access yet.

## Slide 10

- LLMs in Finance Seminar  ·  Implementation Presentation
- Two prompt templates (pre-registered)
- Two templates is enough to measure prompt sensitivity — we don't pick the best, we report both.
- Template A — minimalist
- Classify the sentiment of the following financial text as positive, negative, or neutral.

Text: {text}

Sentiment:
- Template B — structured + definitions
- You are a financial analyst…
- Positive: favorable conditions, growth
- Negative: unfavorable, losses, risks
- Neutral: factual without clear implication

Text: {text}

Answer with one word only:

## Slide 11

- LLMs in Finance Seminar  ·  Implementation Presentation
- Experimental matrix and defenses
- 12 LLM runs (3 models × 2 datasets × 2 templates × {0, 3}-shot) + 4 baseline runs.
- Deterministic decode (temperature=0, do_sample=False) — same input ⇒ same output
Seed locked to 42 for every random draw: subsample, few-shot pick, baselines
Few-shot pool drawn only from FPB train — never from test
Two prompt templates pre-registered — both reported, neither selected post-hoc
Coverage as first-class metric — parse failures excluded from F1, not force-mapped to 'neutral'
Greedy parser with synonym map (bullish/bearish/optimistic…) — case-insensitive, first-match wins
- Reported metrics: F1-macro (primary), accuracy, F1-weighted, per-class precision/recall, confusion matrix, parsing coverage, runtime.

## Slide 12

- 3
- Results and error analysis
- Headline numbers, where models break, and concrete examples.

## Slide 13

- LLMs in Finance Seminar  ·  Implementation Presentation
- Headline: F1-macro by model
- Best configuration per model on each dataset. Bigger = better.
- Source: results/summary/final_table.csv  ·  Re-render with scripts/make_figures.py

## Slide 14

- LLMs in Finance Seminar  ·  Implementation Presentation
- Headline numbers — best config per model × dataset
- F1-macro is the primary metric (it is robust to the class imbalance in FPB and FiQA).

**[Table — Slide 14]**

| Model | FPB · F1m | FPB · Acc | FiQA · F1m | FiQA · Acc | Best config |
|---|---|---|---|---|---|
| FinBERT (specialised, 110M) | 0.925 | 0.935 | 0.482 | 0.498 | Best on FPB · loses on FiQA |
| Qwen2.5-7B-Instruct | 0.832 | 0.860 | 0.673 | 0.717 | Best LLM on FiQA: tpl B 0-shot |
| Mistral-7B-Instruct-v0.3 | 0.890 | 0.903 | 0.599 | 0.620 | Best LLM on FPB: tpl A 3-shot |
| plutus-8B-instruct (focal) | 0.829 | 0.851 | 0.597 | 0.657 | Mid-pack on both — see findings |
| VADER (lexicon) | 0.469 | 0.554 | 0.386 | 0.423 | — |
- Green-highlighted cells: per-dataset best.  ·  Source: results/summary/final_table.csv (dashboard at :8502).

## Slide 15

- LLMs in Finance Seminar  ·  Implementation Presentation
- Three findings the numbers tell us
- Read the table once — but interpret it three times.
- Financial instruction tuning did NOT pull ahead
- plutus-8B (focal financial-tuned model) scored 0.829 F1m on FPB and 0.597 on FiQA. Mistral-7B beat it on FPB (0.890) and Qwen2.5-7B beat it on FiQA (0.673). On the paper's central claim — that financial instruction tuning improves downstream NLP — our subset replication says: not clearly, on these two datasets.
- FinBERT dominates FPB; loses on FiQA — strong reverse on out-of-domain text
- On FPB the 110M specialised classifier (0.925 F1m) outperforms every 8B LLM. On FiQA it drops to 0.482 — Qwen2.5-7B beats it by 19 points. Domain-trained classifiers crush LLMs in-domain but generalise poorly to noisier text. A useful warning when picking models for production.
- Prompt sensitivity is comparable to model choice
- Mistral on FPB: Template A 0-shot 0.803 vs Template B 0-shot 0.689 — an 11-point gap from rewording alone. Qwen on FiQA: Template B 0-shot 0.673 vs Template A 0-shot 0.578 — 9 points. Plutus on FiQA: A 0-shot 0.597 vs B 0-shot 0.432 — 16 points. Single-prompt LLM benchmarks should not be trusted.

## Slide 16

- LLMs in Finance Seminar  ·  Implementation Presentation
- Per-class F1 — Financial PhraseBank
- Class-level numbers expose hidden weaknesses an overall F1 can hide.

## Slide 17

- LLMs in Finance Seminar  ·  Implementation Presentation
- Per-class F1 — FiQA-SA
- Note any class-specific gap between the financial model and the general LLMs.

## Slide 18

- LLMs in Finance Seminar  ·  Implementation Presentation
- Confusion matrices — best config per model × dataset
- Rows: true label  ·  Columns: predicted label  ·  Look for systematic mistakes.

## Slide 19

- LLMs in Finance Seminar  ·  Implementation Presentation
- Few-shot effect — does the financial model need fewer examples?
- ΔF1-macro when going from 0-shot to 3-shot. If the focal model gains less, its tuning already encodes the task.

## Slide 20

- LLMs in Finance Seminar  ·  Implementation Presentation
- Parsing coverage — does the model follow the format?
- Below ~95% is a flag. Coverage is reported separately so we don't paper over format failures.

## Slide 21

- LLMs in Finance Seminar  ·  Implementation Presentation
- Concrete cases — pulled from the dashboard 'Highlights' tab
- Each card shows one sample, the gold label, and how each model classified it.
- Focal wins, generals miss
- [paste sample text]
Gold: positive  ·  plutus-8B: positive  ·  Mistral: neutral  ·  Qwen2.5: neutral
- Generals win, focal misses
- [paste sample text]
Gold: negative  ·  Mistral: negative  ·  plutus-8B: neutral
- Prompt template flips a model
- [paste sample text]
Template A → positive  ·  Template B → negative  (same model, same shots)
- Everyone misses
- [paste sample text]
Gold: neutral  ·  All models predicted: positive (likely a calm-news-defaults-positive bias)

## Slide 22

- LLMs in Finance Seminar  ·  Implementation Presentation
- Where the system works
- Strengths grounded in our numbers and confusion matrices.
- FinBERT on FPB: clean labels + matching domain ⇒ 0.925 F1m and ≥0.93 on every class — strongest result anywhere in the matrix.
Mistral-7B on FPB with 3-shot Template A: 0.890 F1m, 0.903 accuracy — closes most of the FinBERT gap and shows that a general-purpose 7B can match a specialist when shown 3 in-context examples.
Qwen2.5-7B on FiQA Template B 0-shot: 0.673 F1m on the noisy/conversational FiQA where FinBERT collapses to 0.482 — generalist beats specialist out-of-domain.
All LLMs hit 100% parsing coverage with our greedy + synonym-aware parser — no run was poisoned by 'bullish'/'bearish'-style outputs that would have force-mapped to neutral.
VADER on FPB at 0.469 F1m provides a useful 'no-learning floor' — every other model is at least 17 points above it, validating the comparison.

## Slide 23

- LLMs in Finance Seminar  ·  Implementation Presentation
- Error categories — focal model misses (hand-categorized)
- Open results/summary/focal_error_sample.csv, fill the 'category' column, paste the breakdown here.
- Negation — model treats a negated negative as positive (e.g. 'did not miss expectations')
Numerical reasoning — small revenue beat treated as 'positive' on noise
Domain jargon / ambiguity — regulator commentary, hedged policy language
Factual neutrals → mistaken for positive — calm news flagged as upbeat by default
[+ any additional category that emerges during your hand review]
- Tip: when hand-tagging, look for cases where plutus-8B is right but Mistral/Qwen are wrong — those are the real evidence for or against the financial-tuning hypothesis.

## Slide 24

- LLMs in Finance Seminar  ·  Implementation Presentation
- Interactive analysis dashboard
- Live at localhost:8502 — every chart and table on these slides also updates as new runs finish.
- Includes: research-question verdict card, F1 comparison, prompt-sensitivity & few-shot tables, per-class breakdown, confusion matrices.

## Slide 25

- LLMs in Finance Seminar  ·  Implementation Presentation
- Per-example breakdown — every model on the same sentence
- Pick any sample, see how each (model, template, shots) classified it; green = correct, red = wrong.
- The 'highlights' tab auto-curates cases where models disagree — useful for picking concrete examples for a presentation.

## Slide 26

- 4
- Lessons learned
- Reproducibility, prompt sensitivity, and what we'd do differently.

## Slide 27

- LLMs in Finance Seminar  ·  Implementation Presentation
- Five lessons
- Reproducibility cost is real
- Notebook → script split, run-dir schema with checkpoint/resume, parser as a tested module, Modal Volume for cached weights — without these, a 12-run matrix isn't redoable.
- Models can vanish
- TheFinAI/FinLLaMA-instruct was unpublished between paper and our run. Benchmark results are only as durable as the artifacts they reference.
- Prompt sensitivity ≥ model choice (often)
- ΔF1 between Template A and B was [NUM] points — comparable to gaps between models. Single-prompt LLM benchmarks should not be trusted.
- Coverage matters more than people report
- Force-mapping parse failures to 'neutral' would overstate accuracy by [NUM] points. Most papers don't report it.
- Data leakage caveat
- FPB (2014) and FiQA (2018) almost certainly leaked into pretraining. Numbers here are an upper bound on real-world generalisation — a 2025 hold-out would be the honest test.

## Slide 28

- LLMs in Finance Seminar  ·  Implementation Presentation
- What we'd do differently
- Build a 2025 financial-news held-out set to break leakage from FPB/FiQA pretraining
Run three or four prompt templates instead of two — make sensitivity even more legible
Add an LLM-as-judge sanity pass over a 100-sample slice — catch label noise
Try few-shot examples drawn from the same dataset as the test (when available) — closer to real deployment
Include a smaller financial model (e.g. finma-7b) and a non-financial 8B baseline of the same era for a tighter family comparison

## Slide 29

- Bottom line
- Financial instruction tuning did NOT clearly help. plutus-8B scored 0.829 / 0.597 F1m on FPB / FiQA — beaten on FPB by Mistral-7B (0.890) and on FiQA by Qwen2.5-7B (0.673). FinBERT (110M, in-domain trained) crushed every 8B model on FPB but lost by 19 points on FiQA.
- Two takeaways for the audience to remember:
- 1. Don't trust LLM benchmarks reported with a single prompt — sensitivity here was 9-16 F1 points.
2. Specialised small models still beat 8B general LLMs in-domain — and the inverse on noisier text.
3. Reproducibility is structural: artefacts disappear, tokenizer formats break, gates close. Build for it.

## Slide 30

- Thank you
- Questions?
- Code · github.com/shimi777/finllama-sentiment
- Live dashboard · localhost:8502  (run dashboard/run.bat)