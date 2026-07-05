# Deck diff: main vs dated

- Main deck slides: 26
- Dated deck slides: 30

## Table-cell number differences (position-aligned by slide/table/row/col)

### Slide 14, Table 1, row='plutus-8B-instruct (focal)', col='Model'
- MAIN:  `plutus-8B-instruct (focal)`
- DATED: `FinBERT (specialised, 110M)`

### Slide 14, Table 1, row='plutus-8B-instruct (focal)', col='FPB · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.925`

### Slide 14, Table 1, row='plutus-8B-instruct (focal)', col='FPB · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.935`

### Slide 14, Table 1, row='plutus-8B-instruct (focal)', col='FiQA · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.482`

### Slide 14, Table 1, row='plutus-8B-instruct (focal)', col='FiQA · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.498`

### Slide 14, Table 1, row='Mistral-7B-Instruct-v0.3', col='FPB · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.832`

### Slide 14, Table 1, row='Mistral-7B-Instruct-v0.3', col='FPB · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.860`

### Slide 14, Table 1, row='Mistral-7B-Instruct-v0.3', col='FiQA · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.673`

### Slide 14, Table 1, row='Mistral-7B-Instruct-v0.3', col='FiQA · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.717`

### Slide 14, Table 1, row='Mistral-7B-Instruct-v0.3', col='Best config'
- MAIN:  `(missing)`
- DATED: `Best LLM on FiQA: tpl B 0-shot`

### Slide 14, Table 1, row='Qwen2.5-7B-Instruct', col='FPB · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.890`

### Slide 14, Table 1, row='Qwen2.5-7B-Instruct', col='FPB · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.903`

### Slide 14, Table 1, row='Qwen2.5-7B-Instruct', col='FiQA · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.599`

### Slide 14, Table 1, row='Qwen2.5-7B-Instruct', col='FiQA · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.620`

### Slide 14, Table 1, row='Qwen2.5-7B-Instruct', col='Best config'
- MAIN:  `(missing)`
- DATED: `Best LLM on FPB: tpl A 3-shot`

### Slide 14, Table 1, row='FinBERT (specialised classifier)', col='Model'
- MAIN:  `FinBERT (specialised classifier)`
- DATED: `plutus-8B-instruct (focal)`

### Slide 14, Table 1, row='FinBERT (specialised classifier)', col='FPB · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.829`

### Slide 14, Table 1, row='FinBERT (specialised classifier)', col='FPB · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.851`

### Slide 14, Table 1, row='FinBERT (specialised classifier)', col='FiQA · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.597`

### Slide 14, Table 1, row='FinBERT (specialised classifier)', col='FiQA · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.657`

### Slide 14, Table 1, row='VADER (lexicon)', col='FPB · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.469`

### Slide 14, Table 1, row='VADER (lexicon)', col='FPB · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.554`

### Slide 14, Table 1, row='VADER (lexicon)', col='FiQA · F1-macro'
- MAIN:  `[NUM]`
- DATED: `0.386`

### Slide 14, Table 1, row='VADER (lexicon)', col='FiQA · Accuracy'
- MAIN:  `[NUM]`
- DATED: `0.423`

## Numbers that differ between decks (slide-aligned best-effort line match, non-table text)

(none found by the slide-aligned line matcher)

## Full unified text diff (raw)

```diff
--- implementation_deck.pptx
+++ implementation_deck_2026-05-04-07-48.pptx
@@ -28,10 +28,12 @@
 [Slide 4] Open-FinLLMs (Huang et al., 2024) claims financial instruction tuning improves financial NLP.
 [Slide 4] Does TheFinAI's financial instruction tuning beat the LLaMA-family base it was built on?
 [Slide 4] Specifically, on sentiment classification — and how does it compare to classical baselines (FinBERT, VADER) and other 7-8B general-purpose LLMs (Mistral, Qwen2.5)?
-[Slide 4] Inference only — no fine-tuning, no multimodal, no trading simulation
-Two prompt templates × {0, 3}-shot — quantify prompt sensitivity, not pick the best
-Two datasets: Financial PhraseBank (FPB) and FiQA-SA
-Five models in the comparison; reported metrics: F1-macro, accuracy, parsing coverage
+[Slide 4] Reproduced
+[Slide 4] FPB / FiQA-SA sentiment task; 4-bit inference of an 8B financial LLM vs. 7-8B general LLMs vs. classical baselines; F1-macro / accuracy reporting.
+[Slide 4] Adapted
+[Slide 4] Substituted plutus-8B-instruct (TheFinAI's 2025 successor) when FinLLaMA-instruct was unpublished; prompt templates A/B specifically chosen to measure sensitivity.
+[Slide 4] Simplified
+[Slide 4] Inference only (no fine-tuning, no multimodal, no trading sim). 300-sample subsample per dataset for the LLM matrix to fit Modal T4 budget. Two prompts, two shot counts.
 [Slide 5] LLMs in Finance Seminar  ·  Implementation Presentation
 [Slide 5] Reproducibility note
 [Slide 5] What changed since the paper, and what we did about it.
@@ -145,106 +147,138 @@
 [Slide 13] Best configuration per model on each dataset. Bigger = better.
 [Slide 13] Source: results/summary/final_table.csv  ·  Re-render with scripts/make_figures.py
 [Slide 14] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 14] Headline numbers — fill from final_table.csv
-[Slide 14] Best F1-macro per model × dataset. [NUM] placeholders are filled from the live results table.
+[Slide 14] Headline numbers — best config per model × dataset
+[Slide 14] F1-macro is the primary metric (it is robust to the class imbalance in FPB and FiQA).
 [Slide 14] Model
-[Slide 14] FPB · F1-macro
-[Slide 14] FPB · Accuracy
-[Slide 14] FiQA · F1-macro
-[Slide 14] FiQA · Accuracy
+[Slide 14] FPB · F1m
+[Slide 14] FPB · Acc
+[Slide 14] FiQA · F1m
+[Slide 14] FiQA · Acc
+[Slide 14] Best config
+[Slide 14] FinBERT (specialised, 110M)
+[Slide 14] 0.925
+[Slide 14] 0.935
+[Slide 14] 0.482
+[Slide 14] 0.498
+[Slide 14] Best on FPB · loses on FiQA
+[Slide 14] Qwen2.5-7B-Instruct
+[Slide 14] 0.832
+[Slide 14] 0.860
+[Slide 14] 0.673
+[Slide 14] 0.717
+[Slide 14] Best LLM on FiQA: tpl B 0-shot
+[Slide 14] Mistral-7B-Instruct-v0.3
+[Slide 14] 0.890
+[Slide 14] 0.903
+[Slide 14] 0.599
+[Slide 14] 0.620
+[Slide 14] Best LLM on FPB: tpl A 3-shot
 [Slide 14] plutus-8B-instruct (focal)
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] Mistral-7B-Instruct-v0.3
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] Qwen2.5-7B-Instruct
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] FinBERT (specialised classifier)
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
+[Slide 14] 0.829
+[Slide 14] 0.851
+[Slide 14] 0.597
+[Slide 14] 0.657
+[Slide 14] Mid-pack on both — see findings
 [Slide 14] VADER (lexicon)
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] [NUM]
-[Slide 14] Pull live numbers from results/summary/final_table.csv (visible in the dashboard at :8502).
+[Slide 14] 0.469
+[Slide 14] 0.554
+[Slide 14] 0.386
+[Slide 14] 0.423
+[Slide 14] —
+[Slide 14] Green-highlighted cells: per-dataset best.  ·  Source: results/summary/final_table.csv (dashboard at :8502).
 [Slide 15] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 15] Per-class F1 — Financial PhraseBank
-[Slide 15] Class-level numbers expose hidden weaknesses an overall F1 can hide.
+[Slide 15] Three findings the numbers tell us
+[Slide 15] Read the table once — but interpret it three times.
+[Slide 15] Financial instruction tuning did NOT pull ahead
+[Slide 15] plutus-8B (focal financial-tuned model) scored 0.829 F1m on FPB and 0.597 on FiQA. Mistral-7B beat it on FPB (0.890) and Qwen2.5-7B beat it on FiQA (0.673). On the paper's central claim — that financial instruction tuning improves downstream NLP — our subset replication says: not clearly, on these two datasets.
+[Slide 15] FinBERT dominates FPB; loses on FiQA — strong reverse on out-of-domain text
+[Slide 15] On FPB the 110M specialised classifier (0.925 F1m) outperforms every 8B LLM. On FiQA it drops to 0.482 — Qwen2.5-7B beats it by 19 points. Domain-trained classifiers crush LLMs in-domain but generalise poorly to noisier text. A useful warning when picking models for production.
+[Slide 15] Prompt sensitivity is comparable to model choice
+[Slide 15] Mistral on FPB: Template A 0-shot 0.803 vs Template B 0-shot 0.689 — an 11-point gap from rewording alone. Qwen on FiQA: Template B 0-shot 0.673 vs Template A 0-shot 0.578 — 9 points. Plutus on FiQA: A 0-shot 0.597 vs B 0-shot 0.432 — 16 points. Single-prompt LLM benchmarks should not be trusted.
 [Slide 16] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 16] Per-class F1 — FiQA-SA
-[Slide 16] Note any class-specific gap between the financial model and the general LLMs.
+[Slide 16] Per-class F1 — Financial PhraseBank
+[Slide 16] Class-level numbers expose hidden weaknesses an overall F1 can hide.
 [Slide 17] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 17] Confusion matrices — best config per model × dataset
-[Slide 17] Rows: true label  ·  Columns: predicted label  ·  Look for systematic mistakes.
+[Slide 17] Per-class F1 — FiQA-SA
+[Slide 17] Note any class-specific gap between the financial model and the general LLMs.
 [Slide 18] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 18] Few-shot effect — does the financial model need fewer examples?
-[Slide 18] ΔF1-macro when going from 0-shot to 3-shot. If the focal model gains less, its tuning already encodes the task.
+[Slide 18] Confusion matrices — best config per model × dataset
+[Slide 18] Rows: true label  ·  Columns: predicted label  ·  Look for systematic mistakes.
 [Slide 19] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 19] Parsing coverage — does the model follow the format?
-[Slide 19] Below ~95% is a flag. Coverage is reported separately so we don't paper over format failures.
+[Slide 19] Few-shot effect — does the financial model need fewer examples?
+[Slide 19] ΔF1-macro when going from 0-shot to 3-shot. If the focal model gains less, its tuning already encodes the task.
 [Slide 20] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 20] Concrete cases — pulled from the dashboard 'Highlights' tab
-[Slide 20] Each card shows one sample, the gold label, and how each model classified it.
-[Slide 20] Focal wins, generals miss
-[Slide 20] [paste sample text]
+[Slide 20] Parsing coverage — does the model follow the format?
+[Slide 20] Below ~95% is a flag. Coverage is reported separately so we don't paper over format failures.
+[Slide 21] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 21] Concrete cases — pulled from the dashboard 'Highlights' tab
+[Slide 21] Each card shows one sample, the gold label, and how each model classified it.
+[Slide 21] Focal wins, generals miss
+[Slide 21] [paste sample text]
 Gold: positive  ·  plutus-8B: positive  ·  Mistral: neutral  ·  Qwen2.5: neutral
-[Slide 20] Generals win, focal misses
-[Slide 20] [paste sample text]
+[Slide 21] Generals win, focal misses
+[Slide 21] [paste sample text]
 Gold: negative  ·  Mistral: negative  ·  plutus-8B: neutral
-[Slide 20] Prompt template flips a model
-[Slide 20] [paste sample text]
+[Slide 21] Prompt template flips a model
+[Slide 21] [paste sample text]
 Template A → positive  ·  Template B → negative  (same model, same shots)
-[Slide 20] Everyone misses
-[Slide 20] [paste sample text]
+[Slide 21] Everyone misses
+[Slide 21] [paste sample text]
 Gold: neutral  ·  All models predicted: positive (likely a calm-news-defaults-positive bias)
-[Slide 21] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 21] Error categories — focal model misses (hand-categorized)
-[Slide 21] Open results/summary/focal_error_sample.csv, fill the 'category' column, paste the breakdown here.
-[Slide 21] Negation — model treats a negated negative as positive (e.g. 'did not miss expectations')
+[Slide 22] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 22] Where the system works
+[Slide 22] Strengths grounded in our numbers and confusion matrices.
+[Slide 22] FinBERT on FPB: clean labels + matching domain ⇒ 0.925 F1m and ≥0.93 on every class — strongest result anywhere in the matrix.
+Mistral-7B on FPB with 3-shot Template A: 0.890 F1m, 0.903 accuracy — closes most of the FinBERT gap and shows that a general-purpose 7B can match a specialist when shown 3 in-context examples.
+Qwen2.5-7B on FiQA Template B 0-shot: 0.673 F1m on the noisy/conversational FiQA where FinBERT collapses to 0.482 — generalist beats specialist out-of-domain.
+All LLMs hit 100% parsing coverage with our greedy + synonym-aware parser — no run was poisoned by 'bullish'/'bearish'-style outputs that would have force-mapped to neutral.
+VADER on FPB at 0.469 F1m provides a useful 'no-learning floor' — every other model is at least 17 points above it, validating the comparison.
+[Slide 23] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 23] Error categories — focal model misses (hand-categorized)
+[Slide 23] Open results/summary/focal_error_sample.csv, fill the 'category' column, paste the breakdown here.
+[Slide 23] Negation — model treats a negated negative as positive (e.g. 'did not miss expectations')
 Numerical reasoning — small revenue beat treated as 'positive' on noise
 Domain jargon / ambiguity — regulator commentary, hedged policy language
 Factual neutrals → mistaken for positive — calm news flagged as upbeat by default
 [+ any additional category that emerges during your hand review]
-[Slide 21] Tip: when hand-tagging, look for cases where plutus-8B is right but Mistral/Qwen are wrong — those are the real evidence for or against the financial-tuning hypothesis.
-[Slide 22] 4
-[Slide 22] Lessons learned
-[Slide 22] Reproducibility, prompt sensitivity, and what we'd do differently.
-[Slide 23] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 23] Five lessons
-[Slide 23] Reproducibility cost is real
-[Slide 23] Notebook → script split, run-dir schema with checkpoint/resume, parser as a tested module, Modal Volume for cached weights — without these, a 12-run matrix isn't redoable.
-[Slide 23] Models can vanish
-[Slide 23] TheFinAI/FinLLaMA-instruct was unpublished between paper and our run. Benchmark results are only as durable as the artifacts they reference.
-[Slide 23] Prompt sensitivity ≥ model choice (often)
-[Slide 23] ΔF1 between Template A and B was [NUM] points — comparable to gaps between models. Single-prompt LLM benchmarks should not be trusted.
-[Slide 23] Coverage matters more than people report
-[Slide 23] Force-mapping parse failures to 'neutral' would overstate accuracy by [NUM] points. Most papers don't report it.
-[Slide 23] Data leakage caveat
-[Slide 23] FPB (2014) and FiQA (2018) almost certainly leaked into pretraining. Numbers here are an upper bound on real-world generalisation — a 2025 hold-out would be the honest test.
+[Slide 23] Tip: when hand-tagging, look for cases where plutus-8B is right but Mistral/Qwen are wrong — those are the real evidence for or against the financial-tuning hypothesis.
 [Slide 24] LLMs in Finance Seminar  ·  Implementation Presentation
-[Slide 24] What we'd do differently
-[Slide 24] Build a 2025 financial-news held-out set to break leakage from FPB/FiQA pretraining
+[Slide 24] Interactive analysis dashboard
+[Slide 24] Live at localhost:8502 — every chart and table on these slides also updates as new runs finish.
+[Slide 24] Includes: research-question verdict card, F1 comparison, prompt-sensitivity & few-shot tables, per-class breakdown, confusion matrices.
+[Slide 25] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 25] Per-example breakdown — every model on the same sentence
+[Slide 25] Pick any sample, see how each (model, template, shots) classified it; green = correct, red = wrong.
+[Slide 25] The 'highlights' tab auto-curates cases where models disagree — useful for picking concrete examples for a presentation.
+[Slide 26] 4
+[Slide 26] Lessons learned
+[Slide 26] Reproducibility, prompt sensitivity, and what we'd do differently.
+[Slide 27] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 27] Five lessons
+[Slide 27] Reproducibility cost is real
+[Slide 27] Notebook → script split, run-dir schema with checkpoint/resume, parser as a tested module, Modal Volume for cached weights — without these, a 12-run matrix isn't redoable.
+[Slide 27] Models can vanish
+[Slide 27] TheFinAI/FinLLaMA-instruct was unpublished between paper and our run. Benchmark results are only as durable as the artifacts they reference.
+[Slide 27] Prompt sensitivity ≥ model choice (often)
+[Slide 27] ΔF1 between Template A and B was [NUM] points — comparable to gaps between models. Single-prompt LLM benchmarks should not be trusted.
+[Slide 27] Coverage matters more than people report
+[Slide 27] Force-mapping parse failures to 'neutral' would overstate accuracy by [NUM] points. Most papers don't report it.
+[Slide 27] Data leakage caveat
+[Slide 27] FPB (2014) and FiQA (2018) almost certainly leaked into pretraining. Numbers here are an upper bound on real-world generalisation — a 2025 hold-out would be the honest test.
+[Slide 28] LLMs in Finance Seminar  ·  Implementation Presentation
+[Slide 28] What we'd do differently
+[Slide 28] Build a 2025 financial-news held-out set to break leakage from FPB/FiQA pretraining
 Run three or four prompt templates instead of two — make sensitivity even more legible
 Add an LLM-as-judge sanity pass over a 100-sample slice — catch label noise
 Try few-shot examples drawn from the same dataset as the test (when available) — closer to real deployment
 Include a smaller financial model (e.g. finma-7b) and a non-financial 8B baseline of the same era for a tighter family comparison
-[Slide 25] Bottom line
-[Slide 25] [Insert a one-sentence answer to the research question once the matrix completes — e.g. 'Financial instruction tuning gave plutus-8B a [NUM]-point F1 edge over the strongest general 7B LLM on FPB, but the small specialised FinBERT still outperformed every 8B model on this in-domain task.']
-[Slide 25] Two takeaways for the audience to remember:
-[Slide 25] 1. [primary takeaway — fill in once you have the numbers]
-2. [secondary takeaway — likely about prompt sensitivity or reproducibility]
-[Slide 26] Thank you
-[Slide 26] Questions?
-[Slide 26] Code · github.com/shimi777/finllama-sentiment
-[Slide 26] Live dashboard · localhost:8502  (run dashboard/run.bat)
+[Slide 29] Bottom line
+[Slide 29] Financial instruction tuning did NOT clearly help. plutus-8B scored 0.829 / 0.597 F1m on FPB / FiQA — beaten on FPB by Mistral-7B (0.890) and on FiQA by Qwen2.5-7B (0.673). FinBERT (110M, in-domain trained) crushed every 8B model on FPB but lost by 19 points on FiQA.
+[Slide 29] Two takeaways for the audience to remember:
+[Slide 29] 1. Don't trust LLM benchmarks reported with a single prompt — sensitivity here was 9-16 F1 points.
+2. Specialised small models still beat 8B general LLMs in-domain — and the inverse on noisier text.
+3. Reproducibility is structural: artefacts disappear, tokenizer formats break, gates close. Build for it.
+[Slide 30] Thank you
+[Slide 30] Questions?
+[Slide 30] Code · github.com/shimi777/finllama-sentiment
+[Slide 30] Live dashboard · localhost:8502  (run dashboard/run.bat)
```