# Presentation outline (30 min)

Draft per project_plan.md §10. Fill `[NUM]` placeholders from `results/summary/final_table.csv` after the matrix completes. Italicized notes are for the speakers, not slides.

---

## Slide 1 — Title (0:00)
- FinLLaMA-Instruct on financial sentiment: does the financial instruction tuning earn its keep?
- Course / Seminar: LLMs in Finance · Implementation Presentation
- Authors / Date

## Slides 2-3 — Objective & setup (0:00 – 4:00)
- **The claim we're testing:** Open-FinLLMs (Huang et al. 2024) reports financial instruction tuning materially improves downstream NLP. We replicate the *sentiment classification* slice on a small benchmark and compare against the model it's built on (LLaMA-3.1 family) and simpler open baselines.
- **Reproducibility note:** The exact model from the paper, `TheFinAI/FinLLaMA-instruct`, has since been **unpublished by the authors**. We substitute `TheFinAI/plutus-8B-instruct` (Feb 2025) — the same group's current 8B financial-instruction-tuned model. This is itself a finding worth noting: LLM benchmarks are fragile when the artifacts under test can disappear.
- **Scope** (from §2 of the plan):
  - In: FPB + FiQA-SA, 5 models (plutus-8B-instruct as the focal financial model, Mistral-7B-Instruct, Qwen2.5-7B-Instruct, FinBERT, VADER), 2 prompt templates, {0, 3}-shot, F1-macro / accuracy / coverage
  - Out: fine-tuning, multimodal, trading sim, closed models
- **Constraints:** Colab T4 / Modal T4, inference only, 1-2 weeks total.

## Slide 4 — Pipeline (4:00 – 7:00)
- Diagram: `data_loader → prompts → LLM/baseline runner → parser → evaluation → summary`.
- Single unified `Sample` schema across both datasets — every downstream step is dataset-agnostic.
- Modal-hosted T4 container with cached weights volume; baselines run on CPU locally (free).
- Run-directory schema (§14 of plan): per-run `meta.json + predictions.jsonl + progress.json` enables checkpoint/resume.

## Slides 5-6 — Experimental design (7:00 – 12:00)
- **Matrix:** Model × Dataset × Template × Shots → 12 LLM runs + 4 baseline runs.
- **Datasets:**
  - FPB (`sentences_75agree`): 690-sample held-out test split, 2,763 train pool for few-shot.
  - FiQA-SA: 1,173 examples, score-bucketed at ±0.10 → neg/neu/pos.
  - LLM matrix uses a 300-sample subsample per dataset to keep T4 budget bounded; baselines use the full test sets.
- **Two prompt templates** (A: minimalist, B: structured with definition) — 2 templates is enough to *quantify prompt sensitivity*, not pick the best one.
- **Few-shot pool** is *only* FPB train (FiQA has no train). 3-shot = 1/1/1 balanced.
- **Defenses:**
  - Deterministic decode (`temperature=0`, `do_sample=False`).
  - Seed locked (`42`) for every random draw — subsample, few-shot, baselines.
  - Few-shot examples never come from the test set.
  - Pre-registered the two templates; we report both, not the best.
- **Coverage as first-class metric.** Parse failures aren't force-mapped to "neutral" (avoiding the implicit positive/negative-pessimism bias) — they're excluded from F1 and reported separately.

## Slides 7-10 — Results (12:00 – 22:00)
- **Headline figure:** F1-macro per model × dataset (best config per model).
  - File: `presentation/key_figures/f1_comparison.png`
- **Headline numbers** (fill from `final_table.csv`):
  - FPB: FinBERT [NUM], plutus-8B [NUM], Mistral [NUM], Qwen2.5 [NUM], VADER [NUM]
  - FiQA: FinBERT [NUM], plutus-8B [NUM], Mistral [NUM], Qwen2.5 [NUM], VADER [NUM]
- **Confusion-matrix grid:** one heatmap per (model × dataset) for the best config of each.
  - File: `presentation/key_figures/confusion_grid.png`
- **Prompt sensitivity:** Δ F1-macro between Template A and Template B per model. *If the gap is > 5 points, that's the story* — instruction tuning interacts strongly with surface form.
- **Few-shot effect:** 0-shot vs 3-shot Δ. Watch for FinLLaMA being *less* helped by few-shot than the LLaMA-family base — that would suggest the instruction tuning already encodes the task.
- **Coverage:** parsing-success rate per LLM × prompt (`coverage_heatmap.png`). Below ~95% is a flag.
- **3-4 concrete examples** on the *same* sentence — pick one each: clear win, clear loss, prompt-sensitive flip, sarcasm/negation case.

## Slides 11-12 — Error analysis (22:00 – 27:00)
- 30 plutus-8B misses sampled from `results/summary/focal_error_sample.csv`, hand-categorized into 3-4 buckets:
  - Negation (e.g. "did not miss expectations" → neg pred, gold pos)
  - Numerical reasoning (small revenue beat treated as positive on noise)
  - Domain jargon / ambiguity (regulator commentary, policy hedges)
  - Factual neutrals misclassified as positive (model defaults positive on calm news)
- 1-2 representative examples per bucket on slide.
- **Where plutus-8B wins where the general LLMs fail:** disagreement cases — typically domain-specific phrasing.
- **Where everyone fails:** ambiguous / conflicting signals; gold disagrees with intuition. Note inter-annotator-style noise as a ceiling.

## Slides 13-14 — Lessons learned (27:00 – 30:00)
- **Reproducibility cost is real.** Notebook → script split, run-dir schema, seeds, parser as a unit-tested module — without these the matrix isn't redoable.
- **Prompt sensitivity ≥ model choice (often).** The right way to report LLM benchmarks is *with multiple prompts*, not one.
- **Coverage matters.** Force-mapping parse failures to "neutral" would overstate accuracy by [NUM] pts.
- **Data leakage caveat.** FPB (2014) and FiQA (2018) are years older than every LLM here — they almost certainly leaked into pretraining. The numbers are an *upper bound* on real-world generalization.
- **What we'd do differently:** budget for a held-out 2025 news sample; an LLM-as-judge sanity check on the test set; report robustness via a third prompt rather than picking between two.
- **The headline answer to our research question:** [INSERT 1-sentence finding once numbers are in].

---

## Speaker prep notes (not on slides)

- Spend the most time on §11-12 (error analysis) — it's the part the audience can't get from a benchmark table.
- For each slide with [NUM]: if the matrix didn't finish for that cell, *say so on the slide* rather than blank-filling. Honest gaps > polish.
- Have the `final_table.csv` open in a side tab during Q&A.
