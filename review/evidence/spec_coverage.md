# Spec-coverage verdict — pre-submission review

**Scope of this grade:** COVERAGE only (does each required section/deliverable *exist and get addressed*), not numerical correctness — a separate stage verifies the numbers.
**Authoritative deck:** `implementation_deck_2026-05-04-07-48.pptx` (30 slides) as dumped in `deck_dated_dump.md`. `deck_main_dump.md` is an unfinished template (slide 1 title identical, but downstream slides carry `[NUM]` / `[paste sample text]` placeholders); it is **not** graded here.
**Report:** `report_dump.md` (the submitted docx, 12 sections + 2 appendices).

---

## A. Handout's 6 implementation-presentation sections

| # | Handout section | Report section(s) | Deck slide(s) | Verdict | Evidence quote |
|---|---|---|---|---|---|
| 1 | **Objective** — "State exactly what part of the paper you reproduced, adapted, or simplified." | §3 "Objective — exactly what we reproduced, adapted, and simplified" (Table 1); Exec summary | Slides 3-5 (Objective & setup; The question; Reproducibility note) | **Met** | Report §3 table columns "Reproduced / Adapted / Simplified"; deck slide 4 has explicit **Reproduced / Adapted / Simplified** blocks. |
| 2 | **System design** — "Data, model(s), prompts, retrieval/fine-tuning choices, software stack, and major simplifications." | §4 "System design" (pipeline diagram, invariants, "Major simplifications") | Slides 6-11 (Pipeline, Datasets, Models, prompt templates, matrix) | **Met** | §4: "Software stack. transformers + bitsandbytes (4-bit) on a Modal T4 container… Major simplifications: 4-bit (not fp16); subsampled test sets…" |
| 3 | **Experimental design** — "Task definition, metrics, baselines, train/test split, and steps taken to avoid leakage." | §5 "Experimental design" (Metrics; Coverage; Determinism; Baselines; "Leakage and contamination") | Slide 11 "Experimental matrix and defenses"; slides 8-10 | **Met** | §5: "Leakage and contamination — the honest caveats. (i) These benchmarks are old and public; any of these models may have seen FPB/FiQA/FIN…" |
| 4 | **Results** — "Main quantitative findings, comparisons with baselines, and representative examples." | §6 (sentiment, Tables 2-5), §7 (NER, Tables 6-7) | Slides 12-25 (headline table, per-class, confusion matrices, few-shot, coverage, concrete cases) | **Met** | §6 Table 2 headline F1 across 6 models both datasets; deck slide 14 same table with baselines. |
| 5 | **Error analysis** — "Where the system works, where it fails, and why." | §8 "Error analysis" (§8.1 30 hand-tagged FPB errors w/ taxonomy; §8.2 NER strict-match) | Slides 22-23 (Where the system works; Error categories) | **Met** (report); **Partial** (dated deck) | §8.1 taxonomy table (missed_positive_cue 15/50%, numerical_reasoning 8/27%…). Deck slide 23 is a **template stub**: "Open results/summary/focal_error_sample.csv, fill the 'category' column, paste the breakdown here." — categories not yet filled with the real counts. |
| 6 | **Lessons learned** — "Implementation challenges, reproducibility issues, and what you would improve next." | §10 "What we got wrong", §11 "Lessons learned", §12 Conclusion | Slides 26-29 (Five lessons; What we'd do differently; Bottom line) | **Met** (report); **Partial** (dated deck) | §11 five numbered lessons + "What we'd do with another month". Deck slide 27 still carries two `[NUM]` placeholders ("ΔF1… was [NUM] points"; "overstate accuracy by [NUM] points"). |

**Section coverage: all 6 present in the report.** Two sections (Error analysis, Lessons) are *structurally present but numerically unfilled* on the dated deck (placeholder text on slides 23 and 27). The report is the complete artifact; the dated deck is a near-final draft with a handful of un-pasted numbers.

---

## B. Handout's 4 general expectations

| Expectation | Where addressed | Verdict | Evidence quote |
|---|---|---|---|
| **Clear structure + figures/diagrams** | Report: pipeline ASCII diagram §4, 9 tables, confusion matrices referenced; deck: 30 structured slides, pipeline slide 7, bar/confusion/coverage figure slides 13/16-20 | **Met** | §4 renders a `data_loader → prompts → runner → parser → evaluation` diagram; deck slide 13 "Headline: F1-macro by model… Re-render with scripts/make_figures.py". (Figure PNGs live under `presentation/key_figures/` in worktrees and in `report/figures/` on the branch — see Part D note.) |
| **Interpret results, don't just copy tables** | §6 "Interpretation", §6.1-6.4 narratives, §7 Interpretation, deck slide 15 "Three findings" | **Met** | §6.4: "most of their 'errors' on the 75% set were on the sentences humans themselves disagreed about, a textbook demonstration that label quality… sets the ceiling." Deck slide 15 "Read the table once — but interpret it three times." |
| **Explicit assumptions / limitations / bias / leakage** | §5 (leakage), §9 "Critical assessment and limitations", deck slides 8/27 | **Met** | §9: "Possible benchmark contamination. FPB/FiQA/FIN are old and public; pre-training exposure would inflate every model and is unmeasurable here." |
| **Baselines + honest failure discussion** | Baselines: FinBERT/FinBERT-tone/VADER + GLiNER throughout; failures §8, §10 "What we got wrong" | **Met** | §10 logs 8 concrete missteps incl. "Briefly committed a real HF_TOKEN… rotate immediately"; Exec summary: "A dedicated 'What we got wrong' section (§10) documents the missteps." |

All 4 general expectations **met** in the report. The handout's own priority ("rigorous thinking and critical evaluation matter more" than polish) is well served — the report is unusually candid about failure.

---

## C. Time allocation vs dated-deck slide distribution (30 min, 30 slides)

Handout suggested allocation vs. deck section design (the deck's own agenda slide 2 mirrors the handout timings):

| Handout section | Suggested time | Deck section | Slides (content, excl. dividers) | Slide count | Implied pace |
|---|---|---|---|---|---|
| Objective & setup | 5-7 min | 1 (slides 3-5) | 4-5 | 2 content | Light — fine |
| Method & experimental design | 8-10 min | 2 (slides 6-11) | 7-11 | 6 content | Well-matched |
| Results & error analysis | 10-12 min | 3 (slides 12-25) | 13-25 | 12 content | Heaviest — matches "10-12 min" but is the risk zone |
| Lessons & conclusion | 3-5 min | 4 (slides 26-30) | 27-30 | 4 content | Slightly heavy for 3-5 min |

**Is 30 slides too many for 30 minutes?** Borderline-high. A ~1 min/slide average is aggressive but not unreasonable *because* several slides are section dividers (3, 6, 12, 26) and backup/dashboard slides (24, 25) that can be skipped or shown briefly. The **Results block (slides 12-25 = ~13 slides) is the crowding risk**: at 10-12 min that is <1 min/slide, and slides 16-21 (per-class FPB, per-class FiQA, confusion matrices, few-shot, coverage, concrete cases) are content-dense. **Recommendation:** treat slides 24-25 (dashboard tour) as optional/appendix and be ready to compress the Results figure slides, or the deck will overrun. The *weighting* across sections is sensible and matches the handout; the *absolute count* in Results is the thing to rehearse against a timer.

Verdict: **weighting sensible; total count on the high side — manageable with divider/backup slides treated as skippable.**

---

## D. project_plan.md §11 deliverables checklist + §2 scope deviations

### §11 deliverables

| Deliverable (§11) | Verdict | Evidence / note |
|---|---|---|
| Public git repo w/ README that runs everything from Colab | **Partial** | Repo exists (deck slide 30: "Code · github.com/shimi777/finllama-sentiment"). BUT `README.md` "הרצה (Colab T4)" section is a stub: **"TODO: הוראות להרצה מ-Colab…"**. Reproduction instructions *do* exist — in the **report Appendix A** (venv + `pip install -r requirements.txt` + script sequence + "re-run inference on Modal T4… costs ~$1, needs HF_TOKEN"). So repro guidance is delivered, but via the report, not the README, and it targets **Modal**, not Colab (a defensible substitution — see §2 deviation). README should be updated to match Appendix A before final submission. |
| Final results table (`results/summary/final_table.csv`) | **Met** | Present in working tree at `results/summary/final_table.csv`; report Appendix A: "python scripts/aggregate.py → results/summary/final_table.csv". |
| 4-5 key figures (confusion matrices, bar charts, error examples) | **Met** | Figure set exists: `f1_comparison.png`, `confusion_grid.png`, `per_class_f1_{FPB,FiQA}.png`, `fewshot_effect.png`, `coverage_heatmap.png` (found under `presentation/key_figures/`; committed under `report/figures/` on the `prompt-ensemble-improvement` branch per branch_state.md). Well over 4-5. |
| 30-min PPTX/Keynote deck | **Met** (with §A caveats) | `implementation_deck_2026-05-04-07-48.pptx`, 30 slides, agenda slide 2 explicitly time-boxed. Two content placeholders unfilled (slides 23, 27). |
| At least one rehearsal before the real talk | **Cannot verify / Missing evidence** | No artifact in the reviewed inputs attests to a rehearsal. Not a repo deliverable per se; flag as unconfirmed. |

**Deliverables are met or partial; none missing entirely.** The two soft spots are the README Colab stub (content exists elsewhere) and unverifiable rehearsal.

### §2 scope deviations — and whether the report DISCLOSES them

| Plan §2 scope item | What actually happened | Disclosed? | Verdict |
|---|---|---|---|
| **3 shot settings: 0 / 3 / 5-shot** | Only **0 and 3-shot** run; 5-shot never executed | **Disclosed** | Fine. Report §3 Table 1: "Templates A/B… × {0, 3}-shot"; deck slide 4 "Two prompts, two shot counts." The plan's own `configs/experiment.yaml` still lists `shots: [0,3,5]` but the write-ups consistently say 0/3. Deviation is **stated, not hidden** → acceptable for a reproduction study. |
| **2 templates** | Report/final work actually used **3 templates (A/B/C)**; deck presents only A/B | **Disclosed** (expansion, not a cut) | Fine — this is *more* than planned. Report §3: "Templates A (minimal), B (analyst definition list), C (market-reaction)"; §6.3 ensemble uses A/B × {0,3}. Note: dated **deck** slide 10 says "Two prompt templates (pre-registered)" — the deck under-represents the report's Template-C/ensemble work, but that is a deck-completeness gap, not an undisclosed deviation. |
| **4 models: FinLLaMA-Instruct, LLaMA-3.1-8B, FinBERT, VADER** | FinLLaMA-instruct **404/unpublished** → substituted **plutus-8B**; LLaMA-3.1-8B **gated, no access** → dropped; added Mistral-7B, Qwen2.5/3, FinBERT-tone, GLiNER | **Disclosed prominently** | Fine — the substitution is a headline finding, not a silent swap. Report §3: "FinLLaMA-instruct returns 404 (un-published)… Models we could not evaluate (documented, not silently dropped): FinLLaMA-instruct (404), LLaMA-3.1-8B-Instruct (gated, no access)…". Deck slide 5 is a dedicated "Reproducibility note." |
| **Runtime: Colab T4** (plan §5/§15) | Ran on **Modal T4** instead | **Disclosed** | Fine — report Appendix A and §4 say Modal throughout ("Modal T4 container with a hard $7 budget guard"). Equivalent hardware (T4), different orchestration. README still says Colab → cosmetic inconsistency to fix. |

**No undisclosed deviations found.** Every departure from the §2 plan (5-shot dropped, model substitution, Colab→Modal) is stated in the report and/or deck. For a reproduction study this is exactly the right posture — deviations disclosed = no flag.

**One thing to flag for the correctness stage (coverage-adjacent):** the *dated deck* materially under-covers the report's later analyses — Template C, the 4-member **prompt ensemble** (§6.3), **bootstrap CIs**, the **75%-vs-100% AllAgree** re-run (§6.4), and the **FIN/Alvarado + FiNER-ORD NER** track (§7). The deck is a sentiment-only A/B story; the report is the fuller ensemble+NER story. Per `branch_state.md`, all those analyses are backed by artifacts on `prompt-ensemble-improvement` (not `master`). So coverage of those items exists **in the report and on the branch**, but a grader watching only the dated deck would not see them. This is a deck-freshness gap, not a missing deliverable.

---

## E. Verdict — is coverage submission-ready?

**Yes, with two fixable gaps. Coverage is substantially submission-ready.**

Every one of the handout's 6 implementation sections and all 4 general expectations is **met in the report**, which is the complete, candid, baseline-rich artifact the handout asks for (its §9 limitations and §10 "what we got wrong" exceed the bar). All §2 scope deviations are **disclosed**, so nothing is flagged as dishonest — appropriate for a reproduction study.

**Nothing is missing *entirely*.** The residual gaps are completeness/freshness, not absence:

1. **Dated deck has two unfilled placeholders** — slide 23 (error-category counts still say "fill the 'category' column, paste the breakdown here") and slide 27 (two `[NUM]` deltas). The numbers exist in report §8/§11; they just need pasting. *(Blocks a clean live talk; trivial to fix.)*
2. **README Colab section is a `TODO` stub.** Real run instructions live in report Appendix A but target Modal; the §11 deliverable specifically wants a Colab-runnable README. *(Update README to mirror Appendix A.)*
3. **Dated deck under-represents the report's ensemble / bootstrap-CI / AllAgree / NER work** — those are covered in the report and backed on `prompt-ensemble-improvement`, but a deck-only viewer misses them. Consider adding 2-4 slides or explicitly scoping the talk to sentiment-A/B.
4. **Rehearsal (§11)** — no evidence in the reviewed inputs; unverifiable, likely just undocumented.

None of these are content that does not exist; they are transcription/packaging tasks. **Coverage grade: submission-ready pending the deck placeholder fill and the README Colab update.**
