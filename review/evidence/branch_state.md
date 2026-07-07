# Repo-state finding: the submission lives on an unmerged branch

**Date:** 2026-07-05 (review session)

## Summary

`master` (7dfcc98, 2026-05-31) is **five weeks behind** the actual submission state.
All of the report's "missing" analyses live on `prompt-ensemble-improvement`
(dcc6da1, committed 2026-07-05 21:04, also pushed to origin), which was branched
from master's tip and is **strictly ahead** (no divergence, fast-forwardable).

## What the branch adds over master (109 files, +21,391 lines)

| Report claim previously flagged "unbacked" | Backing artifact on the branch |
|---|---|
| Template C (market-reaction framing) | `src/prompts.py` TEMPLATES["C"] + `tests/test_prompts.py` |
| 4-member prompt ensemble (majority vote, CV-weighted) | `src/ensemble.py`, `scripts/aggregate_ensemble.py`, `tests/test_ensemble.py`, `results/summary/ensemble_table.csv` (30 rows), `confusions_ensemble/*` |
| Bootstrap CIs (2,000 resamples) | `scripts/bootstrap_ci.py` |
| 75%-vs-100% agreement (AllAgree) analysis | `scripts/run_allagree.py`, `scripts/eval_allagree.py`, `results/summary/fpb_agreement_comparison.csv` (n=460 full / n=195 subset) |
| FIN/Alvarado NER (98 examples) | `results/summary/ner_table.csv` (3 models, n_total=98), `scripts/run_ner_matrix.py` (from claude/modest-ardinghelli-74ef02 lineage) |
| FinBERT-tone baseline | `final_table.csv` +2 rows (yiyanghkust/finbert-tone) |
| Qwen3-4B/8B NER rows | re-evaluated `results/summary_ner/final_table_ner.csv` (richer schema: backend, cost, latency) |
| The submission bundle itself | `report/_bundle/` (paper-presentation PDF, implementation-presentation PDF, report DOCX+PDF, code zip), `report/figures/*` |

`report/3_implementation_report.docx` in the working tree is **byte-identical**
(git blob 403a25f) to the branch-committed copy.

## Consequences

1. **Anyone grading from master (or the GitHub default branch) cannot reproduce
   the report's §6.3 ensemble, §6.4 gold-label, bootstrap-CI, or FIN-NER numbers.**
   This was this review's single hardest finding: the claims are real and backed,
   but by a branch the deliverable never merged.
2. Master's `results/summary_ner/final_table_ner.csv` is stale (old schema,
   n_samples=200, slightly different F1s) vs the branch's re-evaluated table —
   the docx must be checked against the *branch* table.
3. Fix is trivial and safe: fast-forward master to `prompt-ensemble-improvement`
   (this review merged it into `review/pre-submission` instead; see commits).

## Other traces resolved

- **HF token:** `git log --all -G "hf_[A-Za-z0-9]{20,}"` finds nothing in any
  reachable commit — the briefly-committed token was amended away. Rotation still
  recommended (it was pasted in chat; STATUS.md item 7 already says so).
- **Lecturer's request (STATUS.md item 6):** Qwen3-8B was added to the **NER**
  track on the branch, but **no sentiment cells** exist for it. The transformers
  blocker is gone (NER ran it on Modal with 4.55). Still open.
- **configs/experiment.yaml** stale on both branches: models block lists
  never-run `finllama`/`llama31`, FiQA `hf_id` is the old ChanceFocus id,
  `templates: [A, B]` omits C, `shots: [0, 3, 5]` but 5-shot was never run.
  Only `scripts/run_baselines.py` reads this file (dataset params) — safe to fix.
