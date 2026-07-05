# Report revisions applied (from review findings)

Two artifacts were rewritten with the review findings:
- **`report/report.md`** — the markdown source (edited in place; see git commit).
- **`report/3_implementation_report_REVISED.docx`** — the submission docx, rewritten
  from a copy (the original was open in Word / write-locked). Byte-verified: all 8
  figures preserved, zip intact, paragraph count unchanged (326 → 326).

To adopt: close Word, replace `3_implementation_report.docx` with the `_REVISED`
copy (or run **Review ▸ Compare** in Word against your original to see every change
as a redline), then **File ▸ Save As ▸ PDF** to refresh `3_implementation_report.pdf`.
Also refresh the copies in `report/_bundle/` and `report/finllama-seminar-final/`.

## Changes (each traces to a review finding)

| # | Where | Change | Finding |
|---|---|---|---|
| 1 | Header + Appendix A | Total spend **~$1.3 → ~$1.2**; **≈130 → ≈120 GPU-minutes** | Claim-verify: ledger sums to ~$1.15 / ~120 min |
| 2 | §3 objective table | FiNER-ORD **"200–300" → "300-sentence"** subsample | Claim-verify: all NER rows are n=300 |
| 3 | §5 caveats | **New (iv): FinBERT was fine-tuned *on* FPB** — its FPB/AllAgree scores are in-sample; treated as an upper bound; strengthens the thesis | Gotcha audit **[major]** — the one real disclosure gap |
| 4 | §5 baselines | Added **parity note** — baselines also re-scored on the identical 300-id LLM subsets (same ranking, gaps ≤0.02 F1) | Subsample-parity finding |
| 5 | §6 interpretation | FinBERT FPB win annotated **"on data it was trained on — see §5-iv"** | Same as #3 |
| 6 | §6.1 | **New paragraph**: templates D/F/H recover **+0.04 to +0.09 F1** (plutus FPB 0.66→0.75, FiQA 0.43→0.67); neutral-magnet mechanism; sharpens the thesis | Prompt-design experiment (your requested focus) |
| 7 | §9 limitations | **New caveat**: FiQA ±0.10 neutral band is untuned/unswept; sensitivity check is future work | Gotcha audit [minor] |
| 8 | §10 item 8 | Rewrote the test-suite item: accurate root cause (**unpinned scipy 1.17 + torch 2.11**, not "test ordering"), now **fixed via `scipy < 1.17`** | Code-verify + claim-verify |
| 9 | Appendix A env notes | Added the **`scipy < 1.17`** pin to reproduction instructions | Code-verify |

Nothing else in the report was altered — the 367/420 claims that verified exact were
left untouched. The three headline findings, all tables, and the CIs are unchanged.
