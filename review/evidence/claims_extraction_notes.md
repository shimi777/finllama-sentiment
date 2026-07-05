# Claims extraction — reconciliation notes

## Self-verification summary (docx)

Method: `review/scripts/verify_docx_numbers.py` independently unzips
`review/evidence/report_copy.docx`, pulls all `<w:t>` run text out of
`word/document.xml` with a **separate, minimal** regex
(`(?<![\w.])\d[\d,]*\.?\d*%?`), and diffs the resulting numeric-token set
against the values captured by `review/scripts/extract_docx_claims.py` in
`claims.csv`.

Final reconciliation (after fixes below):

- Distinct numeric tokens in XML (minimal regex): **154**
- Distinct claim values in claims.csv: **132**
- Total claim rows in claims.csv: **271**
- Remaining unmatched tokens: **1** (`12`) — confirmed non-metric, see below.

The gap between 154 distinct XML tokens and 132 distinct CSV values is
expected and correct: several XML tokens are legitimately excluded
(years, section numbers, a table's row/column index) and several distinct
XML tokens collapse onto the same claim value appearing in multiple
sentences (e.g. `0.10` used both for the FiQA neutral band and the dataset
description repeated in Appendix B).

## Bugs found and fixed during reconciliation

The first extractor pass produced 206 claims and missed 14 plausible
metric-like tokens on verification. Root causes, all fixed in
`extract_docx_claims.py`:

1. **CI-range decimals swallowed by the citation-bracket heuristic.**
   `is_citation_or_figure_or_section()` excluded any number immediately
   preceded by `[`, intended to drop bracketed citation markers like
   `[12]`. But the report expresses bootstrap CIs as `[0.776, 0.873]`,
   which was being excluded too. Fix: the bracket check now only excludes
   a bare integer (no decimal point) — `[0.776` is a CI bound and is kept,
   `[12]` (a citation) is still dropped. This recovered `0.776, 0.784,
   0.806, 0.873, 0.878, 0.894, 0.896, 0.908, 0.955, 0.964` — all real
   bootstrap-CI bounds in §6 and §7.

2. **Model/parameter sizes ("110M", "8B") had no matching pattern.**
   `RE_COUNT`'s trailing negative lookahead `(?![\w.])` blocks a match
   when the digits are immediately followed by a letter (as in `110M`),
   so parameter-count claims like "FinBERT (110M, in-domain)" were never
   captured. Fix: added `RE_PARAM_SIZE = r"(?<![\w.])\d{1,4}(?:\.\d+)?\s?[MB](?![\w])"`
   and wired it into both the match pass and `infer_metric_type` (tagged
   `count`). Recovered `110` (×3, FinBERT/FinBERT-tone parameter counts).

3. **Heading text was never scanned for claims.** Numbers inside heading
   paragraphs (e.g. "8.1 Sentiment — where plutus-8B misses (**30**
   hand-tagged FPB errors)") were skipped entirely because heading
   paragraphs `continue` before the sentence-splitting/claim-extraction
   step. Fix: headings are still excluded from claim-scanning as a whole
   number token *only for their leading section-numbering prefix*
   (`^\s*\d+(?:\.\d+)*\.?\s+`, e.g. the "8.1 " or "12. " at the very
   start); the remainder of the heading text is now scanned normally.
   Recovered `30` (hand-tagged FPB error count, §8.1 heading).

After these three fixes, re-running the extractor took claims.csv from 206
to 271 rows and closed 13 of the 14 misses.

## Remaining "miss": `12`

`12` appears exactly once in the XML, in the run text `"...bootstrap
confidence intervals on every F1. 12. Conclusion On four financi..."` —
this is the **section number of the "12. Conclusion" heading**, one of
the categories the task explicitly says to EXCLUDE ("section numbers").
python-docx correctly identifies "12. Conclusion" as a Heading-level
paragraph; the extractor strips exactly this leading numbering token
before scanning the heading body (see fix #3 above) and correctly does
not emit a claim for it. This is the intended behavior, not a bug.

## Other numbers deliberately excluded (by design, per task instructions)

- **Section/subsection numbers** in headings (e.g. "6.1", "6.2", "6.3",
  "6.4", "8.1", "8.2", "10", "11", "12" as heading prefixes) — excluded.
- **Years** (`\d{4}` matching 19xx/20xx, e.g. "FiNER-ORD 2023",
  "Financial PhraseBank" release years if mentioned, "2015" in
  "FIN/Alvarado-2015") — excluded via `RE_YEAR`. Note "Alvarado-2015" is
  a dataset name, not a standalone year claim.
- **Table/Figure numbers** ("Table 1", "Table 2", ... "Figure 3", etc.)
  — excluded via the section/figure/table prefix cues in
  `is_citation_or_figure_or_section()`.
- **Bracketed citation markers** (`[12]`-style, if any appear) — excluded
  (bare integer immediately inside `[...]` with no decimal point).

## False "misses" from the verifier itself (not extractor bugs)

- `0.1` initially appeared as an unmatched token in the first verifier
  pass, but this was the verifier's own minimal regex lacking a
  word-boundary guard: it was matching the substring `0.1` inside the
  version string **"Mistral-v0.1"**, which is a model-version label, not
  a metric value. Added the same `(?<![\w.])` lookbehind the real
  extractor already uses for `RE_DECIMAL`, and the false miss disappeared
  (the *real* `0.10` values — FiQA's neutral-band width — were already
  correctly captured in claims.csv the whole time).

## Files

- `review/scripts/extract_docx_claims.py` — dump + claims extractor (docx)
- `review/scripts/verify_docx_numbers.py` — independent self-verification pass
- `review/evidence/report_copy.docx` — working copy of `report/3_implementation_report.docx`
- `review/evidence/report_dump.md` — full paragraph+table dump
- `review/evidence/claims.csv` — 271 claim rows, columns: claim_id, section, sentence, metric_type, value, context_keys
