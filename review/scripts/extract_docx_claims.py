"""
Stage 2 evidence extraction: dump the implementation report docx and pull out
every quantitative claim into a CSV for later verification against the
actual run artifacts.

Inputs:
  review/evidence/report_copy.docx   (a COPY of report/3_implementation_report.docx;
                                       never touch the original — it may be Word-locked)

Outputs:
  review/evidence/report_dump.md     full paragraph+table dump, headings preserved
  review/evidence/claims.csv         one row per quantitative claim found

Run:
  C:/python_projects/finllama-sentiment/.venv/Scripts/python.exe review/scripts/extract_docx_claims.py
"""
import csv
import re
import sys
from pathlib import Path

import docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCX_PATH = REPO_ROOT / "review" / "evidence" / "report_copy.docx"
DUMP_PATH = REPO_ROOT / "review" / "evidence" / "report_dump.md"
CSV_PATH = REPO_ROOT / "review" / "evidence" / "claims.csv"

# ---------------------------------------------------------------------------
# Number-matching regexes
# ---------------------------------------------------------------------------

# Decimal number, e.g. 0.925, 16.4, 1.0  (also matches integers as a fallback later)
RE_DECIMAL = re.compile(r"(?<![\w.])\d{1,4}\.\d+(?![\w])")
# Percentage, e.g. 92.5%, 43%
RE_PERCENT = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d+)?\s?%")
# Point-gap phrasing, e.g. "16.4-point", "16.4 point", "16-points"
RE_POINT_GAP = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d+)?[\s-]?(?:point|pt)s?\b", re.IGNORECASE)
# Money, e.g. $12,000  $3.50
RE_MONEY = re.compile(r"[$₪€]\s?\d[\d,]*(?:\.\d+)?")
# Time durations, e.g. "3.5 hours", "45 minutes", "12s", "300ms", "2 days"
RE_TIME = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s?(?:ms|milliseconds?|s|sec|secs?|seconds?|min|mins?|minutes?|hrs?|hours?|days?)\b",
    re.IGNORECASE,
)
# Bare integer counts (2-5 digits) — sample sizes, counts of rows/examples etc.
RE_COUNT = re.compile(r"(?<![\w.,])\d{2,6}(?![\w.])")
# Standalone bare decimal-like small integer (single digit) - counts like "3 templates"
RE_SMALL_INT_WITH_UNIT = re.compile(
    r"(?<![\w.])\d{1,2}(?=\s?(?:templates?|shots?|models?|datasets?|runs?|samples?|folds?|seeds?|classes?|labels?))",
    re.IGNORECASE,
)
# Model/parameter sizes, e.g. "110M", "8B", "7-8B"
RE_PARAM_SIZE = re.compile(r"(?<![\w.])\d{1,4}(?:\.\d+)?\s?[MB](?![\w])")

# Things to exclude entirely: years (1900-2099), typical section/figure/table numbering handled contextually.
RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")

MODEL_KEYWORDS = [
    "FinLLaMA", "LLaMA-3.1-8B", "Llama-3.1-8B", "LLaMA", "Llama", "FinBERT", "VADER",
    "GPT-4", "GPT-3.5", "GPT-4o", "Instruct",
]
DATASET_KEYWORDS = ["FPB", "FiQA", "FiQA-SA", "Financial PhraseBank", "financial_phrasebank", "FiNER-ORD", "GLiNER"]
TEMPLATE_KEYWORDS = ["zero-shot", "few-shot", "zero shot", "few shot", "template", "CoT", "chain-of-thought", "shot"]
METRIC_KEYWORDS = {
    "f1_macro": ["f1", "f1-macro", "f1 macro", "macro-f1", "macro f1"],
    "accuracy": ["accuracy", "acc."],
    "coverage": ["coverage", "parse rate", "parse failure", "parse-fail"],
}


def infer_context_keys(sentence: str) -> str:
    keys = []
    low = sentence.lower()
    for kw in MODEL_KEYWORDS:
        if kw.lower() in low and kw not in keys:
            keys.append(kw)
    for kw in DATASET_KEYWORDS:
        if kw.lower() in low and kw not in keys:
            keys.append(kw)
    for kw in TEMPLATE_KEYWORDS:
        if kw.lower() in low and kw not in keys:
            keys.append(kw)
    return "; ".join(keys)


def infer_metric_type(sentence: str, raw_value: str) -> str:
    low = sentence.lower()
    for mtype, kws in METRIC_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                return mtype
    if "%" in raw_value:
        return "percent"
    if re.search(r"[$₪€]", raw_value):
        return "money"
    if RE_POINT_GAP.search(raw_value):
        return "percent"  # point-gap is a percentage-point difference
    if RE_TIME.search(raw_value):
        return "time"
    if RE_PARAM_SIZE.fullmatch(raw_value.strip()):
        return "count"  # model parameter size, e.g. 110M, 8B
    if RE_DECIMAL.fullmatch(raw_value.strip()):
        # decimals in [0,1] range are almost always a score of some kind
        try:
            v = float(raw_value)
            if 0.0 <= v <= 1.0:
                return "f1_macro"  # best-effort guess; disambiguated by context_keys/sentence
        except ValueError:
            pass
        return "other"
    if raw_value.strip().isdigit():
        return "count"
    return "other"


def is_year_context(sentence: str, match_start: int, match_text: str) -> bool:
    """True if this numeric token is a bare year and nothing else (exclude)."""
    if RE_YEAR.fullmatch(match_text):
        return True
    return False


def is_citation_or_figure_or_section(sentence: str, match_start: int, match_text: str) -> bool:
    """Heuristic exclusion: numbers immediately preceded by section/figure/table/reference cues.

    Note: a leading '[' is only treated as a citation marker (e.g. "[12]") when the
    bracketed content is a BARE integer with no decimal point — CI ranges like
    "[0.776, 0.873]" must NOT be excluded just because they start with '['.
    """
    window_start = max(0, match_start - 25)
    prefix = sentence[window_start:match_start].lower()

    # bracket-specific check: only exclude if this looks like a bare citation number [12]
    if prefix.rstrip().endswith("["):
        if "." not in match_text:
            return True
        return False  # decimal inside brackets -> likely a CI bound, keep it

    exclude_cues = [
        "section ", "§", "figure ", "fig. ", "fig ", "table ", "chapter ",
        "appendix ", "eq. ", "equation ", "ref. ", "page ", "p. ",
    ]
    for cue in exclude_cues:
        if prefix.rstrip().endswith(cue.rstrip()) or cue in prefix[-12:]:
            return True
    return False


def extract_numeric_claims(sentence: str):
    """Return list of (value_str, metric_type) tuples found in a sentence, deduplicated by span."""
    spans_found = []  # (start, end, raw_text)

    for regex in (RE_MONEY, RE_POINT_GAP, RE_TIME, RE_PERCENT, RE_PARAM_SIZE, RE_DECIMAL, RE_COUNT):
        for m in regex.finditer(sentence):
            spans_found.append((m.start(), m.end(), m.group()))

    # Deduplicate/merge overlapping spans - prefer the longer/more specific match
    spans_found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept = []
    for s, e, txt in spans_found:
        overlap = False
        for ks, ke, _ in kept:
            if s < ke and e > ks:
                overlap = True
                break
        if not overlap:
            kept.append((s, e, txt))

    results = []
    for s, e, txt in kept:
        if is_year_context(sentence, s, txt):
            continue
        if is_citation_or_figure_or_section(sentence, s, txt):
            continue
        # bare small integers with no unit/context and no decimal/percent — still keep if
        # they look like meaningful sample counts (>= 2 digits); this is handled by RE_COUNT already.
        metric_type = infer_metric_type(sentence, txt)
        results.append((txt.strip(), metric_type))
    return results


# ---------------------------------------------------------------------------
# Docx structural walk (paragraphs + tables in document order)
# ---------------------------------------------------------------------------

def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("unsupported parent type")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def get_heading_level(paragraph: Paragraph):
    style_name = paragraph.style.name if paragraph.style else ""
    m = re.match(r"Heading (\d)", style_name or "")
    if m:
        return int(m.group(1))
    if style_name == "Title":
        return 0
    return None


def main():
    if not DOCX_PATH.exists():
        print(f"ERROR: {DOCX_PATH} not found. Copy report/3_implementation_report.docx there first.")
        sys.exit(1)

    document = docx.Document(str(DOCX_PATH))

    dump_lines = []
    claims = []
    claim_counter = 1

    heading_stack = {}  # level -> text
    current_heading = "(preamble)"

    def current_heading_path():
        levels = sorted(heading_stack.keys())
        return " > ".join(heading_stack[l] for l in levels) if levels else "(preamble)"

    table_index = 0

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            level = get_heading_level(block)
            if level is not None and text:
                # reset deeper levels
                heading_stack.update({k: v for k, v in heading_stack.items() if k < level})
                heading_stack[level] = text
                current_heading = current_heading_path()
                prefix = "#" * max(level, 1)
                dump_lines.append(f"\n{prefix} {text}\n")

                # Headings often start with a bare section number, e.g. "12. Conclusion"
                # or "8.1 Sentiment...". Strip a LEADING numbering token (that's the
                # excluded section number) but still scan the remainder of the heading
                # for genuine quantitative claims, e.g. "(30 hand-tagged FPB errors)".
                heading_body = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", text)
                if heading_body and heading_body != text:
                    found = extract_numeric_claims(heading_body)
                    for raw_value, metric_type in found:
                        claim_id = f"R{claim_counter:03d}"
                        claim_counter += 1
                        claims.append({
                            "claim_id": claim_id,
                            "section": current_heading,
                            "sentence": text,
                            "metric_type": metric_type,
                            "value": raw_value,
                            "context_keys": infer_context_keys(text),
                        })
                continue

            if not text:
                continue
            dump_lines.append(text)

            # split into sentences (simple heuristic split, keep the whole paragraph as fallback)
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
            if not sentences:
                sentences = [text]
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                found = extract_numeric_claims(sentence)
                for raw_value, metric_type in found:
                    claim_id = f"R{claim_counter:03d}"
                    claim_counter += 1
                    claims.append({
                        "claim_id": claim_id,
                        "section": current_heading,
                        "sentence": sentence,
                        "metric_type": metric_type,
                        "value": raw_value,
                        "context_keys": infer_context_keys(sentence),
                    })
        elif isinstance(block, Table):
            table_index += 1
            dump_lines.append(f"\n**[Table {table_index} — section: {current_heading}]**\n")

            # Build a simple grid, dedup merged cells by identity
            grid = []
            for row in block.rows:
                row_cells = []
                seen_tc = set()
                for cell in row.cells:
                    tc_id = id(cell._tc)
                    row_cells.append(cell.text.strip())
                grid.append(row_cells)

            if grid:
                header_row = grid[0]
                # markdown table dump
                dump_lines.append("| " + " | ".join(header_row) + " |")
                dump_lines.append("|" + "---|" * len(header_row))
                for row in grid[1:]:
                    dump_lines.append("| " + " | ".join(row) + " |")

                col_headers = header_row
                for r_idx, row in enumerate(grid[1:], start=1):
                    row_header = row[0] if row else ""
                    for c_idx, cell_text in enumerate(row):
                        if not cell_text:
                            continue
                        col_header = col_headers[c_idx] if c_idx < len(col_headers) else f"col{c_idx}"
                        cell_context = (
                            f"[Table {table_index}] row='{row_header}' col='{col_header}' "
                            f"cell='{cell_text}' (section: {current_heading})"
                        )
                        found = extract_numeric_claims(cell_text)
                        # also try scanning with context of row/col header combined, in case the
                        # cell itself is just a bare number (common in results tables)
                        if not found:
                            found = extract_numeric_claims(cell_text + " ")
                        for raw_value, metric_type in found:
                            claim_id = f"R{claim_counter:03d}"
                            claim_counter += 1
                            # refine metric type using row/col headers as extra context
                            combined_context = f"{row_header} {col_header} {cell_text}".lower()
                            refined_metric = infer_metric_type(combined_context, raw_value)
                            context_keys = infer_context_keys(f"{row_header} {col_header}")
                            claims.append({
                                "claim_id": claim_id,
                                "section": current_heading,
                                "sentence": cell_context,
                                "metric_type": refined_metric,
                                "value": raw_value,
                                "context_keys": context_keys,
                            })

    DUMP_PATH.write_text("\n".join(dump_lines), encoding="utf-8")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["claim_id", "section", "sentence", "metric_type", "value", "context_keys"]
        )
        writer.writeheader()
        for row in claims:
            writer.writerow(row)

    print(f"Wrote {DUMP_PATH} ({len(dump_lines)} lines)")
    print(f"Wrote {CSV_PATH} ({len(claims)} claims)")

    headings = [line.strip("# \n") for line in dump_lines if line.strip().startswith("#")]
    print("Headings found:")
    for h in headings:
        print(f"  - {h}")


if __name__ == "__main__":
    main()
