"""
Self-verification pass for claims.csv.

Independently unzips review/evidence/report_copy.docx, pulls all text runs out
of word/document.xml with a SEPARATE minimal regex, extracts every numeric
token, and diffs that set against the numbers captured in claims.csv.

Reports:
  - count of distinct numeric tokens in the XML
  - count of distinct numeric tokens captured in claims.csv
  - plausible metric-like misses (decimals in [0,1], 2-4 digit counts) not in claims.csv

Run:
  C:/python_projects/finllama-sentiment/.venv/Scripts/python.exe review/scripts/verify_docx_numbers.py
"""
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCX_PATH = REPO_ROOT / "review" / "evidence" / "report_copy.docx"
CSV_PATH = REPO_ROOT / "review" / "evidence" / "claims.csv"

# Minimal, independent text extraction: grab contents of <w:t>...</w:t> runs only.
RE_WT = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
RE_TAG = re.compile(r"<[^>]+>")

# Minimal numeric token regex: any run of digits with optional single internal
# '.' or ',' (thousands) and optional trailing %. A negative lookbehind excludes
# digits glued onto a preceding word/version-string char (e.g. the "0.1" inside
# "Mistral-v0.1" is a version label, not a metric) -- this mirrors the guard the
# real extractor's RE_DECIMAL already uses.
RE_NUM_TOKEN = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*%?")

RE_YEAR = re.compile(r"^(19|20)\d{2}$")


def extract_xml_text():
    with zipfile.ZipFile(DOCX_PATH) as z:
        xml_bytes = z.read("word/document.xml")
    xml = xml_bytes.decode("utf-8")
    # Replace paragraph/table-row boundaries with a space to avoid token gluing,
    # then pull all <w:t> run contents.
    runs = RE_WT.findall(xml)
    texts = []
    for r in runs:
        # unescape a few common xml entities
        r = r.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
        texts.append(r)
    return texts


def main():
    texts = extract_xml_text()
    full_text = " ".join(texts)

    xml_tokens = RE_NUM_TOKEN.findall(full_text)
    # normalize: strip trailing '.' artifacts (e.g. end of sentence glued to number won't happen since regex requires digit start)
    xml_tokens_clean = []
    for t in xml_tokens:
        t = t.strip(",.")
        if t:
            xml_tokens_clean.append(t)
    xml_token_counts = Counter(xml_tokens_clean)
    distinct_xml_tokens = set(xml_token_counts.keys())

    # Load claims.csv values
    csv_values = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_values.append(row["value"].strip())
    distinct_csv_values = set()
    for v in csv_values:
        # normalize percent/point-gap/money/time strings down to their bare numeric core for comparison
        m = re.search(r"\d[\d,]*\.?\d*", v)
        if m:
            distinct_csv_values.add(m.group().strip(",."))
        distinct_csv_values.add(v)

    print(f"Distinct numeric tokens in XML (minimal regex): {len(distinct_xml_tokens)}")
    print(f"Distinct claim values in claims.csv: {len(set(csv_values))}")
    print(f"Total claim rows in claims.csv: {len(csv_values)}")

    # Identify plausible metric-like misses
    misses = []
    for tok in sorted(distinct_xml_tokens, key=lambda x: (len(x), x)):
        bare = tok.strip(",.%")
        if not bare:
            continue

        # skip pure years
        if RE_YEAR.match(bare):
            continue

        # is this token (or its bare numeric core) present anywhere in csv values?
        present = tok in distinct_csv_values or bare in distinct_csv_values
        if present:
            continue

        is_decimal_01 = False
        try:
            if "." in bare:
                fval = float(bare.replace(",", ""))
                if 0.0 <= fval <= 1.0:
                    is_decimal_01 = True
        except ValueError:
            pass

        is_count_2_4_digit = bool(re.match(r"^\d{2,4}$", bare))

        if is_decimal_01 or is_count_2_4_digit:
            misses.append((tok, xml_token_counts[tok]))

    print(f"\nPlausible metric-like MISSES ({len(misses)} distinct tokens):")
    for tok, cnt in misses:
        print(f"  {tok!r}  (occurs {cnt}x in xml)")

    return misses


if __name__ == "__main__":
    main()
