"""Fill the `category` column of focal_error_sample.csv with a hand-assigned
error taxonomy (keyed by example id). Reproducible: re-run to regenerate.

Taxonomy (what the plutus-8B miss actually is):
  missed_positive_cue        - mild / forward-looking positive read as neutral (neutral-bias)
  numerical_reasoning        - needs comparison of figures (profit up/down, loss narrowing, %)
  factual_neutral_misclassed - a neutral fact (appointment, M&A, delisting, restructuring) read as pos/neg
  ambiguous                  - out-of-domain / genuinely unclear sentence
"""
import csv, pathlib

CAT = {
    "FPB_01120": "factual_neutral_misclassed",
    "FPB_01151": "missed_positive_cue",
    "FPB_00380": "missed_positive_cue",
    "FPB_01606": "factual_neutral_misclassed",
    "FPB_03174": "factual_neutral_misclassed",
    "FPB_01337": "missed_positive_cue",
    "FPB_01190": "missed_positive_cue",
    "FPB_00072": "missed_positive_cue",
    "FPB_01349": "missed_positive_cue",
    "FPB_01472": "numerical_reasoning",
    "FPB_00357": "missed_positive_cue",
    "FPB_00710": "numerical_reasoning",
    "FPB_03176": "numerical_reasoning",
    "FPB_01969": "factual_neutral_misclassed",
    "FPB_01417": "missed_positive_cue",
    "FPB_00794": "ambiguous",
    "FPB_00659": "numerical_reasoning",
    "FPB_00499": "missed_positive_cue",
    "FPB_02875": "numerical_reasoning",
    "FPB_00529": "numerical_reasoning",
    "FPB_03003": "factual_neutral_misclassed",
    "FPB_00835": "missed_positive_cue",
    "FPB_00422": "missed_positive_cue",
    "FPB_00531": "numerical_reasoning",
    "FPB_01169": "missed_positive_cue",
    "FPB_00704": "missed_positive_cue",
    "FPB_02759": "factual_neutral_misclassed",
    "FPB_00462": "numerical_reasoning",
    "FPB_00123": "missed_positive_cue",
}

p = pathlib.Path("results/summary/focal_error_sample.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
for r in rows:
    r["category"] = CAT.get(r["id"], "unassigned")
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

from collections import Counter
c = Counter(r["category"] for r in rows)
print(f"Tagged {len(rows)} rows:")
for k, v in c.most_common():
    print(f"  {k:30s} {v}")
