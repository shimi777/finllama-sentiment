"""Compare FPB macro-F1 at 75% vs 100% (AllAgree) annotator agreement.

Reads the two PROPER evaluation tables:
  results/summary/final_table.csv    (75%-agree; FPB rows)
  results/summary/allagree_table.csv (100%-agree; from scripts/run_allagree.py)

Takes the best config per model on each, writes
results/summary/agreement_comparison.csv, and renders a paired-bar figure.
"""
import csv, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "key_figures" / "agreement_comparison.png"
CSV_OUT = ROOT / "results" / "summary" / "agreement_comparison.csv"

CANON = {"finbert": "FinBERT", "FinBERT": "FinBERT",
         "mistral7b": "Mistral-7B", "plutus8b": "plutus-8B",
         "qwen25_7b": "Qwen2.5-7B", "vader": "VADER", "VADER": "VADER"}
ORDER = ["FinBERT", "Mistral-7B", "Qwen2.5-7B", "plutus-8B", "VADER"]


def best_by_model(path, dataset_filter=None):
    best = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if dataset_filter and r["dataset"] != dataset_filter:
            continue
        name = CANON.get(r["model"])
        if name is None:
            continue
        f1 = float(r["f1_macro"])
        cfg = f"{r.get('template', '-')}/{r.get('shots', '0')}shot"
        if name not in best or f1 > best[name][0]:
            best[name] = (f1, cfg)
    return best


b75 = best_by_model(ROOT / "results/summary/final_table.csv", dataset_filter="FPB")
ball = best_by_model(ROOT / "results/summary/allagree_table.csv")

rows = []
for m in ORDER:
    if m in b75 and m in ball:
        rows.append({"model": m, "f1_75agree": round(b75[m][0], 4), "cfg_75": b75[m][1],
                     "f1_allagree": round(ball[m][0], 4), "cfg_all": ball[m][1],
                     "delta": round(ball[m][0] - b75[m][0], 4)})
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

labels = [r["model"] for r in rows]
f75 = [r["f1_75agree"] for r in rows]
f100 = [r["f1_allagree"] for r in rows]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.6))
b1 = ax.bar(x - w/2, f75, w, label="75% agreement (reported headline)", color="#5b8db8")
b2 = ax.bar(x + w/2, f100, w, label="100% agreement (unanimous gold)", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("macro-F1 (best config)"); ax.set_ylim(0, 1.05)
ax.set_title("FPB: every learned model improves on unanimous gold — but the financial-tuned\n"
             "plutus-8B gains the least and stays last; VADER (no learning) is flat",
             fontsize=10, weight="bold")
ax.legend(fontsize=8, loc="lower left")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=150)
print("wrote", OUT)
print("wrote", CSV_OUT)
for r in rows:
    print(f"  {r['model']:12s} 75%={r['f1_75agree']:.3f} ({r['cfg_75']:8s})  "
          f"100%={r['f1_allagree']:.3f} ({r['cfg_all']:8s})  d={r['delta']:+.3f}")
