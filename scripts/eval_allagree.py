"""Measure every FPB run on the 100%-agreement (AllAgree) subset of the gold set.

The Financial PhraseBank agreement subsets are nested (AllAgree subset 75Agree),
with identical labels. Our runs scored the 75%-agree test split; here we restrict,
by sentence text, to the sentences that ALSO have 100% (unanimous) annotator
agreement, and recompute metrics. No GPU / no re-inference needed.

Writes results/summary/fpb_agreement_comparison.csv and prints a table.
"""
import csv, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_loader import load_fpb
from src.evaluation import compute_metrics

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"

def norm(t): return " ".join(t.split()).strip().lower()

# 75Agree test split (seed 42) — the gold our runs were scored against
_, test75 = load_fpb(config="sentences_75agree", test_fraction=0.20, seed=42)
gold_by_id = {s["id"]: s for s in test75}

# AllAgree = unanimous; build a set of its texts (both splits, full coverage)
tr_all, te_all = load_fpb(config="sentences_allagree", test_fraction=0.20, seed=42)
allagree_texts = {norm(s["text"]) for s in (tr_all + te_all)}
print(f"75Agree test: {len(test75)} | AllAgree total sentences: {len(allagree_texts)}\n")

rows = []
for d in sorted(PRED.glob("*__FPB__*")):
    if "_ner__" in d.name or d.name.startswith("finbert_ner"):
        continue
    pf = d / "predictions.jsonl"
    if not pf.exists():
        continue
    preds = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
    # align gold to prediction order; skip preds whose id isn't in this gold split
    samples, preds_aligned = [], []
    for p in preds:
        g = gold_by_id.get(p["id"])
        if g is None:
            continue
        samples.append(g); preds_aligned.append(p)
    if not samples:
        continue
    m75 = compute_metrics(samples, preds_aligned)
    # AllAgree subset (by text)
    s_all, p_all = [], []
    for g, p in zip(samples, preds_aligned):
        if norm(g["text"]) in allagree_texts:
            s_all.append(g); p_all.append(p)
    mall = compute_metrics(s_all, p_all) if s_all else None
    rows.append({
        "run_id": d.name,
        "n_75agree": m75["n_samples"], "acc_75": round(m75["accuracy"],4), "f1_75": round(m75["f1_macro"],4), "cov_75": round(m75["coverage"],4),
        "n_allagree": (mall["n_samples"] if mall else 0),
        "acc_allagree": (round(mall["accuracy"],4) if mall else ""), "f1_allagree": (round(mall["f1_macro"],4) if mall else ""), "cov_allagree": (round(mall["coverage"],4) if mall else ""),
    })

out = ROOT / "results" / "summary" / "fpb_agreement_comparison.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

hdr = f"{'run_id':42s} {'n75':>4} {'acc75':>6} {'f1_75':>6} | {'nAll':>4} {'accAll':>6} {'f1All':>6}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['run_id']:42s} {r['n_75agree']:>4} {r['acc_75']:>6} {r['f1_75']:>6} | "
          f"{r['n_allagree']:>4} {str(r['acc_allagree']):>6} {str(r['f1_allagree']):>6}")
print(f"\nWrote {out}")
