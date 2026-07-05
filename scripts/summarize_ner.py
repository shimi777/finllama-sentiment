"""Aggregate all NER runs into a single comparison table and compare to paper.

Reads `results/predictions/ner__*/meta.json`, gathers `metrics.micro_f1`, and
emits:
  - `results/summary/ner_table.csv`  (one row per run)
  - stdout: a markdown comparison vs. Open-FinLLMs Table 7 NER column.

Paper numbers (Table 7, NER column, 0-1 range):
    FinLLaMA-instruct: 0.57
    GPT-4:             0.80
    ChatGPT:           0.53
    FinTral:           0.40
    FinMA-7B-full:     0.35
    Mistral-7B-Instr:  0.00
    Palmyra-Fin-70B:   0.08

Of these, the open-weight models reachable on T4 are Mistral-7B-Instruct-v0.3
(close enough to v0.1 used by the paper), plutus-8B-instruct (TheFinAI's current
8B financial instruct model, substitute for the unpublished FinLLaMA-instruct),
and Qwen2.5-7B-Instruct as a strong general baseline.
"""

from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "results", "predictions")
SUMMARY_DIR = os.path.join(ROOT, "results", "summary")

PAPER_BASELINES = {
    "Mistral-7B-Instruct-v0.1": 0.00,
    "FinMA-7B-full":            0.35,
    "FinTral":                  0.40,
    "FinLLaMA-instruct":        0.57,
    "ChatGPT (gpt-3.5)":        0.53,
    "GPT-4":                    0.80,
    "Palmyra-Fin-70B":          0.08,
}

# Map of (model_short -> paper-reported counterpart) for direct comparison.
PAPER_ANALOG = {
    "mistral7b": "Mistral-7B-Instruct-v0.1",
    "plutus8b":  "FinLLaMA-instruct",
    "qwen25_7b": None,  # no direct paper analog; report as "general 7B"
    "qwen3_8b":  None,
    "finma_7b":  "FinMA-7B-full",
}


def collect_runs() -> list[dict]:
    rows: list[dict] = []
    if not os.path.isdir(PRED_DIR):
        return rows
    for name in sorted(os.listdir(PRED_DIR)):
        if not name.startswith("ner__"):
            continue
        meta_path = os.path.join(PRED_DIR, name, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "metrics" not in meta:
            continue
        m = meta["metrics"]
        parts = name.split("__")
        # ner__{model}__{ds}__{tpl}__{shots}shot__seed{seed}
        model_short = parts[1]
        ds = parts[2]
        tpl = parts[3]
        shots = parts[4].replace("shot", "")
        rows.append({
            "run_id":     meta["run_id"],
            "model":      model_short,
            "model_hf":   meta.get("model_hf_id"),
            "dataset":    ds,
            "template":   tpl,
            "shots":      shots,
            "n_total":    meta.get("n_total"),
            "n_eval":     m.get("n_evaluated"),
            "coverage":   m.get("coverage"),
            "micro_f1":   m.get("micro_f1"),
            "micro_p":    m.get("micro_precision"),
            "micro_r":    m.get("micro_recall"),
            "macro_f1":   m.get("macro_f1"),
            "parse_fail": m.get("n_parse_failures"),
            "per_type":   m.get("per_type"),
            "runtime_s":  meta.get("runtime_s"),
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["run_id", "model", "model_hf", "dataset", "template", "shots",
            "n_total", "n_eval", "coverage", "micro_f1", "micro_p", "micro_r",
            "macro_f1", "parse_fail", "runtime_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def best_per_model(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        m = r["model"]
        cur = best.get(m)
        if cur is None or (r["micro_f1"] or 0) > (cur["micro_f1"] or 0):
            best[m] = r
    return best


def print_markdown(rows: list[dict]) -> None:
    if not rows:
        print("No NER runs found.")
        return

    print("# NER reproduction — Open-FinLLMs Table 7")
    print()
    print("## All runs")
    print()
    print("| run_id | template | shots | n_eval | cov | micro F1 | P | R | parse fail |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['run_id']}` | {r['template']} | {r['shots']} | "
              f"{r['n_eval']}/{r['n_total']} | {r['coverage']:.2f} | "
              f"**{r['micro_f1']:.3f}** | {r['micro_p']:.3f} | {r['micro_r']:.3f} | "
              f"{r['parse_fail']} |")

    print()
    print("## Best run per model vs. paper")
    print()
    print("| Model | Best micro-F1 (ours) | Paper analog | Paper micro-F1 | Delta |")
    print("|---|---|---|---|---|")
    for short, r in best_per_model(rows).items():
        analog = PAPER_ANALOG.get(short)
        paper = PAPER_BASELINES.get(analog) if analog else None
        delta = (r["micro_f1"] - paper) if paper is not None else None
        delta_s = f"{delta:+.3f}" if delta is not None else "—"
        print(f"| {short} | {r['micro_f1']:.3f} | {analog or '—'} | "
              f"{paper if paper is not None else '—'} | {delta_s} |")

    print()
    print("## Paper Table 7 (NER column, for reference)")
    print()
    print("| Model | NER |")
    print("|---|---|")
    for k, v in PAPER_BASELINES.items():
        print(f"| {k} | {v:.2f} |")


def main():
    rows = collect_runs()
    out_csv = os.path.join(SUMMARY_DIR, "ner_table.csv")
    write_csv(rows, out_csv)
    print_markdown(rows)
    if rows:
        print(f"\nWrote {len(rows)} rows -> {os.path.relpath(out_csv, ROOT)}")
    else:
        print("No runs yet; CSV not written.", file=sys.stderr)


if __name__ == "__main__":
    main()
