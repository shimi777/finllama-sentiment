"""Proper evaluation on the Financial PhraseBank 100%-agreement (AllAgree) split.

Runs the SAME setup as the 75%-agree matrix, but on AllAgree's own train/test
split (seed 42, 20% hold-out):
  - baselines (VADER, FinBERT, FinBERT-tone) on the full AllAgree test set (local CPU)
  - 3 LLMs (Qwen2.5-7B, Mistral-7B, plutus-8B) x templates A/B x {0,3}-shot on Modal T4

Writes to a SEPARATE namespace (dataset label "FPBall") so the 75%-agree runs are
untouched, then aggregates to results/summary/allagree_table.csv.

Usage:
  .venv/Scripts/python.exe scripts/run_allagree.py             # full (baselines + LLMs)
  .venv/Scripts/python.exe scripts/run_allagree.py --baselines-only
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_fpb            # noqa: E402
from src.prompts import build_prompt, sample_fewshot  # noqa: E402
from src.parser import parse                    # noqa: E402
from src.evaluation import compute_metrics      # noqa: E402
from src.utils import set_seed, get_logger      # noqa: E402

logger = get_logger("run_allagree")

SEED = 42
DS = "FPBall"                       # distinct namespace from FPB (75-agree)
SUBSAMPLE = 300                     # LLM subsample (baselines use full test)
CAP_USD = 7.0
GPU = "T4"
LLMS = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct",           {"chat": False}),
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", {"chat": False}),
    ("plutus8b",  "TheFinAI/plutus-8B-instruct",         {"chat": True}),
]
TEMPLATES = ["A", "B"]
SHOTS = [0, 3]


def subsample(samples, n, seed):
    if len(samples) <= n:
        return list(samples)
    return random.Random(seed).sample(samples, n)


def write_run(run_dir, run_id, hf_id, tpl, shots, samples, preds, runtime_s, started_at):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    done = datetime.now(timezone.utc).isoformat()
    meta = {"run_id": run_id, "model_hf_id": hf_id, "dataset": DS, "template": tpl,
            "shots": shots, "seed": SEED, "n_total": len(samples),
            "started_at": started_at, "completed_at": done, "runtime_s": round(runtime_s, 2)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump({"last_completed_idx": len(samples), "n_total": len(samples),
                   "updated_at": done}, f, indent=2)


def already_done(run_dir, n_total):
    pf = os.path.join(run_dir, "progress.json")
    if not os.path.exists(pf):
        return False
    with open(pf) as f:
        return json.load(f).get("last_completed_idx", 0) >= n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-only", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)

    train, test = load_fpb(config="sentences_allagree", test_fraction=0.20, seed=SEED)
    logger.info("AllAgree: %d train, %d test", len(train), len(test))

    pred_dir = os.path.join(ROOT, "results", "predictions")
    summ_dir = os.path.join(ROOT, "results", "summary")
    os.makedirs(pred_dir, exist_ok=True)
    rows = []

    def record_metrics(model, tpl, shots, samples, preds, runtime_s):
        m = compute_metrics(samples, preds)
        rows.append({"model": model, "dataset": DS, "template": tpl, "shots": shots,
                     "seed": SEED, "n_samples": m["n_samples"], "accuracy": round(m["accuracy"], 4),
                     "f1_macro": round(m["f1_macro"], 4), "f1_weighted": round(m["f1_weighted"], 4),
                     "coverage": round(m["coverage"], 4), "runtime_s": round(runtime_s, 2)})
        logger.info("%-13s %s/%sshot  acc=%.3f f1m=%.3f cov=%.2f",
                    model, tpl, shots, m["accuracy"], m["f1_macro"], m["coverage"])

    # ---------- baselines (local CPU) ----------
    from src.models.vader_runner import VADERRunner
    from src.models.finbert_runner import FinBERTRunner
    import torch
    dev = 0 if torch.cuda.is_available() else -1
    texts = [s["text"] for s in test]

    def run_baseline(model_key, hf_id, label, predictor):
        rid = f"{model_key}__{DS}__seed{SEED}"
        rdir = os.path.join(pred_dir, rid)
        t0 = time.perf_counter(); started = datetime.now(timezone.utc).isoformat()
        labels = predictor(texts)
        rt = time.perf_counter() - t0
        preds = [{"id": s["id"], "pred_label": l, "raw_output": "", "parse_ok": True, "latency_ms": 0.0}
                 for s, l in zip(test, labels)]
        write_run(rdir, rid, hf_id, "-", 0, test, preds, rt, started)
        record_metrics(label, "-", 0, test, preds, rt)

    def safe_baseline(model_key, hf_id, label, make_predictor):
        try:
            run_baseline(model_key, hf_id, label, make_predictor())
        except Exception as e:
            logger.warning("[skip] baseline %s failed to load: %s", label, e)

    safe_baseline("vader", "vaderSentiment", "VADER", lambda: VADERRunner().predict)
    safe_baseline("finbert", "ProsusAI/finbert", "FinBERT",
                  lambda: FinBERTRunner(hf_id="ProsusAI/finbert", device=dev).predict)
    safe_baseline("finbert_tone", "yiyanghkust/finbert-tone", "FinBERT-tone",
                  lambda: FinBERTRunner(hf_id="yiyanghkust/finbert-tone", device=dev).predict)

    # ---------- LLMs (Modal T4) ----------
    if not args.baselines_only:
        from scripts.modal_app import app, LLMRunner
        from scripts.modal_budget import can_afford, record, remaining_usd, summary as bsum
        eval_set = subsample(test, SUBSAMPLE, SEED)
        logger.info("LLM eval subsample: %d (budget %s, remaining $%.4f)",
                    len(eval_set), bsum(), remaining_usd(CAP_USD))
        with app.run():
            runner = LLMRunner()
            for model_key, hf_id, flags in LLMS:
                for tpl in TEMPLATES:
                    for shots in SHOTS:
                        rid = f"{model_key}__{DS}__{tpl}__{shots}shot__seed{SEED}"
                        rdir = os.path.join(pred_dir, rid)
                        if already_done(rdir, len(eval_set)):
                            logger.info("[skip] %s", rid);
                            preds = [json.loads(l) for l in open(os.path.join(rdir, "predictions.jsonl"), encoding="utf-8")]
                            record_metrics(model_key, tpl, shots, eval_set, preds, 0.0); continue
                        if not can_afford(280, GPU, CAP_USD):
                            logger.warning("[abort] budget cap reached"); break
                        few = sample_fewshot(train, shots, seed=SEED) if shots > 0 else []
                        prompts = [build_prompt(tpl, s["text"], few) for s in eval_set]
                        started = datetime.now(timezone.utc).isoformat(); t0 = time.perf_counter()
                        raw = runner.generate.remote(hf_id=hf_id, prompts=prompts,
                                                     max_new_tokens=20, batch_size=8,
                                                     use_chat_template=flags.get("chat", False))
                        rt = time.perf_counter() - t0
                        record(rid, GPU, rt)
                        preds = [{"id": s["id"], "pred_label": parse(r), "raw_output": r,
                                  "parse_ok": parse(r) is not None,
                                  "latency_ms": rt * 1000.0 / max(len(raw), 1)}
                                 for s, r in zip(eval_set, raw)]
                        write_run(rdir, rid, hf_id, tpl, shots, eval_set, preds, rt, started)
                        record_metrics(model_key, tpl, shots, eval_set, preds, rt)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(summ_dir, "allagree_table.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
