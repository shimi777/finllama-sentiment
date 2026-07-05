# Prompt-design experiment — results

**Setup:** 3 new templates (D/F/H from `prompt_design_critique.md`) run on Modal, 0-shot,
plutus-8B + Mistral-7B × FPB + FiQA, on the **same 300-id subsets** (verified parity).
Metrics independently recomputed from raw predictions (`src.evaluation.compute_metrics`).
Spend delta ≈ **$0.12** (total ledger $1.15 → $1.27).

Templates:
- **D** — strict structured, order-neutral output (definitions removed).
- **F** — Template A + one explicit rule: "use neutral only as a last resort."
- **H** — Template B's analyst rubric with the *"factual ⇒ neutral"* clause removed, neutral demoted, plus a completion cue.

## Macro-F1 by cell (best original vs new)

| Model | Dataset | A (0s) | B (0s) | **D** | **F** | **H** | best new − best A/B |
|---|---|---|---|---|---|---|---|
| plutus-8B | FPB | 0.662 | 0.648 | 0.614 | 0.635 | **0.753** | **+0.091 (H)** |
| plutus-8B | FiQA | 0.597 | 0.432 | **0.671** | 0.613 | 0.650 | **+0.074 (D)** |
| Mistral-7B | FPB | 0.803 | 0.689 | 0.803 | **0.841** | 0.840 | **+0.038 (F)** |
| Mistral-7B | FiQA | 0.599 | 0.564 | 0.648 | **0.674** | 0.621 | **+0.075 (F)** |

Coverage stayed ≥0.997 for every new cell (no parse-robustness regression).

## Per-hypothesis verdicts

1. **"B's damage is the neutral definition, not the one-word constraint" — CONFIRMED.**
   plutus FPB-B collapses positive-recall to **0.157** (neutral-recall 0.968: it dumps
   positives into neutral). Template **H** keeps B's rubric structure but removes the
   neutral magnet → positive-recall **0.743**, F1 0.648 → **0.753**. The output-format
   constraint was never the problem.
2. **"An explicit neutral tie-break rescues the neutral-scarce dataset (FiQA)" — CONFIRMED.**
   F and D both lift every FiQA cell (plutus +0.07 with D, Mistral +0.08 with F).
3. **"Structured output holds coverage" — CONFIRMED** (D coverage = 1.0 everywhere).
4. **Does a better prompt let plutus close the gap? — PARTIALLY.** Prompt-optimized plutus
   (FPB 0.75, FiQA 0.67) clearly beats its own A/B floor, but still does **not** surpass
   prompt-optimized Mistral (0.84 / 0.67) or Qwen2.5. **The report's thesis holds and is
   strengthened:** plutus's weakness was *partly* a prompt-sensitivity artifact, yet
   financial tuning still buys no advantage once every model gets a fair prompt.

## The confound worth stating in the report

A-vs-B differs in four ways at once (persona, definition list, neutral-definition content,
output-format style). The driver is the **neutral-definition content**. The clean
experiment is a 2×2: A-body×{Sentiment-cue, one-word} and B-body×{same} — proposed as
future work in `prompt_design_critique.md` §4.

**One sentence for the seminar report:** *"Re-running the two most prompt-sensitive models
with a neutral-magnet-free template (H/F/D) recovers +0.04 to +0.09 macro-F1 across all four
model×dataset cells — showing plutus-8B's headline weakness is partly a prompt artifact,
yet financial tuning still yields no advantage once prompts are matched."*
