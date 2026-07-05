# Pre-Submission Assessment — "Does Financial Instruction Tuning Actually Help?"

**Reviewed:** 2026-07-05/06 · **Branch:** `review/pre-submission` (15 commits ahead of `master`)
**Spec:** `llm_finance_seminar_presentation_handout.pdf` (implementation presentation, 6 required sections) + `project_plan.md` (§11 deliverables). Method: the `assess-practicum-assignment` "verify, don't trust" procedure — every numeric claim was traced to a real artifact, headline metrics were independently recomputed from raw predictions, and the methodology code was adversarially audited.

---

## Overall grade: **A− (91/100)** as-submitted — **A (94)** after the fixes below (most already applied on this branch)

This is rigorous, unusually honest graduate work whose core methodology survives adversarial scrutiny intact. It earns marks exactly where the handout says it weights most: *"rigorous thinking and critical evaluation matter more than polish… include baselines and an honest discussion of failures."* The gap between A− and A is **not** substance — it is packaging and one disclosure: the analyses that back four report sections lived on an **unmerged branch**, and the report never flags that its strongest baseline (FinBERT) was trained on the FPB test set. Both are minutes-to-fix, and this review already fixed most of the packaging.

**Verification integrity:** of **420 quantitative claims** extracted from the report and decks, **367 matched** their artifacts exactly, **43 were non-checkable** (qualitative/citations), **10 were minor mismatches**, and **0 were unbacked or fabricated**. Every "too good to check" analysis (bootstrap CIs, leakage-free ensemble weighting, AllAgree re-evaluation) was opened and confirmed real. This is a high-trust submission.

---

## 1. Core tasks & requirements checklist (handout's 6 sections)

| Section | Verdict | Evidence |
|---|---|---|
| **Objective** — what was reproduced/adapted/simplified | ✅ Met | Report §3; every deviation (FinLLaMA→plutus-8B 404 substitution, inference-only, 5-shot descoped, Colab→Modal) is explicitly disclosed. |
| **System design** — data, models, prompts, stack | ✅ Met | §4 + pipeline diagram; unified `Sample` schema, runner split, 3 prompt templates. |
| **Experimental design** — metrics, baselines, split, leakage | ✅ Met | §5; bootstrap CIs, coverage as first-class metric, seed-42 determinism, few-shot from FPB-train only. |
| **Results** — quantitative + baselines + examples | ✅ Met | §6 (sentiment) + §7 (NER); interpreted, not dumped. Baselines: FinBERT, FinBERT-tone, VADER, GLiNER. |
| **Error analysis** — where it works/fails and why | ✅ Met | §8; 30 hand-tagged plutus errors (negation, numeric reasoning, domain jargon). |
| **Lessons learned** — challenges, reproducibility, next steps | ✅ Met | §11 + §10 "What we got wrong" (8 candid missteps). |

**Handout general expectations:** clear structure + figures ✅ · interpret don't copy ✅ · explicit assumptions/limitations/leakage ✅ · baselines + honest failure discussion ✅ (strongest area).

**project_plan.md §11 deliverables:** final_table.csv ✅ · 4–5 key figures ✅ (10+ under `report/figures/`) · 30-min deck ✅ · public repo + README ⚠️ (README was a `TODO` stub — **fixed this branch**) · rehearsal — unverifiable.

---

## 2. Guardrails & gotchas audit (adversarial — "assume leakage exists")

| Pitfall | Verdict | Note |
|---|---|---|
| Few-shot leakage | ✅ Avoided | Exemplars from FPB-train only; FiQA reuses FPB-train (cross-dataset, so exemplar∩test = ∅). |
| Split determinism | ✅ Avoided | `random.Random(42).shuffle` on file-ordered list; reproducible, no set/dict-ordering risk. |
| Ensemble hygiene | ✅ Avoided | `cv_weighted` is genuine positional k-fold — a member's weight never sees the example it judges; `oracle_weighted` explicitly fenced as a non-deployable ceiling. |
| AllAgree hygiene | ✅ Avoided | Subset chosen by sentence text (prediction-independent), framed as label-noise analysis with VADER as flat control. |
| Bootstrap correctness | ✅ Avoided | Paired per-example resample, B=2000, percentile CIs — and the CIs **actively gate** comparative claims (§6.4), not computed-then-ignored. |
| Parse discipline | ✅ Avoided | No force-to-neutral anywhere in parser→runner→evaluation→ensemble; coverage travels with accuracy in every table. |
| Pretraining leakage (LLMs) | ✅ Disclosed | FPB 2014 / FiQA 2018 predate the models; flagged as common-mode inflation. |
| **FinBERT trained on FPB** | ⚠️ **PRESENT (major)** | ProsusAI/finbert was fine-tuned *on* Financial PhraseBank, so its FPB 0.925 / AllAgree 0.977 are effectively in-sample. Report discloses generic contamination but never this specific train-on-test. **It strengthens the thesis** (a general LLM nearly matching a model trained on the test set makes plutus look worse), so it is a disclosure gap, not an invalidation — but an ML grader will flag it. |
| FiQA ±0.10 neutral band | ⚠️ Minor | Disclosed and applied fairly, but no rationale and no sensitivity sweep; FiQA is exactly where rankings flip. |
| Uncertainty framing | ⚠️ Minor | "up to 16 points" / "6/6 cells" are directional-consistency counts that read stronger than n=300 CIs (±0.045) strictly support. |

---

## 3. Claim-verification summary

| | Count |
|---|---|
| Total claims checked (report + decks) | 420 |
| ✅ Matched artifact | 367 |
| ○ Non-checkable (qualitative/citations) | 43 |
| ✗ Mismatched | 10 |
| ⚠ Unbacked / fabricated | **0** |

The 10 mismatches (all independently re-confirmed by an Opus adversarial recheck):

- **Report (3, need manual docx edits — prose is out of auto-fix scope):** total cost stated **"~$1.3"** but the ledger sums to **~$1.15** (rounds to $1.2); **"~130 GPU-minutes"** vs actual ~120; FiNER-ORD scale table says **"200–300"** but all rows are n=300 (stale lower bound).
- **Decks (7 — all in the *generated* `implementation_deck.pptx` snapshot; the actually-presented `rev5`/`rev6` decks were already clean):** "12 LLM runs + 4 baselines" (it's **24 + 6**), "≥0.93 on every class" (two classes are 0.91–0.92), "100% parsing coverage" (plutus dips to 0.97–0.98), "localhost:8502" (default was 8501). **All fixed at source in `build_deck.js` + `run.bat` this branch.**

---

## 4. Extra credit / beyond-baseline (all verified real, not just claimed)

- **Prompt-ensemble study** — 4-member majority + CV-weighted voting; leakage-free; beats mean single-prompt on 6/6 cells (`ensemble_table.csv`).
- **Gold-label quality re-evaluation** — full pipeline re-run on FPB AllAgree (100%-agreement) gold; plutus gains least (+0.022) → the financial-tuning conclusion *sharpens* under cleaner labels (`fpb_agreement_comparison.csv`, `bootstrap_ci.csv`).
- **NER track** — GLiNER vs 5 LLMs on FiNER-ORD + FIN/Alvarado (98 ex.), strict/partial/type F1, cost/latency ledger.
- **Two interactive dashboards**, budget-guarded Modal harness, checkpoint/resume run schema.

This is well above a minimum reproduction.

---

## 5. Critical grading risk assessment — ranked (top deduction risks)

1. **[highest] Unmerged branch → four report sections not reproducible from `master`.** The submission's ensemble/AllAgree/bootstrap/NER analyses + the report bundle lived only on `prompt-ensemble-improvement`; `master` was 5 weeks stale. A grader cloning the default branch could not reproduce §6.3/§6.4/§7 or the CIs. **Fix (1 min):** fast-forward `master` to this branch (this review already merged it here). *This is the single most important item.*
2. **[major] FinBERT-on-FPB contamination undisclosed.** **Fix:** one sentence in §5/§9 noting FinBERT's FPB & AllAgree scores are in-sample and that this *strengthens* (not weakens) the "financial tuning doesn't help" conclusion.
3. **[minor] Self-reported numbers slightly off.** Cost "~$1.3"→~$1.2, "~130"→~120 GPU-min, "200–300"→300. **Fix:** three number edits in the docx (the report otherwise verifies at 367/420 exact).
4. **[minor, fixed] Fresh-clone friction.** `pip install` resolved into a torch-2.11/scipy-1.17 combo that failed 11 tests; README quickstart was a `TODO` stub. **Fixed this branch** (scipy pin + real README); the report already flags the test issue in §10 (though it misattributes the cause to "test ordering" rather than the dependency combo — worth a one-word correction).
5. **[minor] FiQA neutral-band robustness unexamined.** Add a one-line ±0.05/±0.15 sensitivity note or acknowledge it as future work.

---

## 6. Improvement suggestions

### 6a. Prompt design — new templates tested on Modal (this review, ~$0.12)
The report's own data shows Template B collapses plutus on FiQA (F1 0.60→0.43) because its *"factual ⇒ neutral"* definition is a **neutral magnet**. We designed and ran three replacements (0-shot, plutus + Mistral, same 300-id subsets):

| Cell | Best original (A/B) | New template | Gain |
|---|---|---|---|
| plutus · FPB | 0.66 | **H = 0.75** | **+0.09** |
| plutus · FiQA | 0.60 | **D = 0.67** | **+0.07** |
| Mistral · FPB | 0.80 | **H = 0.84** | **+0.04** |
| Mistral · FiQA | 0.60 | **F = 0.67** | **+0.08** |

Mechanism confirmed at the class level: plutus FPB-B drives positive-recall to **0.157**; Template **H** (analyst rubric *minus* the neutral magnet + completion cue) restores it to **0.743**. **Every model×dataset improves with a better prompt, coverage stays ~1.0.** This *strengthens* the thesis: plutus's mediocrity is partly a prompt-sensitivity artifact — but even prompt-optimized it does not clearly beat prompt-optimized general models. **Suggested for the report:** add H/F/D as a "prompt-robustness" paragraph in §6.1; note that A-vs-B is a confounded 4-way change and the clean axis is the *neutral definition*, not the output format. (Templates D/F/H + tests are committed; `prompt_design_critique.md` has the full analysis and a 2×2 deconfounding design.)

### 6b. Other
- **Qwen3-8B sentiment** (lecturer's request): added to the runner + Modal image bumped to transformers 4.55, but the run stalled at model-load and did not complete within budget — documented as pending. The NER track already includes qwen3_8b; sentiment can be finished with one command (`run_llm_matrix.py --only qwen3_8b`).
- Report the **parity-corrected baseline table** (`review/evidence/parity_table.csv`) so all models sit on the same 300 ids — conclusions are unchanged, which is worth stating.

---

## 7. Applied-fixes log (this branch)

| Commit | Fix |
|---|---|
| `89657a9` | Merge `prompt-ensemble-improvement` — makes §6.3/§6.4/bootstrap/NER reproducible |
| `ffada9f` | Pin `scipy<1.17` — restores 11 failing tests → **146 pass** |
| `bd35171` | Sync `experiment.yaml` to the actually-evaluated model/dataset roster |
| `a2a823e` | `aggregate.py` dataset allowlist — re-run now reproduces committed summaries byte-identically |
| `843ee20` | Real README quickstart + correct model roster + parity note |
| `cbeece0` | Notebook 04 runs from any cwd (root-anchored paths) |
| `036e756` | Fill deck placeholders + correct 4 verified deck overclaims + pin dashboard port |
| `15ffbc5`, `f25f15e` | Prompt-experiment templates D/F/H + runner flags |
| `67a4265`, `3de14fc` | Parity table + all review evidence |

**Not applied (yours to do):** the 3 report-prose number edits (§5.1–3 above), the FinBERT-contamination sentence, and the `master` fast-forward. Report `.docx` prose was deliberately left untouched.

---

## 8. Security note

**Rotate your HF token.** STATUS.md item 7 records that a token was pasted in chat and briefly committed before redaction. A history scan (`git log --all -G "hf_[A-Za-z0-9]{20,}"`) found **no token blob in any reachable commit** — the amend was clean — but since the value was exposed in chat, revoke it at https://huggingface.co/settings/tokens, mint a new one, and refresh the Modal secret. Do this **before** making the repo public.

---

## Bottom line

Grade it **A− as-is, A once the branch is merged and two sentences are added.** The work is rigorous where it counts, exceptionally honest about its own failures, and — verified claim-by-claim — carries **zero fabrication** across 420 numbers. Its weaknesses are packaging and disclosure, not method. Ship it.
