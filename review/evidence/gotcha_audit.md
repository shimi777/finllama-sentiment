# Adversarial Methodology Audit — gotcha_audit.md

Repo: `finllama-sentiment` · branch `review/pre-submission`
Analysis code for ensemble/AllAgree/bootstrap read from branch `prompt-ensemble-improvement` via `git show`.
Report text: `review/evidence/report_dump.md`. Working assumption: flaws exist; each was attacked before recording "avoided".

Legend — verdict ∈ {PRESENT, AVOIDED, PARTIAL, NOT-CHECKABLE}; severity ∈ {blocker, major, minor, info}.

---

## 1. Few-shot leakage (exemplar ∩ test = ∅? FiQA pool?)

**PROOF-ATTEMPT.** Tried to show few-shot exemplars are drawn from a pool that overlaps the evaluated test items, or from an undisclosed pool for FiQA.

**EVIDENCE.**
- `scripts/run_llm_matrix.py:132-136` — the pool is built once: `fpb_train, fpb_test = load_fpb(seed=SEED)`; eval sets are `subsample(fpb_test,...)` and `subsample(fiqa_test,...)`. Train/test are the disjoint halves of the seed-42 shuffle (`src/data_loader.py:94-104`, `test = all[:n_test]`, `train = all[n_test:]` — no overlap by construction).
- `scripts/run_llm_matrix.py:177` — `few_shot = sample_fewshot(fpb_train, shots, ...)`. This line sits inside the `for ds in DATASETS` plan loop, so **the same `fpb_train` pool feeds BOTH FPB and FiQA runs.** `src/prompts.py:31` samples only from `pool` (= `fpb_train`); it never touches the eval set.
- FiQA-specific: exemplars are FPB-train sentences, and the FiQA eval items are FiQA rows — different datasets, so exemplar ∩ FiQA-test = ∅ trivially. For FPB runs, exemplars come from `fpb_train` and eval from `fpb_test` (disjoint). `sample_fewshot` never receives `fpb_test`/`fiqa_test`.
- Disclosure of the cross-dataset pool: report **line 69** ("balanced few-shot sampling from the FPB train pool"), **line 111** ("the FPB-sampled exemplars are off-distribution for FiQA's microblog text and drag performance down"). The cross-dataset choice is stated *and* used as an explanation of the FiQA few-shot regression. `run_allagree.py` (branch) uses the same `sample_fewshot(fpb_train,...)` pattern on its own AllAgree train split.

**VERDICT: AVOIDED — info.** No exemplar↔test leakage on either dataset. Using an FPB pool for FiQA is a register mismatch (a modeling weakness), but it is disclosed and cannot leak labels. Cheap fix (optional): state explicitly in §5 that FiQA has no train split so the FPB pool is reused, rather than leaving it implicit at line 111.

---

## 2. FPB split determinism (same 690 test ids across runs/machines?)

**PROOF-ATTEMPT.** Tried to find set/dict iteration-order nondeterminism that would make the seed-42 test split machine-dependent.

**EVIDENCE.**
- `src/data_loader.py:78-99` — `all_samples` is built by `enumerate(lines)` over the zip's `.txt` in **file order** (a `list`, deterministic), then `random.Random(seed).shuffle(all_samples)`. No `set`/`dict` ordering enters the split. Ids are `FPB_{i:05d}` where `i` is the file line index, so they are content-anchored, not run-anchored.
- Verified empirically with the venv python: `random.Random(42).shuffle` on a fixed-order list is reproducible and `test = lst[:n_test]` is identical across two independent runs (`shuffle deterministic: True`, `test ids identical: True`).
- One residual risk: the split depends on the exact line order and count of the `*75Agree*.txt` inside `takala/financial_phrasebank`. If HF ever re-orders or re-encodes that file, ids shift. That is an upstream-artifact risk, not a code bug. `n_test = int(690? ...)`: 75Agree has 3453 sentences → `int(3453*0.20)=690` test, matching the report's "690 test".

**VERDICT: AVOIDED — info.** Split is deterministic on any machine given the same source file. Cheap fix (optional): cache a `subset_ids_FPB.json` checksum (already present in `review/evidence/`) and assert against it at load time to pin the upstream file.

---

## 3. FiQA neutral_band = ±0.10 (tuned on test? justified? disclosed? sensitivity?)

**PROOF-ATTEMPT.** Tried to show the ±0.10 band was chosen to flatter results, is undisclosed, or has an un-examined effect on the label distribution.

**EVIDENCE.**
- `src/data_loader.py:110-133` — `neutral_band=0.10` default; `configs/experiment.yaml:27` `neutral_band: 0.10`. Applied uniformly to build gold labels for every model, so it cannot advantage one model over another.
- Disclosed: report **line 34** and **line 270** ("bucketed to 3 classes with a ±0.10 neutral band"). So it is stated.
- BUT: **no justification** for 0.10 vs any other value, and **no sensitivity analysis** anywhere in the report (grep for "sensitivity"/"band" returns only the two disclosure lines and the prompt-sensitivity section, which is unrelated). The band directly sets the neutral-class prevalence, which drives macro-F1 on a 3-class task — FiQA is where model rankings flip (Qwen wins), so the un-examined band is load-bearing for a headline claim.
- No evidence the band was *tuned on test* (it is a fixed default in config and code, not swept). So the worst reading — "tuned to inflate" — is not provable; the real defect is the missing robustness check.

**VERDICT: PARTIAL — minor.** Disclosed and applied fairly, but unjustified and its sensitivity is un-examined; a reviewer cannot tell if FiQA rankings survive a ±0.05 or ±0.15 band. Cheap fix: add one sentence + a 3-row table re-scoring FiQA at band ∈ {0.05, 0.10, 0.15} to show rankings are stable (the data + `load_fiqa(neutral_band=...)` make this ~10 lines, no GPU).

---

## 4. Ensemble evaluation hygiene — cv_weighted in-sample weighting; oracle labeling; §6.3 honesty

**PROOF-ATTEMPT.** Tried to show `cv_weighted` fits member weights on the same 300 items it scores (in-sample inflation), that `oracle_weighted` is passed off as deployable, or that §6.3 hides an abstention/coverage cost.

**EVIDENCE.**
- `src/ensemble.py` `aggregate()` — respects the parse-failure invariant: abstaining members (`parse_ok=False`) don't vote; a tie under `tie_break="abstain"` returns `parse_ok=False` (counts against coverage, never forced to a label). Clean.
- `scripts/aggregate_ensemble.py::cv_weight_map` — **genuinely leakage-free per-example weighting.** For fold `f`, `train_ids = [ids not in fold f]`, weights = member accuracy on `train_ids`, then assigned to the held-out fold's ids. So the weight applied to example `i` is never estimated using `i`. This is the correct construction; the report's "leakage-free for weighting" (line 115) is accurate.
- `oracle_weighted`: weights = member accuracy on the FULL eval set (`oracle_w_vec = [member_accuracy(ids, ...)]`). The docstring says it "USES TEST LABELS and is reported only as an upper-bound ceiling, never as a deployable result", and the report Table 4 / line 125 frame the oracle *best-single* as "the honest ceiling ... you would only know in hindsight". Correctly labeled.
- §6.3 honesty: Table 4 reports **both** F1 **and** coverage per aggregator (unweighted-abstain cov 0.887; cv-weighted cov 0.996). The tradeoff is stated at line 125: unweighted majority abstains on ties (lower coverage), cv-weighted keeps ~100% coverage at almost the same F1. The claim "beats mean-single 6/6, best-single 0/6 (unweighted) / 1/6 (cv)" is directly the honest ceiling framing — it does NOT claim to beat the oracle.

**VERDICT: AVOIDED — info.** The one place this class of project usually cheats (fitting ensemble weights in-sample) is done correctly with positional k-fold CV, and the oracle is explicitly fenced off as a non-deployable ceiling. Coverage is reported alongside F1.

---

## 5. AllAgree analysis hygiene (subset selection independent of predictions? uncertainty at n≈195?)

**PROOF-ATTEMPT.** Tried to show the AllAgree subset is selected using model predictions (cherry-picking easy sentences to inflate F1), or that it is framed as "model improvement" without the label-noise caveat, or that the tiny-n has no uncertainty attached.

**EVIDENCE.**
- `scripts/eval_allagree.py` (branch) — the subset is selected by **sentence text membership**: `allagree_texts = {norm(s["text"]) for s in tr_all+te_all}`, then `if norm(g["text"]) in allagree_texts`. Selection depends only on gold text + annotator-agreement level, **never on predictions**. Independent of any model output. Correct.
- Framing: report §6.4 (lines 143-145) frames the lift as **label-noise removal**, not model improvement — "most of their 'errors' on the 75% set were on the sentences humans themselves disagreed about" — and uses VADER as a **control** (line 144: +0.014 flat, "confirming the gains are real model behaviour on cleaner labels, not an artefact of an easier subset"). This is exactly the correct framing the audit brief demanded (label-noise analysis, not "our model got better").
- Two independent AllAgree tracks: a **proper re-run** on AllAgree's own split (`run_allagree.py`, namespace `FPBall`, line 129) *and* a no-GPU cross-check filtering the 75%-agree predictions (`eval_allagree.py`, line 146) — reported to agree in ranking/direction.
- Uncertainty at small n: the AllAgree LLM subsample is 300 (not 195; the "195" in the brief is likely the unanimous-subset count from the `eval_allagree` cross-check). Either way, §6.4 **attaches bootstrap CIs** to the AllAgree claim (line 145: plutus [0.806,0.894] vs Mistral [0.908,0.964] non-overlapping) — the one place §6.4 asserts a *real* gap, it is CI-backed.

**VERDICT: AVOIDED — info.** Subset selection is prediction-independent; framed as label-noise analysis with a lexicon control; the sharpened-gap claim carries non-overlapping CIs.

---

## 6. Bootstrap CI correctness + are CIs actually USED?

**PROOF-ATTEMPT.** Tried to show the bootstrap resamples the wrong unit, uses a wrong interval method, or that CIs are computed once and then ignored when the prose claims model A > model B.

**EVIDENCE.**
- `scripts/bootstrap_ci.py::boot_ci` — resampling unit is the **per-example (paired) prediction**: `idx = RNG.integers(0,n,n)` then `f1_score(y_true[idx], y_pred[idx])`. Correct unit (examples, with replacement, n-out-of-n), correct paired resample (same `idx` for true+pred). `B=2000`. Percentile method: `np.percentile(stats,[2.5,97.5])` → proper 95% percentile CI. Seeded `default_rng(42)`.
- Subtlety (not a flaw): only parsed predictions enter (`if not parse_ok: continue`), consistent with the project's coverage-exclusion invariant. The CI therefore describes F1 *on answered items*, matching how the point estimate is computed. Consistent, disclosed by the coverage columns.
- Are CIs used? **Yes, in comparative claims:** line 89 ("plutus≈Qwen [0.776,0.873] vs [0.784,0.878] statistically tied; FinBERT's FPB lead over Mistral within noise"), line 145 (AllAgree non-overlapping CIs), line 204 ("two-model differences under ~0.06-0.09 F1 are not significant"). The report repeatedly refuses to call sub-CI gaps winners. Table 2 bolds "the highest point estimate, not a proven winner" (line 89).

**VERDICT: AVOIDED — info.** Bootstrap is textbook-correct and the CIs actively gate the comparative claims rather than being decorative.

---

## 7. FinBERT trained-on-FPB contamination (train-on-test for the baseline)

**PROOF-ATTEMPT.** Tried to show the report compares FinBERT to LLMs on FPB test without disclosing that ProsusAI/finbert was fine-tuned ON Financial PhraseBank — i.e. FinBERT's FPB test items are (near-)training data for it.

**EVIDENCE.**
- Fact: ProsusAI/finbert is FinBERT (Araci 2019) fine-tuned on the Financial PhraseBank sentiment task. Its published ~0.9+ FPB accuracy is in-domain/in-distribution *by construction*. On FPB test — especially the AllAgree gold, which FinBERT's training used — this is effectively train-on-test (the seed-42 20% hold-out is *our* split, not FinBERT's; FinBERT likely saw those exact sentences during its own training, as FPB was not split the same way).
- Report search (`grep -i finbert|train|contaminat|PhraseBank`): the report calls FinBERT "in-domain" / "the cheap specialist ... best tool only on the distribution it was trained for" (lines 82, 92) and lists "Possible benchmark contamination. FPB/FiQA/FIN are old and public; pre-training exposure would inflate every model" (line 202). **But it never states that FinBERT was fine-tuned on FPB itself**, i.e. that FinBERT's FPB number is a supervised-on-this-dataset result, not a zero-shot one. The contamination caveat (line 202) is about *pre-training* exposure of the LLMs, not FinBERT's *direct supervised training* on FPB.
- Reasoning on direction (the brief asks): this does **not weaken** the report's central thesis — it *strengthens* it. The thesis is "financial instruction tuning (plutus) doesn't help." FinBERT's FPB 0.925 is the near-ceiling a model *actually trained on FPB* reaches; the fact that a general LLM (Mistral 0.890) comes within CI of a model trained on the test distribution, while the financial-tuned plutus (0.829) does not, makes plutus look worse, not better. So the contamination is a real methodological gap but it cuts *toward* the paper's conclusion.
- What it *does* invalidate: the framing "FinBERT wins FPB" as evidence of anything about model quality. FinBERT winning the dataset it was trained on is expected and uninformative; the report treats it as a "small in-domain models don't generalise" lesson (FiQA collapse) — which is fine — but readers may mis-read the FPB 0.925 as a fair zero-shot comparison. The AllAgree FinBERT 0.977 (line 136) is the most contaminated number of all (AllAgree ⊂ FinBERT's training pool) and is presented without that flag.

**VERDICT: PARTIAL — major.** The specific, most-important contamination fact (FinBERT was *supervised-trained on FPB*, so its FPB/AllAgree scores are in-sample) is not disclosed; only the weaker "old/public benchmark, pre-training exposure" caveat is. Because it strengthens the thesis, it is not a blocker — but an ML grader will flag an undisclosed train-on-test baseline. Cheap fix: one sentence in §5/§6 — "FinBERT (ProsusAI/finbert) was fine-tuned on Financial PhraseBank, so its FPB and AllAgree scores are effectively in-sample and are an upper reference bound, not a fair held-out comparison; this makes plutus's failure to match it on FPB, and the fair LLMs' near-match, the informative comparisons."

---

## 8. Pretraining-data leakage for the LLMs (FPB 2014 / FiQA 2018 predate the models)

**PROOF-ATTEMPT.** Tried to show the report ignores that these public benchmarks predate every evaluated LLM and could be in their pretraining corpora.

**EVIDENCE.**
- Report **line 72** (§5): "(i) These benchmarks are old and public; any of these models may have seen FPB/FiQA/FIN during pre-training, which would inflate all of them — we cannot rule it out." Repeated at **line 202** (§9 limitations). Explicitly acknowledged, and correctly noted as inflating *all* models roughly equally (so it doesn't bias the relative "financial tuning doesn't help" conclusion, which is the load-bearing claim).

**VERDICT: AVOIDED — info.** Benchmark-contamination risk for the LLMs is disclosed twice and correctly scoped as a common-mode inflation.

---

## 9. Honest uncertainty on 2-3 point F1 gaps at n=300 (worst overclaim?)

**PROOF-ATTEMPT.** Tried to find a headline claim that reads a sub-CI F1 gap as a real difference.

**EVIDENCE.**
- The report is unusually disciplined: line 89 explicitly ties plutus≈Qwen and FinBERT≈Mistral; line 204 states gaps under ~0.06-0.09 F1 are not significant at n=300; line 199 warns statistical power is thin. Table 2 bold = point estimate, not winner.
- Closest thing to an overclaim: **line 20 / executive summary** — "Swapping one prompt template for another moves macro-F1 by up to 16 points ... larger than most model-to-model gaps." The 16.4 pt figure (Table 3, plutus·FiQA A vs B) is a single most-extreme cell reported as if representative ("up to" hedges it, but the exec-summary phrasing generalises). Still, it is labeled "up to" and backed by Table 3. Also **line 21**: "Majority-vote ... beats the mean single prompt on 6/6 model×dataset cells" — true per Table 4, but "6/6" at n=300 per cell with +0.02–0.08 gains is within the ±0.045 CI for several cells; the 6/6 is a directional-consistency count, not 6 individually-significant wins. The report does not claim per-cell significance, so it stays just inside honest.
- The AllAgree "gap gets sharper / statistically real" claim (line 145) IS backed by non-overlapping CIs — not an overclaim.

**VERDICT: AVOIDED (borderline PARTIAL) — minor.** No sub-CI gap is asserted as a real winner; hedging is present throughout. The mild soft spot is the "6/6 cells" and "up to 16 points" framing, which are directionally true but could read as stronger than n=300 supports. Cheap fix (optional): note that "6/6" is directional consistency, and that per-cell ensemble gains are mostly within the single-run CI.

---

## 10. Parse-failure discipline (no force-to-neutral; coverage reported with accuracy)

**PROOF-ATTEMPT.** Tried to find any code path that defaults an unparseable output to a label, or a metric reported without its coverage.

**EVIDENCE.**
- `src/parser.py:29-43` — `parse()` returns `None` when no token matches; it never defaults to "neutral".
- `scripts/run_llm_matrix.py:202-208` — `lbl = parse(raw); parse_ok = lbl is not None`. Unparseable → `pred_label=None, parse_ok=False`. No fallback label. Same pattern in `run_allagree.py` (branch).
- `src/evaluation.py:29-37` — parse failures are **excluded** from true/pred lists (only counted when `parse_ok and pred_label is not None`) and surface as `coverage = n_parse_ok/len(samples)`. Accuracy/F1 are computed on answered items only.
- `src/ensemble.py` (branch) preserves the invariant: abstaining members don't vote; abstain-tie → `parse_ok=False`.
- Report: coverage is a first-class column in Tables 2? (implicit), Table 4, Tables 6-7, and §5 line 68 states the rule explicitly ("Forcing unparseable outputs to 'neutral' would silently inflate a model that hedges — so we refuse to"). NER tables (6/7) show plutus's coverage drops (0.853, 1.00) next to F1.

**VERDICT: AVOIDED — info.** The no-force invariant holds end-to-end (parser → runner → evaluation → ensemble), and coverage travels with accuracy in every results table.

---

# Ranked list of surviving gotchas (worst first)

1. **[major] #7 FinBERT trained-on-FPB contamination is not disclosed.** The report discloses generic "old public benchmark" contamination but never states that ProsusAI/finbert was *supervised-fine-tuned on Financial PhraseBank*, making its FPB (0.925) and AllAgree (0.977) numbers effectively in-sample. It strengthens the thesis but is an undisclosed train-on-test baseline an ML grader will flag. Fix: one sentence flagging FinBERT's FPB scores as an in-sample reference bound.
2. **[minor] #3 FiQA ±0.10 neutral band is unjustified and its sensitivity un-examined.** Disclosed and applied fairly, but the band sets neutral prevalence and FiQA is where rankings flip; no robustness sweep. Fix: a 3-row band ∈ {0.05,0.10,0.15} re-score (no GPU).
3. **[minor] #9 Mild framing softness** ("up to 16 points", "6/6 cells") — directionally true, could read stronger than n=300 supports. Fix: note these are directional-consistency counts, not per-cell significant wins.
4. **[info] #1 FiQA few-shot pool is the FPB train set** — disclosed and non-leaking, but the "FiQA has no train split, so we reuse FPB" rationale is only implicit. Optional one-line clarification.

Everything else (#2 split determinism, #4 ensemble hygiene, #5 AllAgree hygiene, #6 bootstrap correctness/usage, #8 LLM pretraining-leakage disclosure, #10 parse discipline) — **AVOIDED**. The evaluation code is methodologically clean where it most often isn't: leakage-free CV weighting, prediction-independent AllAgree subsetting, correct paired bootstrap that actually gates the prose, and a strictly enforced no-force-to-neutral coverage invariant.

**No blockers.** One major disclosure gap (#7), two minor (#3, #9), one info (#1).
