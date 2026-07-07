# Prompt-design critique & new-candidate proposal

Scope: prompt-engineering review of the sentiment-classification templates used in the
LLM matrix. Focal model **plutus-8B-instruct** (financially tuned), plus
**Mistral-7B-Instruct-v0.3** and **Qwen2.5-7B-Instruct**, all 4-bit, chat-template
applied, greedy, `max_new_tokens=20`. Output is parsed by `src/parser.py`
(first-match regex over canonical labels + synonyms; parse failure ⇒ sample
**excluded**, not forced to a label).

Evidence base:
- Templates A, B: `src/prompts.py` (working tree).
- Template C: `git show prompt-ensemble-improvement:src/prompts.py` (used only inside the
  prompt-ensemble analysis, never as a headline cell).
- Metrics: `results/summary/final_table.csv`,
  `git show prompt-ensemble-improvement:results/summary/ensemble_table.csv`.
- Confusion matrices: `results/summary/confusions/*.json`.
- Focal errors: `results/summary/focal_error_sample.csv`.

---

## 0. The single most important empirical fact

Across every cell, **the failure mode is the neutral/non-neutral boundary, and every
template moves models along a neutral-bias axis.** The confusion matrices make this
concrete:

| cell | neutral precision | neutral recall | what it means |
|---|---|---|---|
| `plutus8b FiQA B 0shot` | **0.160** | 0.917 | catastrophic over-prediction of neutral: 129/172 true-positives dumped into neutral |
| `qwen25_7b FiQA B 0shot` (best FiQA) | 0.284 | 0.694 | still neutral-biased but positive recall survives (0.63) |
| `mistral7b FPB A 0shot` | 0.821 | 0.968 | healthy; negative recall 0.89 |
| `mistral7b FPB B 0shot` | 0.749 | 0.951 | B collapses **negative** recall to 0.47 (losses/risks read as "factual") |
| `plutus8b FPB A 0shot` | 0.846 | 0.741 | opposite failure: negative *over*-predicted (neg precision 0.46) |

Read this table before the template-by-template critique: it is the lens. A template is
"good" here largely to the extent that it keeps the model off the neutral hedge **without**
tipping it into over-calling negative. The two datasets pull in opposite directions —
FPB is neutral-heavy (185/300 neutral in the subset) so a neutral-leaning prompt looks
fine on accuracy; FiQA is positive-heavy (172/300) and neutral-scarce (36/300) so the same
neutral lean is catastrophic. **Any prompt tuned on FPB accuracy will silently harm FiQA.**
This is the central confound of the whole prompt study.

---

## 1. Critique of Templates A, B, C

### Template A — minimalist completion cue
```
Classify the sentiment of the following financial text as positive, negative, or neutral.
{fewshot_block}Text: {text}
Sentiment:
```

**Good.**
- The bare `Sentiment:` completion cue is the strongest single design choice in the whole
  set. It puts the model in *continuation* mode rather than *chat-response* mode, so the
  first generated token is overwhelmingly the label itself — which is exactly what the
  first-match parser wants. Coverage is 1.0 in nearly every A cell.
- Label list appears once, inline, in the natural reading order "positive, negative,
  neutral". Short prompt ⇒ less room for the model to editorialize inside a 20-token budget.
- Empirically it is the best **general** template: `mistral7b FPB A 3shot` = 0.890 F1 (top
  LLM cell on FPB), `plutus8b FiQA A 0shot` = 0.597 F1 (plutus's best FiQA cell),
  `qwen25_7b FiQA A 0shot` respectable.

**Risky.**
- **Label-order bias.** The prompt states "positive, negative, or neutral" (positive first),
  but the few-shot sampler and the parser both use `_LABEL_ORDER = [negative, neutral,
  positive]`. There is a latent primacy effect (positive named first in A) that is never
  controlled or counterbalanced. We cannot currently separate "A is better" from "A happens
  to prime positive, which helps positive-heavy FiQA".
- **`Sentiment:` under a chat template.** When `use_chat_template=True` (scripts/modal_app.py
  wraps every prompt as a single user turn), the trailing `Sentiment:` is now *inside the user
  message*, followed by the assistant generation prompt. For plutus especially this is the
  reason chat templating was needed at all (the code comment says plutus "otherwise emits EOS
  immediately"). But it means A's completion-cue advantage is partly **neutralized** by the
  chat wrapper — the model is answering a question, not completing a line. A's edge is
  therefore probably *smaller* than it would be in raw-completion mode, and this is untested.
- On FPB-0shot for plutus, A over-calls **negative** (neg precision 0.46). Minimalism gives
  plutus no tie-breaking guidance, so its financial tuning fires "risk detector" too eagerly.

### Template B — investor-framed definition list + hard one-word constraint
```
You are a financial analyst. Classify the sentiment of the text below from the perspective of an investor.
- Positive: the text suggests favorable conditions, growth, or gains
- Negative: the text suggests unfavorable conditions, losses, or risks
- Neutral: the text is factual without clear positive or negative implication

{fewshot_block}Text: {text}
Answer with one word only (positive / negative / neutral):
```

**Good.**
- The definition list is genuinely useful for *one* model on *one* dataset: `qwen25_7b FiQA
  B 0shot` is the **single best FiQA cell in the whole matrix (0.673 F1)**. Qwen is
  instruction-following and disciplined; the explicit rubric plus "one word only" gives it a
  decision procedure it executes cleanly.
- "Answer with one word only" is an honest attempt to protect the parser from prose.
- Investor-perspective framing is arguably the *correct* task definition for FiQA (author
  sentiment about a security), better matched to the label than A's generic "sentiment".

**Risky — and this is where the damage is.**
- **The neutral definition is a magnet.** "Neutral: the text is factual without clear
  positive or negative implication" describes *the majority of FPB sentences and a large
  share of FiQA headlines*, which are literally factual statements that nonetheless carry
  directional sentiment ("net sales rose 2%"). Handing the model an explicit "if it's just
  factual → neutral" rule invites it to route everything ambiguous into neutral. Confusion
  matrices confirm it precisely:
  - `plutus8b FiQA B 0shot`: neutral **precision 0.16** — 129 of 172 true positives dumped
    into neutral. B destroys plutus on FiQA (0.597 A → **0.432 B**, −0.165 F1). This is the
    largest single template-induced regression in the study.
  - `mistral7b FPB B 0shot`: negative recall **0.47** (vs A's 0.89). Mistral now reads
    "narrowed its net loss", "operating loss" as *factual → neutral*. B collapses Mistral on
    FPB (0.803 A → **0.689 B**, −0.114 F1).
- **Verbosity vs a 20-token budget.** B is ~5× the token count of A. It does not overrun the
  *answer* budget (the answer is still one word) but it lengthens the *prompt*, and more
  importantly it spends the model's limited "attention budget" on a rubric that, for
  neutral-ambiguous financial text, is actively misleading.
- **"One word only" vs A's `Sentiment:` cue is a different mechanism, not a stronger one.**
  A biases the *first token* toward a label by continuation. B *instructs* the model to emit
  one word. Instruction-following models (Qwen) honor it; less disciplined or more
  "opinionated" models may still preface with a hedge, and under a 20-token cap a hedge can
  eat the answer. Note plutus coverage dips exactly in B cells (`plutus8b FiQA A 3shot`
  cov 0.97, `plutus8b FPB A 3shot` cov 0.9833 — the sub-1.0 coverage cells cluster on plutus,
  the least instruction-disciplined of the three). B's constraint is only as good as the
  model's compliance, whereas A's cue works mechanically regardless of compliance.

**Why does B hurt Mistral-FPB but help Qwen-FiQA? (hypotheses)**
1. **Base-rate interaction (primary hypothesis).** B's neutral definition pushes probability
   mass toward neutral. FPB is neutral-heavy already, so on *accuracy* the push looks mild,
   but macro-F1 is dominated by the minority classes (negative/positive), and B specifically
   craters *negative* recall on Mistral → macro-F1 falls. FiQA is neutral-*scarce* and
   positive-heavy; a neutral push is pure poison for a model that would otherwise call
   positives (plutus, −0.165), but Qwen is disciplined enough to keep calling positives
   (positive recall 0.63 even under B) so it reaps only the *precision* benefit of the rubric
   without over-hedging. Net: same prompt, opposite sign, entirely explained by
   model-compliance × class base-rate.
2. **Instruction-following capacity.** Qwen executes the rubric as a checklist; Mistral and
   plutus treat the rubric as *suggestive framing* and let "factual ⇒ neutral" dominate. The
   rubric is only safe for models that also weigh the positive/negative branches equally,
   which requires strong instruction-following.
3. **Financial tuning backfires on plutus under B.** plutus's tuning likely makes it *more*
   sensitive to the word "factual" in a financial rubric (financial disclosures are drilled
   as factual), amplifying the neutral magnet. This predicts B hurts plutus *more* than
   Mistral on the positive→neutral leak, which matches (plutus −0.165 vs Mistral's FiQA
   also negative but its worst hit is on FPB-negative).

### Template C — market-reaction / bullish-bearish framing (ensemble-only)
```
Read the financial statement and judge how it would move an investor's outlook.
If it points to a better outlook it is bullish; to a worse outlook, bearish; if there is no clear directional signal, neutral.
{fewshot_block}Text: {text}
Reply with exactly one word — positive, negative, or neutral:
```

**Good.**
- **Best conceptual framing for the actual task.** "How would this move an investor's
  outlook / directional signal" is a *sharper* operationalization than A's vague "sentiment"
  or B's "favorable conditions". Directionality is exactly what FiQA scores encode. As an
  *ensemble member* it earns its keep: it is a genuinely independent third angle (the
  ensemble table shows `cv_weighted/abstain` 4-member ensembles beating single-best on F1 in
  several cells, e.g. `qwen25_7b FiQA` ens_f1 0.627 vs single_best_f1 0.673 — close, and
  ens *acc* 0.653 beats single-mean 0.601), and diversity of members is what makes an
  ensemble work. C contributes diversity precisely because its reasoning path differs.
- The "no clear directional signal → neutral" phrasing is a *better* neutral definition than
  B's "factual", because "factual" describes surface form while "no directional signal"
  describes the actual decision criterion. This should leak *less* into neutral than B.

**Risky.**
- **Vocabulary mismatch — the headline defect.** The reasoning vocabulary is
  *bullish/bearish*, but the required output vocabulary is *positive/negative/neutral*. The
  prompt asks the model to think in one lexicon and answer in another, then relies on the
  final instruction ("Reply with exactly one word — positive, negative, or neutral") to force
  the translation. Two failure risks:
  1. The model emits "bullish"/"bearish" as its one word. The parser *does* map these
     (`SYNONYMS` in `src/parser.py`: bullish→positive, bearish→negative), so this is
     *survivable* — but it is a **latent dependency on the synonym table** that A and B do
     not have. If anyone tightened the parser to canonical-only, C would silently lose
     coverage. This coupling is undocumented and fragile.
  2. Under a 20-token greedy budget, a model primed on "bullish/bearish" may generate a short
     directional *explanation* ("This is bullish because…") before the label. First-match
     regex would grab "bullish" and still parse — but only by luck of the synonym map.
- **Never validated as a headline cell.** C exists only inside the ensemble; there is no
  standalone C row in `final_table.csv`, so its solo accuracy is unknown. We cannot say
  whether C's better neutral definition actually helps in isolation or only as ensemble
  ballast.
- Same **label-order** exposure as A/B (positive named first in the output instruction).

---

## 2. Five new candidate templates (D–H)

All use the repo `TEMPLATES` string format with `{fewshot_block}` and `{text}` placeholders,
drop-in for `src/prompts.py`. Rationale, falsifiable hypothesis (tied to the data above),
expected effect, and risk given for each.

### D — Strict structured single-token output (parse-robustness + label-order neutrality)
```python
"D": (
    "Classify the sentiment of the financial text toward the company or asset it describes.\n"
    "Choose exactly one label from this set: [negative, neutral, positive].\n"
    "Respond with only the label as a single lowercase word. No explanation, no punctuation.\n"
    "{fewshot_block}"
    "Text: {text}\n"
    "Label:"
),
```
- **Rationale.** Combine A's mechanical completion cue (`Label:` instead of `Sentiment:`)
  with B's explicitness, but strip B's misleading definition list. Present the label set as an
  *unordered-looking bracketed set* in the parser's own order `[negative, neutral, positive]`
  to neutralize the positive-primacy of A/B. "single lowercase word / no punctuation" targets
  coverage.
- **Hypothesis (falsifiable).** *The definition list, not the "one word" constraint, is what
  hurts B.* If D (constraint kept, definitions removed) recovers plutus-FiQA back toward A's
  0.597 F1 and does **not** collapse Mistral-FPB negative recall, then the definition list is
  the culprit and "one word only" is harmless. If D still tanks, the constraint/style itself
  is at fault.
- **Expected effect.** Coverage ≥ A. F1 between A and B on FPB; on FiQA closer to A (recovers
  most of the −0.165 plutus loss). Neutral precision on `plutus FiQA` recovers from 0.16
  toward ~0.5.
- **Risk.** The bracketed set may *itself* read as neutral-priming (neutral in the middle).
  Very low risk of over-terse refusal from plutus (mitigated by the `Label:` cue).

### E — Financial-analyst reason-then-answer (mini chain-of-thought)
```python
"E": (
    "You are a financial analyst. Decide the sentiment of the text toward the company or asset.\n"
    "First give a 3-6 word reason, then on a new line give the final label.\n"
    "Format exactly:\nReason: <a few words>\nLabel: <positive|negative|neutral>\n"
    "{fewshot_block}"
    "Text: {text}\n"
    "Reason:"
),
```
- **Rationale.** The dominant error is silently defaulting to neutral. Forcing a *brief*
  justification before the label makes the model commit to a directional reading first, which
  should reduce reflexive hedging. Kept deliberately tiny ("3–6 words") to respect the token
  budget.
- **Token-budget caveat (must-read).** `max_new_tokens=20` in `scripts/modal_app.py` is **too
  small** for reason-then-answer. A 3–6-word reason + `\nLabel: positive` is ~12–18 tokens
  *plus* the "Reason:" the model re-emits ⇒ realistic need is **40 tokens**. Required config
  change: raise `max_new_tokens` to **40** for the E cells only. Cost impact: generation cost
  scales ~linearly with output tokens, so E cells cost ~2× the others. At the quoted
  ~$0.10–0.15 per template (2 models × 2 datasets × 300 × 0-shot), an E run is ~$0.20–0.30,
  which alone is ~half the $0.50 budget. **This is why E is not a first-pick to run** (see §3),
  but it is the highest-*information* candidate if budget allowed.
- **Hypothesis (falsifiable).** *The neutral collapse is a low-effort default, not a genuine
  belief.* If forcing a reason lifts positive recall on `plutus FiQA` and negative recall on
  `mistral FPB` materially (say +0.15 recall on the collapsed class) while coverage stays ≥
  0.95, then the model *can* read directionality and was merely hedging. If recall barely
  moves, the model genuinely cannot tell — a much more damning finding for the whole task.
- **Expected effect.** Higher macro-F1 on the *minority* classes at the cost of a few points
  of neutral precision; higher latency/cost; a small coverage risk if the model wanders.
- **Risk.** Parser sees "Reason: … Label: positive" — first-match regex could grab a sentiment
  word *inside the reason* ("Reason: strong growth ⇒ **positive**") before the real label.
  This is a real parse-ordering hazard. Mitigation for the run: parse only the substring after
  the last "Label:" — a 2-line, reversible change to `parser.py` gated to E, OR accept
  first-match and treat the reason as intentionally label-free (harder to guarantee).
  Flag: do not run E without deciding this.

### F — Definition-light imperative with an explicit neutral tie-breaking rule
```python
"F": (
    "Classify the sentiment of the financial text as positive, negative, or neutral.\n"
    "Rule: only choose neutral if the text has no positive or negative implication at all. "
    "If the text leans even slightly toward gains or losses, choose positive or negative accordingly.\n"
    "{fewshot_block}"
    "Text: {text}\n"
    "Sentiment:"
),
```
- **Rationale.** Directly attacks the neutral magnet identified in §0. Keeps A's proven
  `Sentiment:` cue and A's minimal footprint, but adds *one* tie-breaking sentence that
  raises the bar for neutral. This is the surgical opposite of B: B told the model "factual ⇒
  neutral"; F tells it "when in doubt, do **not** pick neutral".
- **Hypothesis (falsifiable).** *Moving the neutral tie-break threshold is sufficient to fix
  the collapse without any rubric.* If F beats A on FiQA macro-F1 for plutus (raises positive
  recall from 0.23 toward ~0.5 on the B-collapse cell, i.e. reverses the leak) while staying
  within ~2 pts of A on FPB (where the neutral base rate is high and a too-aggressive
  anti-neutral rule could hurt), the neutral boundary is a *tunable knob* and this is the
  headline recommendation for FiQA.
- **Expected effect.** FiQA: clear F1 gain, especially for plutus and Mistral. FPB: neutral
  recall drops slightly (could cost a little accuracy since FPB is neutral-heavy); net FPB F1
  roughly flat because negative/positive recall rises to compensate.
- **Risk.** Over-correction — on FPB the rule may convert genuine neutrals into
  positive/negative and *lower* FPB accuracy. This asymmetry (helps FiQA, risks FPB) is
  itself the finding: it would prove the two datasets need *different* neutral thresholds, and
  that a single "best template" is the wrong framing.

### G — FiQA-targeted, author-perspective / microblog-aware framing
```python
"G": (
    "The text below is a financial headline or social-media post. Judge the sentiment the "
    "author expresses toward the company or asset mentioned — how they feel about its "
    "prospects, not whether the fact stated is good in the abstract.\n"
    "Answer positive, negative, or neutral.\n"
    "{fewshot_block}"
    "Text: {text}\n"
    "Sentiment:"
),
```
- **Rationale.** FiQA is the universal weak spot (best cell only 0.673). Its text is
  microblog/headline and the gold label is *author* sentiment toward a target, bucketed from a
  continuous score. A/B/C all frame the task as sentiment of *the text/situation*; G reframes
  it as sentiment of *the author toward the asset*, which is the actual FiQA labeling
  definition. Retains A's `Sentiment:` cue.
- **Hypothesis (falsifiable).** *FiQA weakness is partly a task-definition mismatch, not just
  difficulty.* If G lifts FiQA macro-F1 above the current 0.673 ceiling for at least one model
  (target: Qwen, the strongest, > 0.70) while *not* helping — or even mildly hurting — FPB
  (whose labels are situation-sentiment, not author-sentiment), that double dissociation
  proves the datasets need different framings and that a single template is mis-specified for
  one of them.
- **Expected effect.** FiQA up (especially Qwen and Mistral); FPB flat-to-slightly-down.
- **Risk.** For FPB the "author expresses" framing is *wrong* (FPB is annotator-labeled
  situation sentiment), so G should not be used as a shared template — it is explicitly a
  FiQA specialist. Running it on FPB is only to confirm the dissociation, not to win.

### H — My best idea: contrastive anchor + forced-choice, order-counterbalanced
```python
"H": (
    "Task: label the financial text's sentiment toward the company or asset.\n"
    "positive = clearly better prospects (growth, gains, gains for investors)\n"
    "negative = clearly worse prospects (losses, risk, decline)\n"
    "neutral = genuinely no directional signal (choose this only as a last resort)\n"
    "Pick the single best-fitting label. Output only that one word.\n"
    "{fewshot_block}"
    "Text: {text}\n"
    "Answer:"
),
```
- **Rationale.** Combines the three things the data says work: (1) a *directional* neutral
  definition like C's "no directional signal" instead of B's "factual"; (2) F's explicit
  neutral-as-last-resort demotion; (3) a completion cue (`Answer:`) plus "output only that one
  word" belt-and-suspenders for coverage. Crucially the definitions are ordered
  positive→negative→neutral with neutral *last*, deliberately demoting it in reading order —
  the inverse of B, whose neutral sits third but is defined most invitingly. It is B's
  *structure* (a rubric models can follow) with B's *content defect* (the factual-neutral
  magnet) surgically removed and neutral demoted.
- **Hypothesis (falsifiable).** *A well-designed rubric can keep Qwen's B-on-FiQA win (0.673)
  AND avoid the plutus/Mistral collapses.* If H matches or beats B for Qwen-FiQA while
  matching or beating A for plutus-FiQA and Mistral-FPB, then the problem was never
  "rubric vs no rubric" — it was *which* rubric — and H is the new default template.
- **Expected effect.** The best all-rounder: within noise of A on FPB, within noise of B's
  best on Qwen-FiQA, and materially above B for plutus/Mistral. Highest coverage of the
  rubric-style templates.
- **Risk.** Longest of the "should be safe" prompts (though far shorter than E); a slim chance
  the neutral demotion over-corrects on FPB like F. Middle placement of neutral in the *output
  instruction* is avoided, but the effect of definition-ordering on label bias is itself
  unproven — which is partly the point of testing it.

---

## 3. What to actually run (≈$0.50 budget)

Per-template 0-shot cost ≈ $0.10–0.15 for 2 models × 2 datasets × 300 samples. Budget buys
**~3 standard templates**, or 2 standard + a shave. E costs ~2× (needs `max_new_tokens=40`)
and eats ~half the budget alone, so E is **deferred** unless a follow-up budget appears.

**Run these three, 0-shot, both datasets:**

1. **F (neutral tie-break)** — *highest value-per-dollar.* Cheapest way to test the single
   most important hypothesis (neutral magnet is a tunable threshold). Directly actionable
   whichever way it falls, and its FPB-vs-FiQA asymmetry is itself a headline finding.
2. **H (contrastive rubric, neutral demoted)** — the candidate most likely to become the new
   *default* template. If it holds Qwen-FiQA while fixing plutus/Mistral it resolves the whole
   A-vs-B tension in one row.
3. **D (strict structured, no definitions)** — the clean control that *isolates* the
   definition list from the one-word constraint. Cheap, and it lets you attribute B's damage
   correctly. Without D, F/H results are confounded (you won't know if their gains come from
   dropping definitions or from the neutral rule).

Defer **G** (FiQA-specialist; run only if F/H both under-perform on FiQA and you want to test
the task-definition-mismatch hypothesis) and **E** (best information but budget-breaking; run
in a second wave with `max_new_tokens=40` and the `Label:`-suffix parse fix).

**Which two models give the most information:** **plutus-8B** and **Mistral-7B**.
- **plutus** is the focal model *and* the biggest B-victim (−0.165 on FiQA) — it exhibits the
  largest, most diagnostic swings, so it's where a fix will most visibly show up.
- **Mistral** is the most **template-sensitive** model (B collapses it on FPB, few-shot swings
  it +8.7 on FPB / −2.3 on FiQA) — it is the sensitive instrument that will detect whether a
  new template helps or hurts.
- **Qwen** is deliberately *excluded* from the paid runs to save budget: it is the *least*
  sensitive (tiny std across templates, `single_std_acc` ≈ 0.014 on FPB in the ensemble
  table) and already near ceiling, so it carries the least marginal information per dollar.
  Add Qwen only for the one template that wins on plutus+Mistral, to confirm it doesn't *break*
  the strong general model (it currently *owns* the best FiQA cell via B, so H must be checked
  against Qwen-FiQA before adopting H as default).

---

## 4. Is "`Sentiment:` cue (A)" vs "`one word only` (B)" the right primary axis? — the confound

**No — and the current A/B comparison cannot answer the question it appears to ask.**

A and B differ on **at least four axes simultaneously**:
1. persona/framing (none vs "financial analyst / investor perspective"),
2. definition list (absent vs 3-line rubric),
3. neutral definition content ("factual ⇒ neutral" magnet, present only in B),
4. output-format style (`Sentiment:` completion cue vs "answer with one word only" instruction).

The task's framing calls #4 "the primary axis", but the evidence points squarely at **#3** as
the driver: it is the neutral-definition magnet, not the output-format style, that explains
the plutus-FiQA and Mistral-FPB collapses (neutral precision 0.16; negative recall 0.47). So
the honest statement is: **A vs B is a fully confounded 4-way change, and attributing B's
damage to "one word only" would be wrong.** The output-format style change (#4) is almost
certainly the *least* important of the four, yet it is the one the axis-name foregrounds.

**Cleanest experimental addition to deconfound.** Add **two output-format-only variants that
hold everything else fixed** and flip *only* axis #4 — a 2×2 on {A-body, B-body} ×
{`Sentiment:` cue, `one word only` instruction}:

- **A′** = Template A body **+** B's "Answer with one word only (…)" ending (drop the
  `Sentiment:` cue). Same content as A, B's output-format style.
- **B′** = Template B body **+** A's `Sentiment:` completion cue (drop "answer with one word
  only"). Same content as B, A's output-format style.

Now the four cells {A, A′, B′, B} let you read each axis cleanly:
- **A → A′** and **B′ → B** isolate the *pure output-format effect* (cue vs instruction) with
  content held constant. If A′≈A and B′≈B, output-format style is a **non-axis** — kill it as a
  variable and stop calling it primary.
- **A → B′** and **A′ → B** isolate the *pure content/definition effect* with output-format
  held constant. This is where the real signal (the neutral magnet) will show up, and it will
  finally be measured *without* the format confound.

Predicted result: A′≈A, B′≈B (format is near-irrelevant), while the content axis carries
nearly all of B's regression — proving the neutral-definition content, not "one word only", is
the story. Candidate **D** already partly serves as the A-content-with-strict-format cell; A′
and B′ complete the square. This 2-cell addition (A′, B′, on plutus+Mistral, 0-shot, both
datasets ≈ $0.20–0.30) is the single most rigorous thing the budget can buy — recommend it as
the wave-2 companion to F/H/D if the confound needs to be nailed for the writeup.
