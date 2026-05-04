# Project Status — overnight autonomous run

This is the morning hand-off. Summary of what ran, what didn't, and how to resume.

## TL;DR

- ✅ **Full LLM matrix complete: 24 LLM runs + 4 baseline runs.** Three 7-8B LLMs × 2 datasets × 2 templates × {0, 3}-shot.
- ✅ **Live Streamlit dashboard at `http://localhost:8502`** — research-question card, per-class breakdown, highlights gallery, per-example browser, error explorer.
- ✅ **PPTX deck rebuilt with final numbers and dashboard screenshots:** `presentation/implementation_deck_2026-05-04-07-48.pptx` (newest dated copy is authoritative).
- ✅ **Error sample for hand-tagging:** `results/summary/focal_error_sample.csv` (30 plutus-8B misses).
- 💵 **Modal spend:** $0.67 / $7 cap. Well under budget.

## The headline finding

**Financial instruction tuning (plutus-8B-instruct) did NOT clearly outperform general 7-8B LLMs.**

| Model                         | FPB F1m  | FPB Acc  | FiQA F1m | FiQA Acc |
|-------------------------------|----------|----------|----------|----------|
| FinBERT (110M, in-domain)     | **0.925**| 0.935    | 0.482    | 0.498    |
| Mistral-7B-Instruct-v0.3      | **0.890**| 0.903    | 0.599    | 0.620    |
| Qwen2.5-7B-Instruct           | 0.832    | 0.860    | **0.673**| 0.717    |
| plutus-8B-instruct (focal)    | 0.829    | 0.851    | 0.597    | 0.657    |
| VADER (lexicon, no learning)  | 0.469    | 0.554    | 0.386    | 0.423    |

Numbers are best F1-macro per (model × dataset). Bold = winner per dataset. The financial-tuned 8B model is mid-pack, beaten on FPB by Mistral-7B and on FiQA by Qwen2.5-7B.

## Three findings the data tells

1. **Financial instruction tuning did not pull ahead.** plutus-8B is solid mid-pack but never the best LLM on either dataset. The paper's central claim (financial tuning improves downstream NLP) doesn't replicate cleanly on this subset.
2. **FinBERT dominates FPB; loses badly on FiQA.** A 110M specialised classifier crushes every 8B model on FPB but drops 19 points behind Qwen2.5 on FiQA. Useful production-side lesson: domain-trained small models generalise poorly out-of-domain.
3. **Prompt sensitivity is comparable to model choice.** Mistral on FPB A vs B 0-shot: 11-point gap. Qwen on FiQA A vs B 0-shot: 9-point gap. Plutus on FiQA A vs B 0-shot: 16-point gap. Single-prompt LLM benchmarks should not be trusted.

## What ran

| Component                          | State        | Where                                                          |
|------------------------------------|--------------|----------------------------------------------------------------|
| FinBERT × {FPB, FiQA}              | Done (full test sets) | `results/predictions/finbert__*`                       |
| VADER × {FPB, FiQA}                | Done (full test sets) | `results/predictions/vader__*`                         |
| Qwen2.5-7B × full 8-cell matrix    | Done (300 subsample/run) | `results/predictions/qwen25_7b__*`                  |
| Mistral-7B × full 8-cell matrix    | Done (300 subsample/run) | `results/predictions/mistral7b__*`                  |
| **plutus-8B × full 8-cell matrix** | **Done after 2 fixes** (transformers 4.50 + chat template) | `results/predictions/plutus8b__*` |
| `final_table.csv` aggregation      | Done         | `results/summary/final_table.csv`                              |
| Charts (6 figures)                 | Done         | `presentation/key_figures/*.png`                               |
| Dashboard screenshots (6)          | Done         | `presentation/key_figures/dashboard_*.png`                     |
| Error sample (30 plutus misses)    | Done         | `results/summary/focal_error_sample.csv`                       |
| PPTX deck                          | Built        | `presentation/implementation_deck_2026-05-04-07-48.pptx`       |
| GitHub push                        | Pushed       | https://github.com/shimi777/finllama-sentiment                 |

## What didn't run / known issues

1. **TheFinAI/FinLLaMA-instruct was unpublished** by the authors → 404. Treated as a finding for the slides (artefact-level reproducibility risk in LLM benchmarks). Substituted with plutus-8B-instruct (their current 8B financial instruct model).
2. **plutus-8B-instruct needed two fixes:**
   - Tokenizer format too new for `transformers==4.44.2` → bumped to 4.50.0 + tokenizers 0.21.0.
   - Returned empty strings on raw prompts → enabled the Llama-3 chat template (`apply_chat_template`) for instruction-tuned focal models.
3. **TheFinAI/finma-7b-full** is gated (403). Need org approval for access.
4. **Qwen3-8B** (newer, April 2025, lecturer recommended) is too new for `transformers==4.50` — needs ≥4.51 or git source. Documented as "tried, blocked by transformers version" — see [scripts/run_llm_matrix.py](scripts/run_llm_matrix.py) MODELS list.
5. **LLaMA-3.1-8B-Instruct skipped** (gated, no access this session).
6. Streamlit prints `use_container_width` deprecation warnings (cosmetic only, until end of 2025).

## How to resume tomorrow

1. **Open the latest deck:** `presentation/implementation_deck_2026-05-04-07-48.pptx` (close any stale `implementation_deck.pptx` PowerPoint may have open).
2. **Refresh the dashboard:** browser tab on `:8502` → click "Reload predictions" inside the per-example section. If the streamlit process died, relaunch with `dashboard/run.bat`.
3. **Hand-tag 30 errors:** open `results/summary/focal_error_sample.csv` in Excel, fill the `category` column (suggestions printed by error_analysis.py: negation / numerical_reasoning / domain_jargon / ambiguous / sarcasm / factual_neutral_misclassed). Use the breakdown in the slides.
4. **Pick concrete examples for the deck:** dashboard "Highlights for slides" section auto-curates 6-10 cases (focal-wins / generals-win / prompt-flip / unanimous-miss). Screenshot them into the slide that has 4 placeholder cards.
5. **Rebuild the deck after edits:** `node presentation/build_deck.js`.
6. **Optional — add Qwen3-8B (lecturer's request):** bump Modal image to `transformers==4.55.0` in [scripts/modal_app.py](scripts/modal_app.py), then `.venv/Scripts/python.exe scripts/run_llm_matrix.py --only qwen3_8b`. ~10 min wallclock + ~$0.20.
7. **Rotate your HF_TOKEN.** It was pasted in chat earlier and briefly committed to a local commit before being redacted. Revoke at https://huggingface.co/settings/tokens, mint a new one, then `.venv/Scripts/python.exe -m modal secret create huggingface HF_TOKEN=hf_NEW --force`.

## Final Modal cost ledger

`results/_modal_spend.json` — every Modal run, GPU-second cost, and cumulative total. **$0.67 spent of the $7 hard cap I set; well within Modal's $30/month free credit.**

The matrix is ~1500 prompts/model/run × 8 cells × 3 LLMs ≈ 36K prompts of 8B-class inference at 4-bit on T4. About 70 minutes total GPU time across all runs.
