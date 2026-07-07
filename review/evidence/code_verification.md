# Stage 4 — Code Verification

Branch: `review/pre-submission`. All commands run with the venv interpreter only
(`C:\python_projects\finllama-sentiment\.venv\Scripts\python.exe`), from repo root,
unless noted otherwise.

---

## 1. `pytest -q`

Command: `.venv\Scripts\python.exe -m pytest -q`

**Result: 90 passed, 11 failed** (not the expected ~101 passed / 0 failed).

```
11 failed, 90 passed in 30.20s
```

All 11 failures are in `tests/test_evaluation.py` (every test in that file), all with
the identical root cause — a `TypeError` raised while importing `sklearn.metrics`
(triggered by `src/evaluation.py:15`, a lazy in-function import):

```
scipy\_lib\array_api_compat\common\_helpers.py:69: in _issubclass_fast
    return issubclass(cls, parent_cls)
TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union
```

Root cause confirmed independently: this venv resolved
`torch==2.11.0+cpu`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `numpy==2.4.4`. scipy's
`array_api_compat.is_torch_array()` calls `issubclass(cls, torch.Tensor)`, but in this
torch build `torch.Tensor` is not a plain class — `type(torch.Tensor)` is
`torch._C._TensorMeta`, and `issubclass()` rejects it outright
(`issubclass(np.ndarray, torch.Tensor)` raises the same `TypeError` when run
standalone). `requirements.txt` pins only lower bounds (`torch>=2.3.0`,
`scikit-learn>=1.4.0`, no upper bound on scipy at all), so a fresh `pip install` today
resolves into this broken combination.

**Verdict: FINDING.** Not a bug in this repo's test or source logic — it's an
environment/dependency-pinning gap. Every test that calls `compute_metrics()` (i.e.
anything touching `src/evaluation.py`) will fail on a freshly-installed venv until
either scipy is downgraded/pinned away from this range, or torch/scipy are pinned to a
mutually-compatible combination. No warnings otherwise; no other test files affected.

---

## 2. Aggregation re-run and drift check

### 2a. `scripts/aggregate.py`

Ran cleanly, wrote 44 rows to `results/summary/final_table.csv` (matches expectation:
no exceptions, valid table printed).

`git diff --stat` showed the committed file gained **16 new rows** (pure appends, no
reordering or reformatting of the 28 previously-committed rows) plus 16 new untracked
`results/summary/confusions/*.json` files, all tagged either `FPBall` (a dataset
variant) or `finbert_tone` (a model not in the committed table).

Cross-checked against `results/predictions/`: 13 run directories tagged `FPBall` and 2
tagged `finbert_tone` exist on disk with complete `meta.json`/`progress.json`
(`finbert_tone` runs dated 2026-05-11; `FPBall` runs dated 2026-06-25) — i.e. genuine,
finished local experiment runs that were executed **after** the committed
`final_table.csv` (last touched in commit `9be517c`, 2026-05-04) and never
re-aggregated into the committed summary.

**Diagnosis: stale committed artifact, not nondeterminism.** All previously-committed
rows are byte-identical; the diff is purely additive, driven by real runs sitting in
`results/predictions/` that post-date the last summary commit.

Full diff saved to `review/evidence/aggregate_drift.diff`. Working tree restored via
`git checkout -- results/summary` + manual removal of the newly-created untracked
confusion JSONs.

### 2b. `scripts/aggregate_ner.py`

Ran cleanly, wrote 7 rows to `results/summary_ner/final_table_ner.csv`.

`git diff --stat` showed **6 tracked files modified** (not just additions this time):
`final_table_ner.csv` plus 5 of the 7 confusion JSONs (`gliner-large`, `gliner-small`,
`mistral7b`, `plutus8b`, `qwen25_7b` — all `FiNER-ORD`), plus 2 new untracked files
(`qwen3_4b`, `qwen3_8b` — these are simply new rows, analogous to case 2a).

The 5 modified files are the interesting case — two distinct effects:

- **Stale predictions, same root cause as 2a.** The committed
  `mistral7b__FiNER-ORD...json` records `n_samples: 200`, but
  `results/predictions/mistral7b__FiNER-ORD__A__0shot__seed42/predictions.jsonl` on
  disk has 300 lines today. The committed `final_table_ner.csv` row for mistral7b/
  plutus8b/qwen25_7b all show `n_samples=200`; a fresh aggregation correctly picks up
  the full 300-row predictions file that exists now. Same story as the sentiment
  table: runs were extended/re-run locally after the last commit, and the aggregated
  summary was never refreshed.
- **A second, smaller effect on the gold side.** For `gliner-large`/`gliner-small`
  — whose `predictions.jsonl` already had 300 rows in both the committed version and
  today — the gold entity `support` count still shifted (314 → 316, and per-type ORG
  144 → 146). `src/ner/data_loader.load_finer_ord()` calls
  `datasets.load_dataset("gtfintechlab/finer-ord-bio", split="test")` with **no
  pinned `revision=`** (confirmed absent from both `src/ner/data_loader.py` and
  `configs/ner.yaml`), and `max_samples=None` so no shuffle/cap is applied. The HF Hub
  dataset's `lastModified` is 2024-09-10 (i.e. not recently changed upstream), and the
  test split loads a stable 1075 rows both times, so this isn't an upstream dataset
  edit or a shuffle-seed issue — it's most likely differing BIO→span extraction
  across the local `datasets`-library cache / environment used to generate the
  original committed artifact vs. this run. Net effect is small (2 extra gold spans
  out of ~314-316) but demonstrates the NER aggregation is not perfectly
  reproducible bit-for-bit across environments, independent of the (larger,
  same-root-cause-as-2a) stale-predictions issue.

Full diff saved to `review/evidence/aggregate_ner_drift.diff`. Working tree restored
via `git checkout -- results/summary_ner` + manual removal of the two new untracked
confusion JSONs (qwen3_4b, qwen3_8b).

### 2c. `scripts/error_analysis.py`

Ran cleanly, wrote `results/summary/focal_error_sample.csv` (30 misses sampled for
hand-tagging). `git diff` on this file was **empty** — regenerated output is
byte-identical to the committed version. Clean pass, nothing to restore.

**Tree state after check 2:** `git status --short results/` returned nothing —
fully restored.

---

## 3. `scripts/make_figures.py`

Command: `.venv\Scripts\python.exe scripts\make_figures.py` — **exit 0**.

Regenerated 6 PNGs under `presentation/key_figures/`
(`f1_comparison.png`, `coverage_heatmap.png`, `confusion_grid.png`,
`per_class_f1_FPB.png`, `per_class_f1_FiQA.png`, `fewshot_effect.png`).

`git status --short presentation/key_figures` showed **no diff** — all regenerated
PNGs are identical to the committed versions (git saw no content change). Clean pass,
nothing to restore.

**Verdict: PASS.**

---

## 4. Dashboard smoke tests

Both dashboards started successfully and served HTTP 200 within ~2 seconds; no
tracebacks in server output.

| Dashboard | Port | Result |
|---|---|---|
| `dashboard/app.py` | 8601 | HTTP 200, clean Uvicorn startup log, no errors |
| `dashboard_ner/app.py` | 8602 | HTTP 200, clean Uvicorn startup log, no errors |

Both processes were killed by PID (`Stop-Process -Id <pid> -Force`) and confirmed dead
(port free, `Get-Process -Id <pid>` returns nothing) — no orphaned processes left.

**Finding (environment, not a code bug):** despite launching with the fully-qualified
venv path (`C:\python_projects\finllama-sentiment\.venv\Scripts\python.exe -m
streamlit run ...`), the actual server process that ended up bound to the port was
running under the **standalone Python 3.12** interpreter
(`C:\Users\shimi\AppData\Local\Programs\Python\Python312\python.exe`), confirmed via
`Get-CimInstance Win32_Process | select CommandLine`. This reproduced on a second,
independent attempt. Streamlit's own process bootstrap appears to re-exec via a bare
`python` PATH lookup rather than staying in `sys.executable`'s process. The standalone
Python 3.12 install happens to have its own separate `streamlit` install (v1.55.0)
alongside the venv's (v1.57.0), so the app still ran correctly and imported the
right code (dashboard scripts don't import `src` in a way that broke), but this is
the same two-Python-install hazard called out in the user's global CLAUDE.md, now
also observed for `streamlit run`, not just `pip install`. Worth knowing if a future
dashboard change needs a package that's only in the venv, not the standalone install.

**Verdict: PASS** functionally (both apps serve, no tracebacks), with the interpreter-
resolution caveat noted above.

---

## 5. Notebooks

Jupyter/nbconvert **available** in the venv (`nbconvert 7.17.1`, `nbformat 5.10.4`,
`ipykernel 7.2.0`).

### 04_analysis_results.ipynb — executed

No venv-backed Jupyter kernel was registered by default — the only kernels present
were `python3` (Anaconda) and `py312` (hardcoded to the standalone
`Programs\Python\Python312\python.exe`), neither of which has this repo's venv
packages. Registered a throwaway kernel (`finllama-venv-tmp`, removed again at the end
of this check) via `.venv\Scripts\python.exe -m ipykernel install --user --name
finllama-venv-tmp`, and drove execution via `nbclient.NotebookClient` (Python API, not
the `nbconvert` CLI) so the working directory could be pinned explicitly.

Execution outcome, cwd = repo root (matching `CLAUDE.md`'s stated convention "the
`src` package is on the path because tests/notebooks run from repo root"):

- Cell 1 (imports + `load_config()`, `cfg['paths']['predictions_dir']`, etc.):
  **passed** — `from src... import ...` resolved, `configs/experiment.yaml` loaded via
  its relative default path `configs/experiment.yaml`, which only resolves when
  cwd = repo root.
- Cell with the FPB/FiQA loaders and cell computing the full 44-row aggregate table
  (the same aggregation logic as `scripts/aggregate.py`): **passed**.
- First figure-saving cell (`F1-macro` bar chart): **FAILED** —
  `FileNotFoundError: '../presentation/key_figures\\f1_macro_FPB.png'`.

**Root cause — a genuine, pre-existing path-convention conflict inside the notebook
itself**, not an artifact of how I invoked nbconvert: `configs/experiment.yaml`'s
`predictions_dir: results/predictions` / `summary_dir: results/summary`, plus
`src/utils.load_config()`'s default arg `path: str = "configs/experiment.yaml"`, are
only resolvable with cwd = repo root. But notebooks 01 and 04 both write figures via
`plt.savefig('../presentation/key_figures/...')`, a path only resolvable with
cwd = `notebooks/`. No cwd satisfies both. (I also tried cwd = `notebooks/` +
`PYTHONPATH` pointed at repo root to fix the `import src` problem without `chdir` —
that made cell 1's `import` succeed but then `load_config()`'s relative-path default
failed instead, confirming the conflict is inherent to the notebook, not to my
driver script.) This likely "worked" for whoever authored the notebook only because
of how their interactive Jupyter session's default cwd happened to be set — it is not
reproducible via a scripted `nbconvert --execute` run from a clean state either way.

Executed notebook (partial — through the failing cell) saved to the scratchpad at
`04_executed.ipynb`; not copied into the repo (per task instructions, execution output
goes to the scratchpad dir only).

**Verdict: FINDING.** Aggregation logic itself (the part that matters most — cells 1–4)
runs correctly under the venv. The notebook cannot currently be executed end-to-end via
`nbconvert`/scripted automation from a single cwd; only the figure-saving cells are
affected, and only because of the `../presentation/key_figures` vs. `configs/…`-relative-path
inconsistency described above.

### 01 / 02 / 03 — static check only (not executed, per instructions)

Checked with `nbformat`: does every code cell have saved outputs, are there any
`output_type == "error"` outputs, and do the notebook's own top-level imports resolve
in this venv.

| Notebook | Code cells | Cells with saved outputs | Error outputs | Imports resolve in venv? |
|---|---|---|---|---|
| `01_data_exploration.ipynb` | 5 | 0 | 0 | Yes (`pandas`, `matplotlib`, `seaborn`) |
| `02_baselines.ipynb` | 5 | 0 | 0 | Yes (`pandas`) |
| `03_finllama_inference.ipynb` | 4 | 0 | 0 | Yes (`torch`) |
| `04_analysis_results.ipynb` (committed, pre-execution) | 7 | 0 | 0 | — (see execution above) |

None of the four notebooks carry saved cell outputs in the committed `.ipynb` JSON —
all were committed "clean" (no evidence they were ever run-and-saved, or outputs were
stripped before commit). No stored error outputs to report. Per task instructions,
01/02/03 were **not executed** (01/02 download datasets/models from the network; 03
needs GPU) — static check only, as directed.

**Verdict:** static PASS for 01/02/03 (nothing on disk indicates a prior failure; all
imports importable); real code correctness of 01–03 remains unverified by execution
(out of scope for this stage per the given instructions).

---

## 6. Deck build

Command: `node presentation\build_deck.js` (from repo root, `node_modules` already
present) — **exit 0**.

Output: `wrote C:\python_projects\finllama-sentiment\presentation\implementation_deck.pptx`

This overwrote the existing tracked `presentation/implementation_deck.pptx`
(722,341 bytes → 942,965 bytes). The new size matches the already-committed
timestamped snapshot `presentation/implementation_deck_2026-05-04-07-48.pptx`
(942,965 bytes) almost exactly, suggesting the tracked `implementation_deck.pptx`
was stale (from an earlier build than the timestamped snapshot sitting next to it)
and the fresh build brings it back in line with the more recent snapshot. Left as
rebuilt per task instructions (only `results/summary`, `results/summary_ner`, and
conditionally the figure PNGs were to be restored; the deck was not on that list, and
nothing was deleted).

**Verdict: PASS**, with the note that the deck was rebuilt and is now modified
relative to git HEAD (`presentation/implementation_deck.pptx`, binary diff, left as
built rather than reverted).

---

## 7. README quickstart vs. reality

`README.md` is effectively a stub, not a working quickstart:

- The `## הרצה (Colab T4)` ("Running — Colab T4") section literally reads
  `TODO: הוראות להרצה מ-Colab + מקור נתונים + טעינת מודלים` ("TODO: instructions for
  running from Colab + data source + model loading"). There are no commands at all —
  no `pip install`, no venv setup, no mention of `scripts/aggregate.py`,
  `make_figures.py`, the dashboards, or how to reproduce `results/summary/final_table.csv`.
- **"Models" section is stale relative to what was actually run.** README lists
  "FinLLaMA-Instruct · LLaMA-3.1-8B-Instruct · FinBERT · VADER". A search of
  `results/predictions/` for any `llama`/`finllama` run directory returned **zero
  matches** — no FinLLaMA or LLaMA-3.1 runs exist anywhere in committed or local
  results. The actual model roster evaluated (per `final_table.csv` and the run
  directory listing) is `finbert`, `finbert_tone`, `vader`, `mistral7b`, `plutus8b`,
  `qwen25_7b` (plus `qwen3_4b`/`qwen3_8b` for the separate NER track) — i.e. the
  project pivoted away from LLaMA-3.1/FinLLaMA (likely gated-model access, per the
  `HF_TOKEN` gotcha noted in `CLAUDE.md`) but the README was never updated to match.

**Verdict: FINDING.** README does not describe a working reproduction path, and its
model list does not match the models the project actually evaluated.

---

## Cleanup performed

- `results/summary/` and `results/summary_ner/` restored to committed state
  (`git checkout --`) after each aggregation re-run; all newly-created untracked
  confusion JSONs deleted.
- Temporary Jupyter kernelspec `finllama-venv-tmp` removed.
- Both Streamlit dashboard processes killed by PID; ports 8601/8602 confirmed free,
  parent wrapper PIDs also confirmed gone.
- `presentation/key_figures/*.png` — no diff after `make_figures.py`, nothing to
  restore.
- `presentation/implementation_deck.pptx` — left as freshly rebuilt (not restored; not
  in the restore list, nothing was deleted).
- Files under `review/evidence/` and `review/scripts/` other than this file and the
  two `*_drift.diff` files were **not created or modified by this stage** — they
  belong to a concurrent Stage 1–3 review session running in parallel against the
  same working tree (timestamps interleave with this session's own timestamps).
