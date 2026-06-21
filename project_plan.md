# FinLLaMA-Instruct Sentiment Analysis — תכנית פרויקט

**סמינר:** LLMs in Finance · Implementation Presentation (30 min)
**מאמר מקור:** Open-FinLLMs (Huang et al., 2024), arXiv:2408.11878
**צוות:** 2 אנשים · **זמן:** 1-2 שבועות · **Runtime:** Google Colab T4 (16GB VRAM)

---

## 1. מטרת הפרויקט

לבדוק האם ה-instruction tuning הפיננסי של FinLLaMA-Instruct באמת מוסיף ערך על ביצועי sentiment classification, בהשוואה ל-baselines ממספר קטגוריות שונות (lexicon, classic NLP, general LLM).

**השאלה המרכזית:** כמה באמת תורם instruction tuning על corpus פיננסי ביחס למודל הבסיס עליו הוא נבנה (LLaMA-3.1-8B), וביחס למתודות פשוטות יותר?

זו בדיוק הטענה המרכזית של המאמר — ואנחנו בודקים אותה על subset קטן באופן בלתי תלוי.

---

## 2. Scope — מה בפנים ומה בחוץ

### בפנים
- 2 datasets: Financial PhraseBank (FPB) ו-FiQA-SA
- 4 מודלים להשוואה: FinLLaMA-Instruct, LLaMA-3.1-8B-Instruct, FinBERT, VADER
- 3 הגדרות: zero-shot, 3-shot, 5-shot (רק עבור LLMs)
- Metrics: Accuracy, F1-macro, F1-weighted, confusion matrix, כיסוי parsing
- Error analysis: סיווג שגיאות ב-3-4 קטגוריות
- Prompt sensitivity: 2 templates שונים כדי לכמת שונות

### בחוץ (מוגדר במודע)
- Fine-tuning — רק inference
- Multimodal — טקסט בלבד
- Trading simulation — רק classification
- Datasets נוספים (Twitter, headlines וכו')
- מודלים סגורים (GPT-4, Claude)

---

## 3. מבנה הפרויקט

```
finllama-sentiment/
├── README.md                     # הסבר קצר, הוראות הרצה
├── requirements.txt              # תלויות Python
├── .gitignore
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA על שני datasets
│   ├── 02_baselines.ipynb           # FinBERT + VADER (מהיר, CPU)
│   ├── 03_finllama_inference.ipynb  # FinLLaMA + LLaMA baseline (GPU)
│   └── 04_analysis_results.ipynb    # ניתוח, גרפים, error analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # טעינה ואחידות תוויות של FPB + FiQA
│   ├── prompts.py                # prompt templates + few-shot examples
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_runner.py         # inference wrapper ל-FinLLaMA / LLaMA
│   │   ├── finbert_runner.py     # wrapper ל-FinBERT
│   │   └── vader_runner.py       # wrapper ל-VADER
│   ├── parser.py                 # פענוח תשובת LLM לתווית + coverage tracking
│   ├── evaluation.py             # חישוב metrics, confusion matrix
│   └── utils.py                  # logging, seeding, config helpers
│
├── configs/
│   └── experiment.yaml           # hyperparameters, model IDs, paths
│
├── results/                      # פלטי ריצה (ב-.gitignore חוץ מ-summary)
│   ├── predictions/              # תחזיות גולמיות (JSON per run)
│   └── summary/                  # טבלאות סופיות לפרזנטציה
│
└── presentation/
    ├── slides_outline.md         # מבנה 30 דקות
    └── key_figures/              # גרפים מוכנים להטמעה
```

**הסבר לגבי הגישה:** משלב בין notebook (ל-exploration וויזואליזציה) לבין קוד מודולרי ב-`src/` (כדי שהלוגיקה תהיה נקייה וניתנת לשימוש חוזר בכל notebook). כל notebook מייבא מ-`src/` במקום להחזיק לוגיקה ארוכה בתאים.

---

## 4. Dataset Plan

### Financial PhraseBank (FPB)
- **מקור:** `takala/financial_phrasebank` או `TheFinAI/en-fpb` מ-HuggingFace
- **גודל:** ~4,840 משפטים
- **תוויות:** positive / neutral / negative (3 מחלקות)
- **חלוקה מוצעת:** 20% test = ~970 משפטים (להימנע מ-test מלא בשביל זמן ריצה)
- **גרסה:** נשתמש ב-`sentences_75agree` (הסכמה של 75% מהמתייגים — איכות גבוהה יותר)

### FiQA-SA
- **מקור:** `ChanceFocus/fiqa-sentiment-classification` או המקור המקורי של FiQA-2018
- **גודל:** ~1,174 דוגמאות (headlines + posts)
- **תוויות במקור:** continuous score ב-[-1, 1] — נמפה ל-3 מחלקות:
  - negative: score < -0.1
  - neutral: -0.1 ≤ score ≤ 0.1
  - positive: score > 0.1
- **שימוש:** כל test split — אין כאן train.

### מיפוי תוויות אחיד
שני הדאטהסטים יעברו דרך `data_loader.py` ויצאו עם אותו סכמה:
```python
{"text": str, "label": "positive"|"neutral"|"negative", "dataset": "FPB"|"FiQA", "id": str}
```

### שיקול חשוב — data leakage
FPB פורסם ב-2014, FiQA ב-2018. FinLLaMA אומן על corpus פיננסי עד 2024. **יש סבירות גבוהה שהמודל ראה את הדאטה בטריינינג.** זה תקף גם למאמר המקורי. נציין זאת מפורשות ב-critical assessment של הפרזנטציה.

---

## 5. Models Plan

| מודל | מקור | גודל | טעינה ב-T4 | זמן ריצה צפוי על 2,000 דוגמאות |
|------|------|------|-------------|-------------------------------|
| FinLLaMA-Instruct | `TheFinAI/FinLLaMA-instruct` | 8B | 4-bit (bitsandbytes) | ~30-45 דק' |
| LLaMA-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 4-bit | ~30-45 דק' |
| FinBERT | `ProsusAI/finbert` | 110M | full precision | ~2-3 דק' (GPU) |
| VADER | `vaderSentiment` pip | — | CPU | ~30 שניות |

**חשוב:** LLaMA-3.1-8B-Instruct דורש גישה מ-HuggingFace (gated). יש לבקש גישה מראש (בדר"כ אישור מהיר). אם זו בעיה, חלופה: `meta-llama/Meta-Llama-3-8B-Instruct` או `mistralai/Mistral-7B-Instruct-v0.3`.

### הגדרות inference אחידות
- `temperature=0.0` (deterministic)
- `max_new_tokens=20` (מספיק ל-"positive" + קצת buffer)
- `do_sample=False`
- batch size = 8 (נכוון לפי זיכרון)
- seed מקובע בכל ההרצות

---

## 6. Prompting Strategy

### Template A (minimalist)
```
Classify the sentiment of the following financial text as positive, negative, or neutral.
Text: {text}
Sentiment:
```

### Template B (structured, with definition)
```
You are a financial analyst. Classify the sentiment of the text below from the perspective of an investor.
- Positive: the text suggests favorable conditions, growth, or gains
- Negative: the text suggests unfavorable conditions, losses, or risks
- Neutral: the text is factual without clear positive or negative implication

Text: {text}
Answer with one word only (positive / negative / neutral):
```

### Few-shot construction
- 3-shot ו-5-shot נבנים על-ידי דגימה אקראית (seed קבוע) של דוגמאות מתוך ה-train split של FPB (ב-FiQA אין train, אז נדגום מ-FPB עבור שניהם כדי לשמור על עקביות).
- מקפידים על balance: ב-3-shot = אחד מכל מחלקה, ב-5-shot = 2/2/1 או דומה.

### Parsing
מודול `parser.py` יטפל בווריאציות כמו:
- `"positive"` ✓
- `"The sentiment is positive."` → positive ✓
- `"I'd say positive but..."` → positive (first match)
- `"bullish"` → positive (synonym map)
- `"unclear"` → ❌ nil (רושמים ב-coverage)

נדווח על `parsing_coverage` כ-metric נפרד בכל ריצה.

---

## 7. Experimental Design

### Matrix של ריצות
| Model × Dataset × Prompt × Shots |
|---|
| FinLLaMA × {FPB, FiQA} × {A, B} × {0, 3, 5} |
| LLaMA-3.1 × {FPB, FiQA} × {A, B} × {0, 3, 5} |
| FinBERT × {FPB, FiQA} × — × — |
| VADER × {FPB, FiQA} × — × — |

**סה"כ ~28 ריצות.** בפועל זה הרבה יותר זול ממה שנראה: FinBERT ו-VADER מהירים, והטעינה של המודלים הגדולים נעשית פעם אחת לכל מודל.

### Metrics
- **ראשיים:** F1-macro, Accuracy
- **משניים:** F1-weighted, per-class precision/recall, confusion matrix
- **טכניים:** parsing coverage (LLMs בלבד), runtime per 100 samples

### מניעת leakage ו-bias
- Few-shot examples נדגמים *רק מ-FPB train split*, לא מ-test
- Seed מקובע (42) לכל דגימה אקראית
- לא "מכוונים" את ה-prompt על test set — בוחרים 2 prompts מראש ומדווחים את שניהם

---

## 8. תכנית עבודה שבועית

### שבוע 1 — תשתית ו-baselines

**יום 1: Setup (שניכם יחד, 2-3 שעות)**
- יצירת repo, הגדרת Colab עם Google Drive persistence
- יצירת `requirements.txt`, התקנות
- אימות גישה ל-FinLLaMA ו-LLaMA מ-HuggingFace
- הרצת "hello world" על FinLLaMA-Instruct — וידוא שנטען ב-4-bit ועושה inference על דוגמה אחת

**יום 2: Data pipeline (1 אדם, ~4 שעות)**
- `data_loader.py` מלא, עם tests ידניים
- Notebook 01: EDA — התפלגות מחלקות, אורכי טקסט, דוגמאות מכל מחלקה

**יום 3: Baselines הקלים (1 אדם במקביל, ~4 שעות)**
- `vader_runner.py` + `finbert_runner.py`
- Notebook 02 — הרצת שני ה-baselines על שני ה-datasets
- תוצאות ראשוניות ב-`results/summary/baselines.csv`

**יום 4: LLM infrastructure (אדם טכני, ~5-6 שעות)**
- `llm_runner.py` עם bitsandbytes, batching, progress bars
- `parser.py` עם unit tests על 20 דוגמאות גבול
- `prompts.py` עם שני templates
- הרצת sanity check: FinLLaMA על 50 דוגמאות FPB — וידוא coverage > 95%

**יום 5: הרצות LLM מלאות (שניכם, ~6-8 שעות רקע)**
- הרצת כל ה-matrix (~28 ריצות) — יום הרצה מלא
- ניהול checkpointing: שמירת תחזיות כ-JSON לכל ריצה ב-`results/predictions/`
- הכנת `evaluation.py` במקביל

### שבוע 2 — ניתוח ופרזנטציה

**יום 6-7: Analysis**
- Notebook 04: איסוף כל התחזיות, חישוב metrics, יצירת טבלה סופית
- Confusion matrices בצורת heatmap
- Error analysis ידני: דגימה של 30 שגיאות של FinLLaMA, סיווגן ידנית ל-3-4 קטגוריות (למשל: negation, sarcasm, domain jargon, numerical reasoning)

**יום 8-9: Presentation**
- Outline של 30 דקות (ראה סעיף 10)
- יצירת slides
- חזרות

**יום 10: Buffer**
- תמיד נדרש. ריצה שנופלת, מטריקה חסרה, גרף שלא ברור.

---

## 9. סיכונים ומלכודות — עם mitigation

| סיכון | הסתברות | Mitigation |
|-------|---------|-----------|
| LLaMA-3.1 gated access מתעכב | גבוהה | לבקש ביום 1. חלופה: Mistral-7B-Instruct |
| FinLLaMA לא נטען ב-4-bit על T4 | בינונית | חלופה: 8-bit; אם גם זה לא — Colab Pro (~$12) |
| Parsing coverage נמוך (<85%) | בינונית | lenient parser, רשימת נרדפים, fallback ל-"neutral" |
| Data leakage משמעותי → תוצאות לא אמינות | גבוהה | מזכירים מפורשות בפרזנטציה. בונוס: בודקים על sample קטן מחדשות 2025 |
| ריצות ארוכות + Colab מתנתק | גבוהה | checkpoint כל 100 דוגמאות ל-JSON ב-Drive, resume logic |
| חוסר הסכמה בין prompts (שונות גבוהה) | בינונית | זה **ממצא בפני עצמו** — לדווח בכנות, לא לבחור את הטוב ביותר |
| "הממצאים משעממים" (כל המודלים דומים) | נמוכה | גם זה ממצא. ה-error analysis יראה הבדלים איכותיים גם אם מספרים דומים |

---

## 10. מבנה הפרזנטציה (30 דק')

לפי ה-handout של הסמינר:

| זמן | נושא | תוכן |
|-----|------|------|
| 0-4 | Objective & setup | מה בדיוק ממשנו, איזו טענה מהמאמר, scope |
| 4-7 | System design | Pipeline diagram, מודלים, datasets, dedup של הקוד |
| 7-12 | Experimental design | Matrix, metrics, הגנות מפני leakage, prompt templates |
| 12-22 | Results | טבלת metrics ראשית, confusion matrices, גרף השוואה, 3-4 דוגמאות קונקרטיות |
| 22-27 | Error analysis | 3-4 קטגוריות שגיאה עם דוגמאות, מתי FinLLaMA מצליח איפה שהאחרים נכשלים (ולהיפך) |
| 27-30 | Lessons learned | בעיות reproducibility, prompt sensitivity, מה היינו עושים אחרת, תובנה מרכזית |

---

## 11. Deliverables סופיים

- [ ] Git repo ציבורי עם README שמאפשר להריץ את הכל מ-Colab
- [ ] טבלת תוצאות סופית (`results/summary/final_table.csv`)
- [ ] 4-5 דמויות מרכזיות לפרזנטציה (confusion matrices, bar charts, error examples)
- [ ] מצגת PPTX / Keynote של 30 דקות
- [ ] חזרה פעם אחת לפחות לפני ההצגה האמיתית

---

## 12. שימוש ב-Claude Code — זרימה מוצעת

**אסטרטגיה:** לא לבקש מ-Claude Code לבנות הכל בבת אחת. לעבוד במחזורים של "אפיין → בנה → בחן → חזור".

**רצף מוצע:**
1. "בנה לי את שלד הפרויקט לפי התכנית ב-`project_plan.md`" — Claude יוצר תיקיות וקבצים ריקים
2. "מלא את `src/data_loader.py`" — קוד לטעינה ואיחוד
3. "בנה את `src/prompts.py` עם שני ה-templates"
4. "בנה את `src/parser.py` ותוסיף לו unit tests"
5. "בנה את `src/models/llm_runner.py`"
6. "עכשיו notebook 03 שמשתמש בכל הנ"ל"

**טיפ:** לפני כל שלב, תן ל-Claude Code את `project_plan.md` כקונטקסט. זה חוסך סבבי הסבר.

---

## הערה מסכמת

התכנית הזו שמרנית בכוונה — מעדיפה שליש פחות דוגמאות ושליש יותר ניתוח, על פני scope גדול יותר עם error analysis רדוד. בסמינר, *ההבנה של מה שראיתם* חשובה יותר מכמות ההרצות.

בהצלחה! 🚀

---

## 13. Module Interfaces (design spec)

Goal: freeze data shapes and function signatures before coding, so every notebook and module agrees. All dicts are plain Python dicts (no pydantic).

### Core data shapes

```python
# Unified per-example dict — produced by data_loader, consumed by everything downstream
Sample = {
    "id": str,          # stable per-dataset, e.g. "FPB_00142", "FiQA_0007"
    "text": str,
    "label": str,       # "positive" | "neutral" | "negative"
    "dataset": str,     # "FPB" | "FiQA"
    "split": str,       # "train" | "test"
}

# Per-example model output — produced by runners (after parsing)
Prediction = {
    "id": str,                 # matches Sample.id
    "pred_label": str | None,  # None = parse failure
    "raw_output": str,         # the actual string the model emitted (LLMs only; "" for baselines)
    "parse_ok": bool,
    "latency_ms": float,
}
```

### Function signatures

```python
# src/data_loader.py
def load_fpb(config: str = "sentences_75agree",
             test_fraction: float = 0.20,
             seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    """Returns (train, test)."""

def load_fiqa(neutral_band: float = 0.10) -> list[Sample]:
    """All FiQA as test split. Maps continuous score -> 3 classes using neutral_band."""

# src/prompts.py
TEMPLATES: dict[str, str]  # keys: "A", "B"; values: format strings with {text} and {fewshot_block}

def sample_fewshot(pool: list[Sample], n_shots: int, seed: int) -> list[Sample]:
    """Balanced: n=3 -> 1/1/1, n=5 -> 2/2/1. Sampled only from FPB train."""

def build_prompt(template: str, text: str, few_shot: list[Sample] | None = None) -> str: ...

# src/parser.py
SYNONYMS: dict[str, str]  # e.g. {"bullish": "positive", "bearish": "negative"}

def parse(raw_output: str) -> str | None:
    """Returns canonical label or None. Case-insensitive, first-match wins, synonym-aware."""

# src/models/llm_runner.py
class LLMRunner:
    def __init__(self, hf_id: str, load_in_4bit: bool = True, seed: int = 42): ...
    def generate(self, prompts: list[str], batch_size: int = 8,
                 max_new_tokens: int = 20) -> list[tuple[str, float]]:
        """Returns [(raw_output, latency_ms), ...] in input order."""
    def unload(self) -> None:
        """Free GPU memory before loading the next model."""

# src/models/finbert_runner.py, vader_runner.py
class BaselineRunner:  # same interface for both
    def predict(self, texts: list[str]) -> list[str]:
        """Returns canonical labels, always non-None."""

# src/evaluation.py
def compute_metrics(samples: list[Sample],
                    preds: list[Prediction]) -> dict:
    """
    Returns:
    {
        "accuracy": float,
        "f1_macro": float,
        "f1_weighted": float,
        "per_class": {label: {"precision": ..., "recall": ..., "f1": ..., "support": ...}},
        "confusion": list[list[int]],   # rows = true, cols = pred, order = [neg, neu, pos]
        "coverage": float,              # fraction with parse_ok=True (LLMs); 1.0 for baselines
        "n_samples": int,
    }
    """

# src/utils.py
def set_seed(seed: int) -> None: ...
def get_logger(name: str) -> logging.Logger: ...
def load_config(path: str = "configs/experiment.yaml") -> dict: ...
```

### Invariant: parse failures

When `parse_ok=False`, `pred_label` is `None`. `compute_metrics` excludes those from accuracy/F1 but counts them in `coverage`. This avoids the "force to neutral" bias.

---

## 14. Run Naming & Checkpoint Format

### Run ID schema
```
{model}__{dataset}__{template}__{shots}shot__seed{seed}
```
LLM examples:
- `finllama__FPB__A__0shot__seed42`
- `llama31__FiQA__B__5shot__seed42`

Baseline examples (no template/shots):
- `finbert__FPB__seed42`
- `vader__FiQA__seed42`

### Per-run directory
```
results/predictions/{run_id}/
├── meta.json           # model_hf_id, dataset, template, shots, seed, config_hash,
│                       # started_at, completed_at (null until done), n_total
├── predictions.jsonl   # one Prediction dict per line, appended incrementally
└── progress.json       # {"last_completed_idx": int, "n_total": int, "updated_at": str}
```

### Checkpoint rules
- Flush `predictions.jsonl` + rewrite `progress.json` every 100 samples.
- **Resume logic:** on startup, if `progress.json` exists and `last_completed_idx < n_total`, skip the first `last_completed_idx` samples and continue appending.
- A run is complete when `last_completed_idx == n_total`; at that point set `meta.completed_at` and do not re-enter.
- `config_hash` = sha256 of the relevant slice of `experiment.yaml` — if it changes, the run is considered stale and must be re-run (rename old dir to `{run_id}.stale_{timestamp}`).

### Summary aggregation
After the matrix completes, `src/evaluation.py` walks `results/predictions/*/` and writes:

`results/summary/final_table.csv` with columns:
```
model, dataset, template, shots, seed,
accuracy, f1_macro, f1_weighted, coverage, n_samples, runtime_s
```

Per-run confusion matrices go to `results/summary/confusions/{run_id}.json` (not CSV, since they are 3x3 matrices).

---

## 15. Colab Setup

### Drive mount + repo location
```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_DIR = '/content/drive/MyDrive/finllama-sentiment'
# First session: clone the repo into PROJECT_DIR.
# Later sessions: just `cd` in.
%cd {PROJECT_DIR}
```

Why Drive and not ephemeral `/content`: `results/predictions/` must survive disconnects, and the HF cache (~16GB for an 8B model) should not be re-downloaded every session.

### HF auth (gated Llama-3.1)
Store a HuggingFace token in Colab Secrets (left sidebar, key icon) as `HF_TOKEN`, then:
```python
from google.colab import userdata
from huggingface_hub import login
login(userdata.get('HF_TOKEN'))
```
Request Llama-3.1-8B-Instruct access on day 1 — usually approved within hours, but can be 24h.

### Persistent HF cache
```python
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/hf_cache'
```
Set these **before** importing `transformers`. Drive I/O is slower than local disk, but one-time download >> repeated downloads.

### Runtime & deps check (top of every notebook)
```python
import torch
assert torch.cuda.is_available(), "Select Runtime -> T4 GPU"
!pip install -q -r {PROJECT_DIR}/requirements.txt
```

### Disconnect strategy
We don't try to prevent disconnects. Section 14's checkpoint/resume makes them cheap: at worst we lose <100 samples of progress. Do not rely on long-lived `nohup`-style tricks — they fight the platform.

### Memory hygiene between LLMs
Between FinLLaMA and LLaMA-3.1 runs, call `LLMRunner.unload()` and:
```python
import gc, torch
gc.collect(); torch.cuda.empty_cache()
```
Otherwise 4-bit + 4-bit won't fit on 16GB T4.

---

## 16. שיפור: Prompt-Ensemble Voting (multi-prompt)

**הקשר:** הפרופ' ביקש שיפור שמשתמש ב-multiple prompts כדי להגיע ל-accuracy/performance טובים יותר. הסעיף הזה הוא ה-source of truth לשיפור הזה.

### 16.1 מוטיבציה — מתוך התוצאות שלנו

ה-`final_table.csv` מראה **prompt sensitivity קיצונית**: על אותו מודל, רק החלפת ה-prompt מזיזה accuracy עד **26 נקודות** (plutus8b · FiQA: A/0-shot=0.657 מול B/0-shot=0.400). כדי לדעת איזה prompt הוא "הטוב ביותר" צריך להציץ ב-test labels — **וזה leakage**. למשתמש אמיתי אין דרך עקבית לבחור מראש.

**הטענה:** majority vote על כמה prompts קבועים ומגוונים מבטל את ההימור של בחירת ה-prompt — בלי להציץ ב-test — ובכך נותן תוצאה (א) טובה מהממוצע של prompt בודד, (ב) יציבה (variance≈0), ו-(ג) שואפת ל/עוקפת את ה-prompt הטוב ביותר. בונוס: מאפשר השוואת מודלים *הוגנת* (בלי prompt-luck) — תורם לשאלה המרכזית "האם ה-tuning הפיננסי של plutus8b באמת עוזר".

### 16.2 השיטה

לכל דוגמה: אוספים את התווית המפוענחת מכל ensemble member ומבצעים **majority vote**. חברים עם `parse_ok=False` נמנעים (לא מצביעים) — שומר על ה-invariant של הפרויקט "לא לכפות parse-failure לתווית". מימוש: `src/ensemble.py::aggregate(votes, tie_break)`, פלט תואם ל-`compute_metrics`.

### 16.3 וריאנטים של ה-ensemble

- **E4 (חינמי, מהקיים):** vote על 4 התאים שכבר רצנו — `{A,B} × {0,3}-shot`. אפס inference חדש; מחושב מ-`results/predictions/`.
- **E3 (headline, נקי):** שלושה ניסוחי prompt מגוונים ב-0-shot — `{A, B, C}`. מספר אי-זוגי → פחות תיקו; 0-shot → ריאליסטי ל-deployment ועובד גם ל-FiQA שאין לו train. **דורש template C חדש.**
- **E5 (extended):** `{A0, B0, C0, A3, B3}` — מוסיף few-shot members למקסימום גיוון.

### 16.4 Aggregation ו-tie-break

ברירת מחדל: plurality; **תיקו → abstain** (`parse_ok=False`, נספר ב-coverage). זה הישר וה-leakage-free. אלטרנטיבות שנבדקו: `order` (לפי [neg,neu,pos]) ו-`neutral` (majority-class prior) — שתיהן מחזירות coverage ל-1.0 אבל **פוגעות ב-F1** (ראה 16.6), כי דוגמאות התיקו באמת עמומות. ב-E3 (3 חברים) תיקו 2-2 לא קיים כך ש-coverage עולה ממילא.

### 16.5 Evaluation protocol

לכל (model × dataset) מדווחים זה לצד זה: **mean-single**, **worst-single**, **best-single (oracle = חסם עליון שדורש leakage)**, ו-**ensemble**, בתוספת **std של accuracy בין ה-prompts** (לכימות ה-variance שנמחק). מטריקה ראשית: F1-macro. סקריפט: `scripts/aggregate_ensemble.py` → `results/summary/ensemble_table.csv`.

### 16.6 תוצאות PoC — E4 (2026-06-21, tie→abstain)

| model | dataset | ens_f1 | mean_f1 | best_f1 | Δ vs mean | Δ vs best | ens_acc | std(single) | cov |
|---|---|---|---|---|---|---|---|---|---|
| mistral7b | FPB | 0.8306 | 0.7585 | 0.8897 | **+0.072** | −0.059 | 0.875 | 0.057 | 0.88 |
| plutus8b | FPB | 0.7857 | 0.7210 | 0.8291 | **+0.065** | −0.043 | 0.856 | 0.053 | 0.90 |
| qwen25_7b | FPB | 0.8316 | 0.7986 | 0.8324 | **+0.033** | −0.001 | 0.868 | 0.014 | 0.96 |
| mistral7b | FiQA | 0.5945 | 0.5589 | 0.5989 | **+0.036** | −0.004 | 0.616 | 0.039 | 0.86 |
| plutus8b | FiQA | 0.5094 | 0.4877 | 0.5966 | **+0.022** | −0.087 | 0.496 | 0.099 | 0.85 |
| qwen25_7b | FiQA | 0.6395 | 0.5800 | 0.6727 | **+0.060** | −0.033 | 0.651 | 0.068 | 0.87 |

**Tie-break ablation (ממוצע על 6 התאים):** abstain → F1=0.699, cov=0.89 · order → F1=0.665, cov=1.0 · neutral → F1=0.634, cov=1.0. → **abstain עדיף ל-accuracy; כפיית coverage עם order/neutral עולה ב-F1.**

**מסקנות (unweighted):**
1. ✅ **מנצח את הממוצע (blind-pick) ב-6/6** — +0.02 עד +0.07 F1. זה ה-win המוצק.
2. ❌ **לא מנצח את ה-oracle best ב-0/6** (אך qwen·FPB בתוך 0.001, mistral·FiQA בתוך 0.004).
3. ✅ **Variance נמחק** — מחליף הגרלה עם std≈0.055 בערך דטרמיניסטי יחיד שתמיד מעל הממוצע.

### 16.6.1 תוצאות weighted voting — cv_weighted מול oracle (tie→abstain)

cv_weighted = soft vote עם משקלי-חברים שנלמדים ב-**k-fold (k=5) leakage-free** (משקל של חבר אף פעם לא רואה את הדוגמה שהוא שופט). oracle_weighted = משקלים לפי accuracy על *כל* ה-eval set (משתמש ב-test labels — חסם עליון בלבד, לא deployable).

| dataset | model | unweighted | cv_weighted | oracle_weighted | best_single |
|---|---|---|---|---|---|
| FPB | mistral7b | 0.8306 | 0.8428 | 0.8428 | 0.8897 |
| FPB | plutus8b | 0.7857 | 0.7519 | 0.7622 | 0.8291 |
| FPB | qwen25_7b | 0.8316 | 0.8167 | 0.8252 | 0.8324 |
| FiQA | mistral7b | 0.5945 | 0.5993 | **0.6044** | 0.5989 |
| FiQA | plutus8b | 0.5094 | 0.5235 | 0.5235 | 0.5966 |
| FiQA | qwen25_7b | 0.6395 | 0.6267 | 0.6267 | 0.6727 |

ממוצע על 6 תאים: unweighted F1=0.699 @ cov=0.887 · cv_weighted F1=0.693 @ cov=**0.996** · oracle F1=0.698 @ cov=1.0.

**שתי מסקנות מפתיעות וחשובות:**
1. **weighting הוא בעיקר פתרון ל-coverage, לא ל-F1.** משקלים רציפים מבטלים תיקו (2-2) ולכן coverage קופץ 0.887→0.996 כמעט בלי לאבד F1. cv_weighted **שולט (Pareto)** על כל דרך אחרת להגיע ל-coverage מלא: 0.693 מול 0.665 (order) ו-0.634 (neutral).
2. **אפילו ה-oracle ceiling עוקף את best single רק ב-1/6 תאים.** כלומר עם ה-pool הזה (A/B × 0/3-shot), **שום aggregation — משוקלל או לא — לא יכול לעקוף באופן עקבי את ה-prompt הבודד הטוב ביותר.** הסיבה עקרונית: vote לא יכול לעלות על החבר הטוב ביותר כשהתשובות הנכונות שלו הן עמדת מיעוט. ⇒ כדי לעקוף את ה-oracle צריך **חברים חדשים ומגוונים יותר** (template C / E3-E5), לא aggregation חכם יותר.

### 16.7 Success criteria (מעודכן לפי הממצאים)

- **Phase A — E4 unweighted (חינמי):** ens F1 > mean single ב-**6/6** ✅ · variance מבוטל ✅. *(הושג.)*
- **Phase B-agg — weighted voting (חינמי):** aggregator עם coverage מלא ש**שולט** על tie-break כפוי — cv_weighted F1=0.693 @ cov=0.996 מול 0.665/0.634. ✅ *(הושג; "ה-best deployable aggregator".)*
- **Phase B-div — member diversity (GPU, טרם):** ens F1 ≥ best single ב-**≥ 3 מתוך 6** תאים. הוכח אנליטית שלא ניתן עם ה-4 חברים הקיימים → דורש **E3 = שלישיית 0-shot `{A,B,C}`** (template C כבר קיים ב-`src/prompts.py`, צריך הרצת Colab) ואולי E5.

### 16.8 צעדים הבאים

1. ✅ **template C** נוסף ל-`src/prompts.py` ו**מחווט ל-matrix** (`run_llm_matrix.py`, `TEMPLATE_SHOTS` → C ב-0-shot בלבד = ריצה אחת נוספת לכל model×dataset). הדרייבר כבר מחשב E3/E5 אוטומטית ברגע שתחזיות C קיימות. **נותר רק:** הרצת GPU/Modal —
   ```
   python scripts/run_llm_matrix.py --only qwen25_7b mistral7b plutus8b
   ```
   (ריצות A/B קיימות מדולגות דרך progress.json; רק 6 ריצות C0 חדשות) ואז `python scripts/aggregate_ensemble.py`.
2. ✅ **weighted / cv_weighted voting** מומש ב-`aggregate` + driver. שדרוג אופציונלי: **soft voting לפי first-token logprob** (דורש שינוי קטן ב-`llm_runner`) במקום משקלי-accuracy.
3. אופציונלי — **cascade** לחיסכון: prompt זול קודם, escalation ל-vote מלא רק על דוגמאות עם disagreement → accuracy של ensemble בעלות חלקית (ה-"performance" angle).
4. ✅ **figures** נוצרו (ראה 16.9).

### 16.9 Deliverables

- `src/ensemble.py` — `aggregate(votes, tie_break, weights)` (+ `tests/test_ensemble.py`, 16 בדיקות).
- `scripts/aggregate_ensemble.py` — driver: unweighted + tie-break ablation + cv_weighted + oracle ceiling.
- `scripts/make_ensemble_figures.py` → `presentation/key_figures/ensemble_vs_single.png`, `ensemble_coverage_tradeoff.png`.
- `results/summary/ensemble_table.csv` (30 שורות) + `results/summary/confusions_ensemble/`.
- template C ב-`src/prompts.py` (Phase B-div, ממתין להרצת GPU).
