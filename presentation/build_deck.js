// Generate the implementation-presentation deck.
// Output: presentation/implementation_deck.pptx
//
// Style: Midnight Executive (navy + ice blue + white) — clean, academic.
// Layout: 16x9 (10" × 5.625")
//
// To rebuild after results change:
//   node presentation/build_deck.js

const path = require("path");
const fs = require("fs");
const pptxgen = require("./node_modules/pptxgenjs");

const ROOT = path.dirname(__dirname);
const FIG_DIR = path.join(ROOT, "presentation", "key_figures");
// Write to a fresh filename if the primary is locked (PowerPoint open).
function pickOutPath() {
  const primary = path.join(ROOT, "presentation", "implementation_deck.pptx");
  try {
    fs.openSync(primary, "r+"); // probe for lock
    return primary;
  } catch (_) {
    const ts = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
    return path.join(ROOT, "presentation", `implementation_deck_${ts}.pptx`);
  }
}
const OUT = pickOutPath();

// ---- palette ----
const NAVY    = "1E2761";
const ICE     = "CADCFC";
const WHITE   = "FFFFFF";
const INK     = "1E293B";   // body text on light bg
const MUTED   = "64748B";   // captions
const ACCENT  = "F96167";   // sparingly, for "key findings"
const FACE_H  = "Calibri";
const FACE_B  = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "LLMs in Finance Seminar";
pres.title  = "Does Financial Instruction Tuning Help LLMs?";

// ---- helpers ----
function imgIfExists(name) {
  const p = path.join(FIG_DIR, name);
  return fs.existsSync(p) ? p : null;
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide() {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  // Slim header bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.05, fill: { color: NAVY }, line: { color: NAVY },
  });
  // Footer
  s.addText("LLMs in Finance Seminar  ·  Implementation Presentation", {
    x: 0.4, y: 5.35, w: 9.2, h: 0.25, fontSize: 9, fontFace: FACE_B, color: MUTED,
  });
  return s;
}
function slideTitle(slide, text) {
  slide.addText(text, {
    x: 0.4, y: 0.25, w: 9.2, h: 0.6,
    fontSize: 26, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
  });
}
function slideSubtitle(slide, text) {
  slide.addText(text, {
    x: 0.4, y: 0.85, w: 9.2, h: 0.35,
    fontSize: 13, italic: true, fontFace: FACE_B, color: MUTED, margin: 0,
  });
}
function bullets(slide, items, opts = {}) {
  const lines = items.map((t, i) => ({
    text: t,
    options: { bullet: true, breakLine: i < items.length - 1 },
  }));
  slide.addText(lines, {
    x: opts.x || 0.5, y: opts.y || 1.5, w: opts.w || 9, h: opts.h || 3.5,
    fontSize: opts.fontSize || 14, fontFace: FACE_B, color: opts.color || INK,
    paraSpaceAfter: 6, valign: "top",
  });
}
function caption(slide, text, x = 0.4, y = 5.05, w = 9.2) {
  slide.addText(text, {
    x, y, w, h: 0.25, fontSize: 9, italic: true,
    fontFace: FACE_B, color: MUTED, margin: 0,
  });
}

// =========================================================================
// 1. Title slide
// =========================================================================
{
  const s = darkSlide();
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.6, w: 10, h: 0.05, fill: { color: ICE }, line: { color: ICE },
  });
  s.addText("Does Financial Instruction Tuning Help LLMs?", {
    x: 0.6, y: 1.6, w: 8.8, h: 1.2,
    fontSize: 38, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("Replicating the sentiment-classification slice of Open-FinLLMs (Huang et al., 2024)", {
    x: 0.6, y: 2.8, w: 8.8, h: 0.6,
    fontSize: 17, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
  s.addText("Implementation Presentation  ·  30 min", {
    x: 0.6, y: 3.5, w: 8.8, h: 0.4,
    fontSize: 13, fontFace: FACE_B, color: ICE, margin: 0,
  });
  s.addText("[Authors]   ·   LLMs in Finance Seminar", {
    x: 0.6, y: 4.8, w: 8.8, h: 0.4,
    fontSize: 12, fontFace: FACE_B, color: WHITE, margin: 0,
  });
}

// =========================================================================
// 2. Outline
// =========================================================================
{
  const s = lightSlide();
  slideTitle(s, "What we'll cover");
  const rows = [
    ["1", "Objective and setup", "5-7 min", "Research question, scope, reproducibility note"],
    ["2", "Method and experimental design", "8-10 min", "Datasets, models, prompts, matrix, defenses"],
    ["3", "Results and error analysis", "10-12 min", "Headline numbers, per-class behavior, concrete cases"],
    ["4", "Lessons learned", "3-5 min", "Reproducibility, prompt sensitivity, what we'd change"],
  ];
  rows.forEach((r, i) => {
    const y = 1.4 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.6, h: 0.7, fill: { color: NAVY }, line: { color: NAVY },
    });
    s.addText(r[0], {
      x: 0.5, y, w: 0.6, h: 0.7, fontSize: 26, bold: true,
      fontFace: FACE_H, color: ICE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(r[1], {
      x: 1.3, y: y + 0.05, w: 6.5, h: 0.35,
      fontSize: 16, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
    });
    s.addText(r[3], {
      x: 1.3, y: y + 0.42, w: 6.5, h: 0.3,
      fontSize: 11, fontFace: FACE_B, color: INK, margin: 0,
    });
    s.addText(r[2], {
      x: 8.0, y: y + 0.18, w: 1.6, h: 0.4,
      fontSize: 12, fontFace: FACE_B, color: MUTED, align: "right", margin: 0,
    });
  });
}

// =========================================================================
// SECTION 1 — Objective and setup
// =========================================================================
{
  const s = darkSlide();
  s.addText("1", {
    x: 0.6, y: 1.4, w: 1.4, h: 1.6, fontSize: 110, bold: true,
    fontFace: FACE_H, color: ICE, valign: "middle", margin: 0,
  });
  s.addText("Objective and setup", {
    x: 2.2, y: 2.0, w: 7.5, h: 1.0,
    fontSize: 36, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("What we set out to test, and why it matters.", {
    x: 2.2, y: 3.0, w: 7.5, h: 0.5,
    fontSize: 16, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
}

// 1.1 — Research question
{
  const s = lightSlide();
  slideTitle(s, "The question");
  slideSubtitle(s, "Open-FinLLMs (Huang et al., 2024) claims financial instruction tuning improves financial NLP.");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 9, h: 1.6, fill: { color: ICE }, line: { color: ICE },
  });
  s.addText("Does TheFinAI's financial instruction tuning beat the LLaMA-family base it was built on?", {
    x: 0.7, y: 1.7, w: 8.6, h: 0.7,
    fontSize: 22, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
  });
  s.addText("Specifically, on sentiment classification — and how does it compare to classical baselines (FinBERT, VADER) and other 7-8B general-purpose LLMs (Mistral, Qwen2.5)?",
    { x: 0.7, y: 2.4, w: 8.6, h: 0.7,
      fontSize: 13, italic: true, fontFace: FACE_B, color: INK, margin: 0 });

  // Reproduced / Adapted / Simplified — required by the seminar handout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.3, w: 9, h: 1.7, fill: { color: WHITE }, line: { color: NAVY, width: 1 },
  });
  const cellW = 3.0;
  const cells = [
    { hdr: "Reproduced", body: "FPB / FiQA-SA sentiment task; 4-bit inference of an 8B financial LLM vs. 7-8B general LLMs vs. classical baselines; F1-macro / accuracy reporting." },
    { hdr: "Adapted",    body: "Substituted plutus-8B-instruct (TheFinAI's 2025 successor) when FinLLaMA-instruct was unpublished; prompt templates A/B specifically chosen to measure sensitivity." },
    { hdr: "Simplified", body: "Inference only (no fine-tuning, no multimodal, no trading sim). 300-sample subsample per dataset for the LLM matrix to fit Modal T4 budget. Two prompts, two shot counts." },
  ];
  cells.forEach((c, i) => {
    const x = 0.5 + i * cellW + i * 0.05;
    s.addText(c.hdr, {
      x: x + 0.15, y: 3.4, w: cellW - 0.2, h: 0.35,
      fontSize: 13, bold: true, fontFace: FACE_H, color: ACCENT, margin: 0,
    });
    s.addText(c.body, {
      x: x + 0.15, y: 3.75, w: cellW - 0.2, h: 1.2,
      fontSize: 10.5, fontFace: FACE_B, color: INK, margin: 0,
    });
    if (i < cells.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + cellW + 0.025, y: 3.4, w: 0, h: 1.5,
        line: { color: ICE, width: 1 },
      });
    }
  });
}

// 1.2 — Reproducibility note
{
  const s = lightSlide();
  slideTitle(s, "Reproducibility note");
  slideSubtitle(s, "What changed since the paper, and what we did about it.");

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 9, h: 0.06, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  s.addText("The model in the paper, TheFinAI/FinLLaMA-instruct, has been unpublished by the authors.", {
    x: 0.5, y: 1.75, w: 9, h: 0.5,
    fontSize: 17, bold: true, fontFace: FACE_H, color: ACCENT, margin: 0,
  });

  bullets(s, [
    "We substitute TheFinAI/plutus-8B-instruct (Feb 2025) — the same group's current 8B financial-instruction-tuned model",
    "Same architecture (LLaMA-3 8B), same research direction, same goal: beat the base on financial NLP",
    "This itself is a finding: LLM benchmarks are fragile when the artifact under test can disappear",
    "We document the swap in our slides and report and re-run will be possible if FinLLaMA-instruct returns",
  ], { x: 0.5, y: 2.4, w: 9, h: 2.3 });
}

// =========================================================================
// SECTION 2 — Method and experimental design
// =========================================================================
{
  const s = darkSlide();
  s.addText("2", {
    x: 0.6, y: 1.4, w: 1.4, h: 1.6, fontSize: 110, bold: true,
    fontFace: FACE_H, color: ICE, valign: "middle", margin: 0,
  });
  s.addText("Method and experimental design", {
    x: 2.2, y: 2.0, w: 7.5, h: 1.0,
    fontSize: 32, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("Pipeline, datasets, models, prompts, defenses against leakage.", {
    x: 2.2, y: 3.0, w: 7.5, h: 0.5,
    fontSize: 16, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
}

// 2.1 — Pipeline
{
  const s = lightSlide();
  slideTitle(s, "Pipeline");
  slideSubtitle(s, "Single unified Sample schema; every downstream stage is dataset-agnostic.");

  const stages = ["data_loader", "prompts", "model runner", "parser", "evaluation"];
  const desc = [
    "FPB + FiQA → unified dicts",
    "Templates A/B + few-shot",
    "Modal T4 (LLMs) or CPU (baselines)",
    "Canonical label or null",
    "F1m / accuracy / coverage",
  ];
  const w = 1.7, gap = 0.15, startX = 0.4, y = 1.7;
  stages.forEach((stage, i) => {
    const x = startX + i * (w + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 1.0, fill: { color: ICE }, line: { color: NAVY, width: 1 },
    });
    s.addText(stage, {
      x, y: y + 0.1, w, h: 0.4, fontSize: 12, bold: true,
      fontFace: FACE_H, color: NAVY, align: "center", valign: "middle", margin: 0,
    });
    s.addText(desc[i], {
      x, y: y + 0.5, w, h: 0.45, fontSize: 9.5,
      fontFace: FACE_B, color: INK, align: "center", valign: "top", margin: 0,
    });
    if (i < stages.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + w, y: y + 0.5, w: gap, h: 0,
        line: { color: NAVY, width: 2, endArrowType: "triangle" },
      });
    }
  });

  bullets(s, [
    "Per-run directory: meta.json + predictions.jsonl + progress.json — checkpoint/resume built in",
    "Run ID schema: {model}__{dataset}__{template}__{shots}shot__seed{seed}",
    "Modal Volume caches HF weights — multi-GB download paid only once across the project",
    "Coverage tracked separately so parse failures aren't force-mapped to 'neutral'",
  ], { x: 0.5, y: 3.0, w: 9, h: 2.0, fontSize: 12 });
}

// 2.2 — Datasets
{
  const s = lightSlide();
  slideTitle(s, "Datasets");
  slideSubtitle(s, "Two financial-sentiment benchmarks with different difficulty profiles.");

  const tableData = [
    [
      { text: "Dataset", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left" } },
      { text: "Source", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left" } },
      { text: "Size", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left" } },
      { text: "Labels", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left" } },
      { text: "Note", options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left" } },
    ],
    [
      "Financial PhraseBank",
      "Malo et al. 2014",
      "690 test (sentences_75agree)",
      "neg / neu / pos (3-class)",
      "Annotator agreement ≥ 75% — clean test signal",
    ],
    [
      "FiQA-SA",
      "FiQA 2018",
      "1,173 test",
      "score ∈ [-1, 1] → 3-class with ±0.10 band",
      "Headlines + posts; noisier text, harder",
    ],
  ];
  s.addTable(tableData, {
    x: 0.4, y: 1.6, w: 9.2, colW: [1.6, 1.4, 1.9, 2.1, 2.2],
    fontSize: 11, fontFace: FACE_B, color: INK,
    border: { type: "solid", color: "D0D7DE", pt: 0.5 },
  });

  bullets(s, [
    "LLM matrix uses a 300-sample subsample per dataset (Modal time budget); baselines use full test sets",
    "Few-shot pool: only FPB train (FiQA has no train split) — preserves test purity",
    "Leakage caveat: both datasets predate every LLM here by 7-12 years and were almost certainly seen at pretraining time",
  ], { x: 0.4, y: 3.5, w: 9.2, h: 1.5, fontSize: 11.5 });
}

// 2.3 — Models
{
  const s = lightSlide();
  slideTitle(s, "Models");
  slideSubtitle(s, "Five models across three families: financial-tuned, general 7-8B LLM, and classical.");

  const t = [
    [
      { text: "Family", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "Model", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "Size / Type", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "Why included", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
    ],
    [
      { text: "Financial-tuned", options: { bold: true, color: ACCENT } },
      "TheFinAI/plutus-8B-instruct",
      "8B · LLaMA-3, 4-bit",
      "Focal model — successor to FinLLaMA-instruct",
    ],
    [
      "General-purpose LLM",
      "mistralai/Mistral-7B-Instruct-v0.3",
      "7B · 4-bit",
      "Open LLaMA-family-like comparator",
    ],
    [
      "General-purpose LLM",
      "Qwen/Qwen2.5-7B-Instruct",
      "7B · 4-bit",
      "Strong open instruct baseline",
    ],
    [
      "Classical specialised",
      "ProsusAI/finbert",
      "110M · BERT classifier",
      "Domain-specific small model — strong on FPB",
    ],
    [
      "Classical lexicon",
      "VADER",
      "Rule-based",
      "Cheap floor — what does no learning give you?",
    ],
  ];
  s.addTable(t, {
    x: 0.4, y: 1.6, w: 9.2, colW: [1.7, 2.7, 1.6, 3.2],
    fontSize: 11, fontFace: FACE_B, color: INK,
    border: { type: "solid", color: "D0D7DE", pt: 0.5 },
  });

  caption(s, "All LLMs run in 4-bit on a single Modal T4 (16GB VRAM). LLaMA-3.1-8B-Instruct skipped — gated, no access yet.");
}

// 2.4 — Prompts
{
  const s = lightSlide();
  slideTitle(s, "Two prompt templates (pre-registered)");
  slideSubtitle(s, "Two templates is enough to measure prompt sensitivity — we don't pick the best, we report both.");

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.5, w: 4.5, h: 3.5, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Template A — minimalist", {
    x: 0.6, y: 1.55, w: 4.2, h: 0.35,
    fontSize: 14, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
  });
  s.addText(
    "Classify the sentiment of the following financial text as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:",
    {
      x: 0.6, y: 1.95, w: 4.2, h: 2.9,
      fontSize: 12, fontFace: "Consolas", color: INK, margin: 0,
    }
  );

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.5, w: 4.5, h: 3.5, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  s.addText("Template B — structured + definitions", {
    x: 5.3, y: 1.55, w: 4.2, h: 0.35,
    fontSize: 14, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
  });
  s.addText(
    "You are a financial analyst…\n- Positive: favorable conditions, growth\n- Negative: unfavorable, losses, risks\n- Neutral: factual without clear implication\n\nText: {text}\n\nAnswer with one word only:",
    {
      x: 5.3, y: 1.95, w: 4.2, h: 2.9,
      fontSize: 11.5, fontFace: "Consolas", color: INK, margin: 0,
    }
  );
}

// 2.5 — Matrix and defenses
{
  const s = lightSlide();
  slideTitle(s, "Experimental matrix and defenses");
  slideSubtitle(s, "24 LLM runs (3 models × 2 datasets × 2 templates × {0, 3}-shot) + 6 baseline runs (FinBERT, FinBERT-tone, VADER × 2 datasets).");

  bullets(s, [
    "Deterministic decode (temperature=0, do_sample=False) — same input ⇒ same output",
    "Seed locked to 42 for every random draw: subsample, few-shot pick, baselines",
    "Few-shot pool drawn only from FPB train — never from test",
    "Two prompt templates pre-registered — both reported, neither selected post-hoc",
    "Coverage as first-class metric — parse failures excluded from F1, not force-mapped to 'neutral'",
    "Greedy parser with synonym map (bullish/bearish/optimistic…) — case-insensitive, first-match wins",
  ], { x: 0.5, y: 1.6, w: 9, h: 2.6, fontSize: 13 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.3, w: 9, h: 0.8, fill: { color: ICE }, line: { color: ICE },
  });
  s.addText(
    "Reported metrics: F1-macro (primary), accuracy, F1-weighted, per-class precision/recall, confusion matrix, parsing coverage, runtime.",
    { x: 0.7, y: 4.4, w: 8.6, h: 0.6,
      fontSize: 12, italic: true, fontFace: FACE_B, color: NAVY, margin: 0, valign: "middle" }
  );
}

// =========================================================================
// SECTION 3 — Results and error analysis
// =========================================================================
{
  const s = darkSlide();
  s.addText("3", {
    x: 0.6, y: 1.4, w: 1.4, h: 1.6, fontSize: 110, bold: true,
    fontFace: FACE_H, color: ICE, valign: "middle", margin: 0,
  });
  s.addText("Results and error analysis", {
    x: 2.2, y: 2.0, w: 7.5, h: 1.0,
    fontSize: 36, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("Headline numbers, where models break, and concrete examples.", {
    x: 2.2, y: 3.0, w: 7.5, h: 0.5,
    fontSize: 16, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
}

// 3.1 — Headline F1 chart
{
  const s = lightSlide();
  slideTitle(s, "Headline: F1-macro by model");
  slideSubtitle(s, "Best configuration per model on each dataset. Bigger = better.");
  const fig = imgIfExists("f1_comparison.png");
  if (fig) {
    s.addImage({ path: fig, x: 0.8, y: 1.4, w: 8.4, h: 3.5 });
  } else {
    s.addText("[ figure: f1_comparison.png — re-run scripts/make_figures.py after the matrix completes ]", {
      x: 0.8, y: 2.3, w: 8.4, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
  caption(s, "Source: results/summary/final_table.csv  ·  Re-render with scripts/make_figures.py");
}

// 3.2 — Headline numbers (real)
{
  const s = lightSlide();
  slideTitle(s, "Headline numbers — best config per model × dataset");
  slideSubtitle(s, "F1-macro is the primary metric (it is robust to the class imbalance in FPB and FiQA).");

  const t = [
    [
      { text: "Model", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "FPB · F1m", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "FPB · Acc", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "FiQA · F1m", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "FiQA · Acc", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
      { text: "Best config", options: { fill: { color: NAVY }, color: WHITE, bold: true } },
    ],
    [{ text: "FinBERT (specialised, 110M)", options: { bold: true, color: NAVY } },
     { text: "0.925", options: { fill: { color: "D6F5D6" } } },
     "0.935",
     "0.482",
     "0.498",
     "Best on FPB · loses on FiQA"],
    ["Qwen2.5-7B-Instruct",
     "0.832",
     "0.860",
     { text: "0.673", options: { fill: { color: "D6F5D6" } } },
     "0.717",
     "Best LLM on FiQA: tpl B 0-shot"],
    ["Mistral-7B-Instruct-v0.3",
     "0.890",
     "0.903",
     "0.599",
     "0.620",
     "Best LLM on FPB: tpl A 3-shot"],
    [{ text: "plutus-8B-instruct (focal)", options: { bold: true, color: ACCENT } },
     "0.829",
     "0.851",
     "0.597",
     "0.657",
     { text: "Mid-pack on both — see findings", options: { italic: true, color: ACCENT } }],
    ["VADER (lexicon)", "0.469", "0.554", "0.386", "0.423", "—"],
  ];
  s.addTable(t, {
    x: 0.4, y: 1.55, w: 9.2, colW: [2.5, 1.0, 1.0, 1.0, 1.0, 2.7],
    fontSize: 11, fontFace: FACE_B, color: INK,
    border: { type: "solid", color: "D0D7DE", pt: 0.5 },
  });
  caption(s, "Green-highlighted cells: per-dataset best.  ·  Source: results/summary/final_table.csv (dashboard at :8502).");
}

// 3.2b — Three findings
{
  const s = lightSlide();
  slideTitle(s, "Three findings the numbers tell us");
  slideSubtitle(s, "Read the table once — but interpret it three times.");

  const findings = [
    {
      h: "Financial instruction tuning did NOT pull ahead",
      b: "plutus-8B (focal financial-tuned model) scored 0.829 F1m on FPB and 0.597 on FiQA. Mistral-7B beat it on FPB (0.890) and Qwen2.5-7B beat it on FiQA (0.673). On the paper's central claim — that financial instruction tuning improves downstream NLP — our subset replication says: not clearly, on these two datasets.",
    },
    {
      h: "FinBERT dominates FPB; loses on FiQA — strong reverse on out-of-domain text",
      b: "On FPB the 110M specialised classifier (0.925 F1m) outperforms every 8B LLM. On FiQA it drops to 0.482 — Qwen2.5-7B beats it by 19 points. Domain-trained classifiers crush LLMs in-domain but generalise poorly to noisier text. A useful warning when picking models for production.",
    },
    {
      h: "Prompt sensitivity is comparable to model choice",
      b: "Mistral on FPB: Template A 0-shot 0.803 vs Template B 0-shot 0.689 — an 11-point gap from rewording alone. Qwen on FiQA: Template B 0-shot 0.673 vs Template A 0-shot 0.578 — 9 points. Plutus on FiQA: A 0-shot 0.597 vs B 0-shot 0.432 — 16 points. Single-prompt LLM benchmarks should not be trusted.",
    },
  ];
  findings.forEach((f, i) => {
    const y = 1.55 + i * 1.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9, h: 1.0, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.08, h: 1.0, fill: { color: NAVY }, line: { color: NAVY },
    });
    s.addText(f.h, {
      x: 0.7, y: y + 0.08, w: 8.7, h: 0.32,
      fontSize: 14, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
    });
    s.addText(f.b, {
      x: 0.7, y: y + 0.42, w: 8.7, h: 0.55,
      fontSize: 10.5, fontFace: FACE_B, color: INK, margin: 0,
    });
  });
}

// 3.3 — Per-class F1 — FPB
{
  const s = lightSlide();
  slideTitle(s, "Per-class F1 — Financial PhraseBank");
  slideSubtitle(s, "Class-level numbers expose hidden weaknesses an overall F1 can hide.");
  const fig = imgIfExists("per_class_f1_FPB.png");
  if (fig) {
    s.addImage({ path: fig, x: 1.5, y: 1.4, w: 7.0, h: 3.5 });
  } else {
    s.addText("[ figure: per_class_f1_FPB.png ]", {
      x: 1.5, y: 2.3, w: 7.0, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
}

// 3.4 — Per-class F1 — FiQA
{
  const s = lightSlide();
  slideTitle(s, "Per-class F1 — FiQA-SA");
  slideSubtitle(s, "Note any class-specific gap between the financial model and the general LLMs.");
  const fig = imgIfExists("per_class_f1_FiQA.png");
  if (fig) {
    s.addImage({ path: fig, x: 1.5, y: 1.4, w: 7.0, h: 3.5 });
  } else {
    s.addText("[ figure: per_class_f1_FiQA.png ]", {
      x: 1.5, y: 2.3, w: 7.0, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
}

// 3.5 — Confusion grid
{
  const s = lightSlide();
  slideTitle(s, "Confusion matrices — best config per model × dataset");
  slideSubtitle(s, "Rows: true label  ·  Columns: predicted label  ·  Look for systematic mistakes.");
  const fig = imgIfExists("confusion_grid.png");
  if (fig) {
    s.addImage({ path: fig, x: 0.5, y: 1.4, w: 9, h: 3.6 });
  } else {
    s.addText("[ figure: confusion_grid.png ]", {
      x: 0.5, y: 2.3, w: 9, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
}

// 3.6 — Few-shot effect
{
  const s = lightSlide();
  slideTitle(s, "Few-shot effect — does the financial model need fewer examples?");
  slideSubtitle(s, "ΔF1-macro when going from 0-shot to 3-shot. If the focal model gains less, its tuning already encodes the task.");
  const fig = imgIfExists("fewshot_effect.png");
  if (fig) {
    s.addImage({ path: fig, x: 1, y: 1.4, w: 8, h: 3.6 });
  } else {
    s.addText("[ figure: fewshot_effect.png ]", {
      x: 1, y: 2.3, w: 8, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
}

// 3.7 — Coverage
{
  const s = lightSlide();
  slideTitle(s, "Parsing coverage — does the model follow the format?");
  slideSubtitle(s, "Below ~95% is a flag. Coverage is reported separately so we don't paper over format failures.");
  const fig = imgIfExists("coverage_heatmap.png");
  if (fig) {
    s.addImage({ path: fig, x: 1, y: 1.4, w: 8, h: 3.6 });
  } else {
    s.addText("[ figure: coverage_heatmap.png ]", {
      x: 1, y: 2.3, w: 8, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
}

// 3.8 — Concrete examples (placeholders to fill from dashboard "Highlights")
{
  const s = lightSlide();
  slideTitle(s, "Concrete cases — pulled from the dashboard 'Highlights' tab");
  slideSubtitle(s, "Each card shows one sample, the gold label, and how each model classified it.");

  const cards = [
    { tag: "Focal wins, generals miss", body: "[paste sample text]\nGold: positive  ·  plutus-8B: positive  ·  Mistral: neutral  ·  Qwen2.5: neutral" },
    { tag: "Generals win, focal misses", body: "[paste sample text]\nGold: negative  ·  Mistral: negative  ·  plutus-8B: neutral" },
    { tag: "Prompt template flips a model", body: "[paste sample text]\nTemplate A → positive  ·  Template B → negative  (same model, same shots)" },
    { tag: "Everyone misses", body: "[paste sample text]\nGold: neutral  ·  All models predicted: positive (likely a calm-news-defaults-positive bias)" },
  ];
  const cardW = 4.4, cardH = 1.7, gap = 0.2;
  cards.forEach((c, i) => {
    const x = 0.5 + (i % 2) * (cardW + gap);
    const y = 1.5 + Math.floor(i / 2) * (cardH + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: cardW, h: cardH, fill: { color: WHITE }, line: { color: NAVY, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h: cardH, fill: { color: ACCENT }, line: { color: ACCENT },
    });
    s.addText(c.tag, {
      x: x + 0.2, y: y + 0.08, w: cardW - 0.3, h: 0.32,
      fontSize: 12, bold: true, fontFace: FACE_H, color: ACCENT, margin: 0,
    });
    s.addText(c.body, {
      x: x + 0.2, y: y + 0.42, w: cardW - 0.3, h: cardH - 0.5,
      fontSize: 11, fontFace: FACE_B, color: INK, margin: 0,
    });
  });
}

// 3.8b — Where the system works (per handout: "Where it works")
{
  const s = lightSlide();
  slideTitle(s, "Where the system works");
  slideSubtitle(s, "Strengths grounded in our numbers and confusion matrices.");

  bullets(s, [
    "FinBERT on FPB: clean labels + matching domain ⇒ 0.925 F1m and 0.91–0.95 per-class F1 — strongest result anywhere in the matrix.",
    "Mistral-7B on FPB with 3-shot Template A: 0.890 F1m, 0.903 accuracy — closes most of the FinBERT gap and shows that a general-purpose 7B can match a specialist when shown 3 in-context examples.",
    "Qwen2.5-7B on FiQA Template B 0-shot: 0.673 F1m on the noisy/conversational FiQA where FinBERT collapses to 0.482 — generalist beats specialist out-of-domain.",
    "Parsing coverage ≥97% everywhere; only plutus-8B dipped below 100% (2 of 24 runs) — no run was poisoned by 'bullish'/'bearish'-style outputs that would have force-mapped to neutral.",
    "VADER on FPB at 0.469 F1m provides a useful 'no-learning floor' — every other model is at least 17 points above it, validating the comparison.",
  ], { x: 0.5, y: 1.6, w: 9, h: 3.4, fontSize: 12.5 });
}

// 3.9 — Error categories (hand-tagged, 30 plutus-8B FPB misses; report §8.1)
{
  const s = lightSlide();
  slideTitle(s, "Error categories — focal model misses (hand-categorized)");
  slideSubtitle(s, "30 hand-tagged plutus-8B errors on FPB (results/summary/focal_error_sample.csv).");

  bullets(s, [
    "Missed positive cue — 15 (50%): mild/forward-looking positives read as neutral (neutral-bias), e.g. 'plans to expand internationally'",
    "Numerical reasoning — 8 (27%): needs comparing figures, e.g. 'loss narrowed 3.7→1.8mn' misread as negative",
    "Factual neutral misclassed — 6 (20%): a neutral fact (appointment, M&A, delisting) read as positive/negative",
    "Ambiguous / out-of-domain — 1 (3%): genuinely unclear sentence",
  ], { x: 0.5, y: 1.6, w: 9, h: 2.5, fontSize: 14 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.3, w: 9, h: 0.8, fill: { color: ICE }, line: { color: ICE },
  });
  s.addText(
    "Takeaway: half the misses are a conservative neutral-bias, not random noise — only 1 of 30 errors is genuinely ambiguous label noise.",
    { x: 0.7, y: 4.4, w: 8.6, h: 0.6,
      fontSize: 12, italic: true, fontFace: FACE_B, color: NAVY, margin: 0, valign: "middle" }
  );
}

// 3.10 — Dashboard (overview screenshot)
{
  const s = lightSlide();
  slideTitle(s, "Interactive analysis dashboard");
  slideSubtitle(s, "Live at localhost:8502 — every chart and table on these slides also updates as new runs finish.");
  const fig = imgIfExists("dashboard_overview.png");
  if (fig) {
    s.addImage({ path: fig, x: 0.5, y: 1.45, w: 9, h: 3.6, sizing: { type: "contain", w: 9, h: 3.6 } });
  } else {
    s.addText("[ figure: dashboard_overview.png ]", {
      x: 0.5, y: 2.3, w: 9, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
  caption(s, "Includes: research-question verdict card, F1 comparison, prompt-sensitivity & few-shot tables, per-class breakdown, confusion matrices.");
}

// 3.11 — Dashboard: per-example explorer
{
  const s = lightSlide();
  slideTitle(s, "Per-example breakdown — every model on the same sentence");
  slideSubtitle(s, "Pick any sample, see how each (model, template, shots) classified it; green = correct, red = wrong.");
  const fig = imgIfExists("dashboard_per_example.png");
  if (fig) {
    s.addImage({ path: fig, x: 0.5, y: 1.45, w: 9, h: 3.6, sizing: { type: "contain", w: 9, h: 3.6 } });
  } else {
    s.addText("[ figure: dashboard_per_example.png ]", {
      x: 0.5, y: 2.3, w: 9, h: 1, fontSize: 14, italic: true,
      fontFace: FACE_B, color: MUTED, align: "center", valign: "middle",
    });
  }
  caption(s, "The 'highlights' tab auto-curates cases where models disagree — useful for picking concrete examples for a presentation.");
}

// =========================================================================
// SECTION 4 — Lessons learned
// =========================================================================
{
  const s = darkSlide();
  s.addText("4", {
    x: 0.6, y: 1.4, w: 1.4, h: 1.6, fontSize: 110, bold: true,
    fontFace: FACE_H, color: ICE, valign: "middle", margin: 0,
  });
  s.addText("Lessons learned", {
    x: 2.2, y: 2.0, w: 7.5, h: 1.0,
    fontSize: 36, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("Reproducibility, prompt sensitivity, and what we'd do differently.", {
    x: 2.2, y: 3.0, w: 7.5, h: 0.5,
    fontSize: 16, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
}

// 4.1 — Lessons
{
  const s = lightSlide();
  slideTitle(s, "Five lessons");
  const lessons = [
    ["Reproducibility cost is real", "Notebook → script split, run-dir schema with checkpoint/resume, parser as a tested module, Modal Volume for cached weights — without these, a 24-run matrix isn't redoable."],
    ["Models can vanish", "TheFinAI/FinLLaMA-instruct was unpublished between paper and our run. Benchmark results are only as durable as the artifacts they reference."],
    ["Prompt sensitivity ≥ model choice (often)", "ΔF1 between Template A and B was up to 16 points on the same model and dataset — comparable to gaps between models. Single-prompt LLM benchmarks should not be trusted."],
    ["Coverage matters more than people report", "Force-mapping plutus-8B's unparsed outputs to 'neutral' would overstate accuracy by up to 1.25 points (FPB, Template A, 3-shot). Most papers don't report it."],
    ["Data leakage caveat", "FPB (2014) and FiQA (2018) almost certainly leaked into pretraining. Numbers here are an upper bound on real-world generalisation — a 2025 hold-out would be the honest test."],
  ];
  lessons.forEach((l, i) => {
    const y = 1.4 + i * 0.74;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9, h: 0.66, fill: { color: WHITE }, line: { color: ICE, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.08, h: 0.66, fill: { color: NAVY }, line: { color: NAVY },
    });
    s.addText(l[0], {
      x: 0.7, y: y + 0.04, w: 8.7, h: 0.28,
      fontSize: 13, bold: true, fontFace: FACE_H, color: NAVY, margin: 0,
    });
    s.addText(l[1], {
      x: 0.7, y: y + 0.32, w: 8.7, h: 0.32,
      fontSize: 10.5, fontFace: FACE_B, color: INK, margin: 0,
    });
  });
}

// 4.2 — What we'd do differently
{
  const s = lightSlide();
  slideTitle(s, "What we'd do differently");

  bullets(s, [
    "Build a 2025 financial-news held-out set to break leakage from FPB/FiQA pretraining",
    "Run three or four prompt templates instead of two — make sensitivity even more legible",
    "Add an LLM-as-judge sanity pass over a 100-sample slice — catch label noise",
    "Try few-shot examples drawn from the same dataset as the test (when available) — closer to real deployment",
    "Include a smaller financial model (e.g. finma-7b) and a non-financial 8B baseline of the same era for a tighter family comparison",
  ], { x: 0.5, y: 1.6, w: 9, h: 3.0, fontSize: 14 });
}

// 4.3 — Final takeaway
{
  const s = darkSlide();
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.6, w: 10, h: 0.05, fill: { color: ICE }, line: { color: ICE },
  });
  s.addText("Bottom line", {
    x: 0.6, y: 0.8, w: 8.8, h: 0.6,
    fontSize: 22, italic: true, fontFace: FACE_H, color: ICE, margin: 0,
  });
  s.addText(
    "Financial instruction tuning did NOT clearly help. plutus-8B scored 0.829 / 0.597 F1m on FPB / FiQA — beaten on FPB by Mistral-7B (0.890) and on FiQA by Qwen2.5-7B (0.673). FinBERT (110M, in-domain trained) crushed every 8B model on FPB but lost by 19 points on FiQA.",
    {
      x: 0.6, y: 1.6, w: 8.8, h: 1.8,
      fontSize: 18, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
    }
  );
  s.addText("Two takeaways for the audience to remember:", {
    x: 0.6, y: 3.7, w: 8.8, h: 0.4,
    fontSize: 13, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
  s.addText([
    { text: "1. Don't trust LLM benchmarks reported with a single prompt — sensitivity here was 9-16 F1 points.", options: { breakLine: true } },
    { text: "2. Specialised small models still beat 8B general LLMs in-domain — and the inverse on noisier text.", options: { breakLine: true } },
    { text: "3. Reproducibility is structural: artefacts disappear, tokenizer formats break, gates close. Build for it.", },
  ], {
    x: 0.6, y: 4.05, w: 8.8, h: 0.6,
    fontSize: 12, fontFace: FACE_B, color: WHITE, margin: 0,
  });
}

// 4.4 — Q&A
{
  const s = darkSlide();
  s.addText("Thank you", {
    x: 0.6, y: 1.6, w: 8.8, h: 1.0,
    fontSize: 56, bold: true, fontFace: FACE_H, color: WHITE, margin: 0,
  });
  s.addText("Questions?", {
    x: 0.6, y: 2.6, w: 8.8, h: 0.6,
    fontSize: 24, italic: true, fontFace: FACE_B, color: ICE, margin: 0,
  });
  s.addText("Code · github.com/shimi777/finllama-sentiment", {
    x: 0.6, y: 4.5, w: 8.8, h: 0.4,
    fontSize: 12, fontFace: "Consolas", color: ICE, margin: 0,
  });
  s.addText("Live dashboard · localhost:8502  (run dashboard/run.bat)", {
    x: 0.6, y: 4.85, w: 8.8, h: 0.4,
    fontSize: 12, fontFace: "Consolas", color: ICE, margin: 0,
  });
}

// =========================================================================
pres.writeFile({ fileName: OUT }).then((p) => console.log("wrote", p));
