"""Unified API runner for OpenAI / Anthropic / Gemini NER.

One class, three providers, dispatched by model name prefix. The choice was
to keep all three in one file (vs three modules) because the call surface is
nearly identical — prompt -> JSON -> usage tuple — and the differences are
literally three SDK import paths.

Cost contract: every successful call goes through `CostTracker.record`. The
runner refuses to dispatch a call if the pre-flight estimate would push us
over the cap (raises `BudgetExceeded`).
"""

from __future__ import annotations

import os
import time
from typing import Iterable

from src.utils import get_logger
from src.ner.cost import (
    BudgetExceeded, CostTracker, MODEL_PRICING,
    estimate_chars_to_tokens, have_key, usd_cost,
)
from src.ner.data_loader import NerSample
from src.ner.parser import parse_json_to_bio
from src.ner.prompts import SYSTEM_PROMPT, build_prompt

logger = get_logger(__name__)


def _provider_of(model: str) -> str:
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    raise ValueError(f"Unknown provider for model '{model}'")


class APINERRunner:
    """One instance per (model, template, n_shots) tuple."""

    def __init__(
        self,
        model: str,
        template: str = "A",
        n_shots: int = 0,
        cost_tracker: CostTracker | None = None,
        max_output_tokens: int = 200,
    ):
        if model not in MODEL_PRICING:
            raise ValueError(f"Unknown model {model}; not in MODEL_PRICING")
        if not have_key(model):
            raise RuntimeError(
                f"No API key in env for {model}. "
                "Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY."
            )
        self.model = model
        self.template = template
        self.n_shots = n_shots
        self.tracker = cost_tracker or CostTracker()
        self.max_output_tokens = max_output_tokens
        self.provider = _provider_of(model)
        self._client = None  # lazy

    # -------- provider clients (lazy) --------

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _anthropic(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _google(self):
        if self._client is None:
            from google import genai
            key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=key)
        return self._client

    # -------- single call --------

    def _call_openai(self, prompt: str) -> tuple[str, int, int]:
        client = self._openai()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_output_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, int(getattr(usage, "prompt_tokens", 0)), int(getattr(usage, "completion_tokens", 0))

    def _call_anthropic(self, prompt: str) -> tuple[str, int, int]:
        client = self._anthropic()
        resp = client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        # Concatenate text blocks defensively.
        parts = []
        for blk in resp.content:
            t = getattr(blk, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts)
        usage = resp.usage
        return text, int(getattr(usage, "input_tokens", 0)), int(getattr(usage, "output_tokens", 0))

    def _call_google(self, prompt: str) -> tuple[str, int, int]:
        client = self._google()
        full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        resp = client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        text = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        n_in = int(getattr(meta, "prompt_token_count", 0)) if meta else 0
        n_out = int(getattr(meta, "candidates_token_count", 0)) if meta else 0
        return text, n_in, n_out

    def predict_one(self, sample: NerSample, run_id: str) -> dict:
        prompt = build_prompt(self.template, sample["text"], n_shots=self.n_shots)

        # Pre-flight budget check (best-effort using char heuristic).
        est_in = estimate_chars_to_tokens(SYSTEM_PROMPT + "\n" + prompt)
        est_out = self.max_output_tokens
        est_cost = usd_cost(self.model, est_in, est_out)
        if not self.tracker.can_afford(est_cost):
            raise BudgetExceeded(
                f"Skipping {sample['id']} on {self.model}: "
                f"would exceed cap (cum=${self.tracker.cumulative_usd:.4f}, "
                f"est=${est_cost:.4f}, cap=${self.tracker.cap_usd:.4f})"
            )

        t0 = time.perf_counter()
        try:
            if self.provider == "openai":
                raw, n_in, n_out = self._call_openai(prompt)
            elif self.provider == "anthropic":
                raw, n_in, n_out = self._call_anthropic(prompt)
            elif self.provider == "google":
                raw, n_in, n_out = self._call_google(prompt)
            else:
                raise RuntimeError(f"Unknown provider {self.provider}")
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning("%s call failed on %s: %s", self.model, sample["id"], e)
            return {
                "id": sample["id"], "pred_tags": None, "pred_entities": [],
                "raw_output": f"<<api-error: {type(e).__name__}: {e}>>",
                "parse_ok": False,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            }
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cost = self.tracker.record(self.model, run_id, n_in, n_out)
        bio, pred_entities = parse_json_to_bio(raw, sample["tokens"])
        return {
            "id": sample["id"],
            "pred_tags": bio,
            "pred_entities": pred_entities,
            "raw_output": raw,
            "parse_ok": bio is not None,
            "latency_ms": latency_ms,
            "input_tokens": n_in,
            "output_tokens": n_out,
            "cost_usd": cost,
        }

    def predict_many(self, samples: Iterable[NerSample], run_id: str) -> Iterable[dict]:
        for s in samples:
            yield self.predict_one(s, run_id)
