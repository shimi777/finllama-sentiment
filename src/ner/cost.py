"""Cost tracking + budget guard for the NER API runners.

Why this exists: this is a student project. Every API run must be billable
to the penny and capped so a bug can't run up a surprise charge.

Pricing is per 1M tokens. Numbers below were current as of early 2026 — if
you re-run later, refresh from the providers' pricing pages.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


# Per-1M-token prices in USD (input, output).
# Keep this dict the *single source of truth* for cost math.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4.1-nano":            (0.10, 0.40),
    "gpt-4.1-mini":            (0.40, 1.60),
    "gpt-4o-mini":             (0.15, 0.60),
    "gpt-4o":                  (2.50, 10.00),
    # Anthropic
    "claude-haiku-4-5":        (1.00, 5.00),
    "claude-sonnet-4-6":       (3.00, 15.00),
    # Google
    "gemini-2.5-flash-lite":   (0.10, 0.40),
    "gemini-2.5-flash":        (0.30, 2.50),
    # Local — free
    "gliner-large":            (0.0, 0.0),
    "gliner-medium":           (0.0, 0.0),
    "gliner-small":            (0.0, 0.0),
    "nuner-zero":              (0.0, 0.0),
}


def usd_cost(model: str, n_in: int, n_out: int) -> float:
    """Return USD cost for n_in input + n_out output tokens of `model`."""
    p_in, p_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return (n_in / 1_000_000.0) * p_in + (n_out / 1_000_000.0) * p_out


def estimate_chars_to_tokens(s: str) -> int:
    """Cheap pre-flight estimate (~4 chars/token). Real cost uses provider usage."""
    return max(1, len(s) // 4)


@dataclass
class CostTracker:
    """Persistent per-process spend tracker with a hard cap.

    Loads cumulative spend from `results/_ner_spend.json` so the cap survives
    process restarts (you can't bypass it by re-running the script).
    """

    cap_usd: float = 2.0
    state_path: Path = field(default_factory=lambda: Path("results/_ner_spend.json"))
    _cumulative: float = 0.0
    _by_model: dict[str, float] = field(default_factory=dict)
    _by_run: dict[str, float] = field(default_factory=dict)
    _calls: int = 0
    _last_save: float = 0.0

    def __post_init__(self):
        self.state_path = Path(self.state_path)
        if self.state_path.exists():
            try:
                d = json.loads(self.state_path.read_text())
                self._cumulative = float(d.get("cumulative_usd", 0.0))
                self._by_model = dict(d.get("by_model", {}))
                self._by_run = dict(d.get("by_run", {}))
                self._calls = int(d.get("n_calls", 0))
            except (json.JSONDecodeError, OSError):
                pass

    @property
    def cumulative_usd(self) -> float:
        return self._cumulative

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self._cumulative)

    def can_afford(self, est_usd: float) -> bool:
        return (self._cumulative + est_usd) <= self.cap_usd

    def record(self, model: str, run_id: str, n_in: int, n_out: int) -> float:
        """Record a successful API call and return its USD cost."""
        cost = usd_cost(model, n_in, n_out)
        self._cumulative += cost
        self._by_model[model] = self._by_model.get(model, 0.0) + cost
        self._by_run[run_id] = self._by_run.get(run_id, 0.0) + cost
        self._calls += 1
        now = time.time()
        # Flush at most every 2s to avoid disk thrash.
        if now - self._last_save > 2.0:
            self.save()
            self._last_save = now
        return cost

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "cumulative_usd": round(self._cumulative, 6),
            "by_model": {k: round(v, 6) for k, v in self._by_model.items()},
            "by_run":   {k: round(v, 6) for k, v in self._by_run.items()},
            "n_calls":  self._calls,
            "cap_usd":  self.cap_usd,
            "updated_at": time.time(),
        }, indent=2))

    def summary(self) -> dict:
        return {
            "cumulative_usd": round(self._cumulative, 6),
            "remaining_usd":  round(self.remaining_usd, 6),
            "by_model":       {k: round(v, 6) for k, v in self._by_model.items()},
            "by_run":         {k: round(v, 6) for k, v in self._by_run.items()},
            "n_calls":        self._calls,
            "cap_usd":        self.cap_usd,
        }


class BudgetExceeded(RuntimeError):
    """Raised when a call would push cumulative spend over the cap."""


def have_key(model: str) -> bool:
    """Return True iff the env var for `model`'s provider is set."""
    if model.startswith("gpt-"):
        return bool(os.environ.get("OPENAI_API_KEY"))
    if model.startswith("claude-"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if model.startswith("gemini-"):
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return True  # local models — no key required


def is_local(model: str) -> bool:
    return model in ("gliner-large", "gliner-medium", "gliner-small", "nuner-zero")
