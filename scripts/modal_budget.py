"""Tracks cumulative Modal GPU-seconds used by this project. Hard-stops past a budget.

State lives in `results/_modal_spend.json` (gitignored). Updated after each run.

Pricing assumptions (T4): $0.000164/sec = $0.59/hour. Update RATES if you switch GPU.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEND_FILE = os.path.join(ROOT, "results", "_modal_spend.json")

# $/sec
RATES = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40": 0.000583,
}


def _load() -> dict:
    if not os.path.exists(SPEND_FILE):
        return {"runs": [], "total_seconds": 0.0, "total_usd": 0.0}
    with open(SPEND_FILE) as f:
        return json.load(f)


def _save(state: dict) -> None:
    os.makedirs(os.path.dirname(SPEND_FILE), exist_ok=True)
    with open(SPEND_FILE, "w") as f:
        json.dump(state, f, indent=2)


def remaining_usd(cap_usd: float) -> float:
    return max(0.0, cap_usd - _load()["total_usd"])


def can_afford(estimated_seconds: float, gpu: str, cap_usd: float) -> bool:
    state = _load()
    cost = estimated_seconds * RATES[gpu]
    return state["total_usd"] + cost <= cap_usd


def record(label: str, gpu: str, seconds: float) -> dict:
    state = _load()
    rate = RATES[gpu]
    cost = seconds * rate
    state["runs"].append({
        "label": label,
        "gpu": gpu,
        "seconds": round(seconds, 2),
        "usd": round(cost, 4),
        "at": datetime.now(timezone.utc).isoformat(),
    })
    state["total_seconds"] = round(state["total_seconds"] + seconds, 2)
    state["total_usd"] = round(state["total_usd"] + cost, 4)
    _save(state)
    return state


def summary() -> str:
    s = _load()
    return f"Modal spend: ${s['total_usd']:.4f} ({s['total_seconds']:.0f}s) over {len(s['runs'])} runs"


if __name__ == "__main__":
    print(summary())
