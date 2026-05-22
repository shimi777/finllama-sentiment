"""Tests for src.ner.cost — pricing math + budget guard."""

import json

from src.ner.cost import CostTracker, usd_cost, have_key, is_local


def test_usd_cost_known_model():
    # gpt-4.1-nano: $0.10 in / $0.40 out per 1M
    # 1000 in + 500 out -> 1000/1e6 * 0.10 + 500/1e6 * 0.40 = 1e-4 + 2e-4 = 3e-4
    c = usd_cost("gpt-4.1-nano", 1000, 500)
    assert abs(c - 0.0003) < 1e-9


def test_usd_cost_unknown_zero():
    assert usd_cost("not-a-model", 1000, 500) == 0.0


def test_is_local():
    assert is_local("gliner-large")
    assert not is_local("gpt-4.1-nano")


def test_cost_tracker_record_and_save(tmp_path):
    state = tmp_path / "spend.json"
    t = CostTracker(cap_usd=1.0, state_path=state)
    cost = t.record("gpt-4.1-nano", run_id="run1", n_in=1_000_000, n_out=0)
    assert abs(cost - 0.10) < 1e-9
    assert abs(t.cumulative_usd - 0.10) < 1e-9
    t.save()
    d = json.loads(state.read_text())
    assert abs(d["cumulative_usd"] - 0.10) < 1e-9
    assert d["by_model"]["gpt-4.1-nano"] > 0


def test_cost_tracker_cap_blocks(tmp_path):
    t = CostTracker(cap_usd=0.05, state_path=tmp_path / "spend.json")
    # 1M in tokens on nano = $0.10, over the $0.05 cap
    assert not t.can_afford(0.10)
    # And small ones should pass
    assert t.can_afford(0.04)


def test_cost_tracker_loads_existing(tmp_path):
    state = tmp_path / "spend.json"
    state.write_text(json.dumps({
        "cumulative_usd": 0.5,
        "by_model": {"gpt-4o-mini": 0.5},
        "by_run": {"r1": 0.5},
        "n_calls": 10,
        "cap_usd": 1.0,
    }))
    t = CostTracker(cap_usd=1.0, state_path=state)
    assert abs(t.cumulative_usd - 0.5) < 1e-9
    assert t.remaining_usd == 0.5


def test_have_key_local_always_true():
    assert have_key("gliner-large")
    assert have_key("nuner-zero")


def test_have_key_no_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not have_key("gpt-4.1-nano")
    assert not have_key("claude-haiku-4-5")
    assert not have_key("gemini-2.5-flash-lite")
