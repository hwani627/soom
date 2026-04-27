"""Evaluator tests with synthetic AHI/leak inputs."""
from __future__ import annotations

import pandas as pd

from src import evaluator


def _ahi_df(total_events_per_hour: float) -> pd.DataFrame:
    return pd.DataFrame({"Metric": ["Total Events"], "Value": [total_events_per_hour]})


def test_ahi_grading_thresholds():
    s_pass = {"ahi_summary": _ahi_df(3.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [10] * 100})}
    s_warn = {"ahi_summary": _ahi_df(8.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [10] * 100})}
    s_fail = {"ahi_summary": _ahi_df(20.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [10] * 100})}

    assert evaluator.evaluate_quality(s_pass, 8.0)["cards"][0]["grade"] == "pass"
    assert evaluator.evaluate_quality(s_warn, 8.0)["cards"][0]["grade"] == "warn"
    assert evaluator.evaluate_quality(s_fail, 8.0)["cards"][0]["grade"] == "fail"


def test_leak_card_p95():
    s = {"ahi_summary": _ahi_df(3.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [5] * 95 + [50] * 5})}
    cards = {c["name"]: c for c in evaluator.evaluate_quality(s, 8.0)["cards"]}
    # 95th percentile of [5*95 + 50*5] is just at the boundary; should still be < 50 → check pass when below 24
    s_low = {"ahi_summary": _ahi_df(3.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [5] * 100})}
    assert evaluator.evaluate_quality(s_low, 8.0)["cards"][1]["grade"] == "pass"
    s_high = {"ahi_summary": _ahi_df(3.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [50] * 100})}
    assert evaluator.evaluate_quality(s_high, 8.0)["cards"][1]["grade"] == "fail"


def test_usage_card_threshold():
    s = {"ahi_summary": _ahi_df(3.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [10] * 100})}
    pass_card = evaluator.evaluate_quality(s, 5.0)["cards"][2]
    fail_card = evaluator.evaluate_quality(s, 3.0)["cards"][2]
    assert pass_card["grade"] == "pass"
    assert fail_card["grade"] == "fail"


def test_overall_is_worst_card():
    s_fail_ahi = {"ahi_summary": _ahi_df(20.0), "leakrate": pd.DataFrame({"LeakRate_Lpm": [10] * 100})}
    out = evaluator.evaluate_quality(s_fail_ahi, 8.0)
    assert out["overall"]["grade"] == "fail"
