"""CPAP usage quality evaluator (XAI-style multi-card grading).

Each card returns: value, threshold (text), source, grade in {pass, warn, fail}
The overall grade follows an AND-gate: the worst card determines the overall grade.

References
----------
- AHI grading: AASM Manual (Berry et al., 2012)
- Mask leak: ResMed Clinical Guideline (95th pct < 24 L/min)
- Adherence: CMS Medicare 2008 (CAG-00093R2): >=4 h/night
- Pressure adequacy: ResMed AutoSet manual
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

GRADE_PASS = "pass"
GRADE_WARN = "warn"
GRADE_FAIL = "fail"
GRADE_NA = "na"

GRADE_RANK = {GRADE_PASS: 0, GRADE_WARN: 1, GRADE_FAIL: 2, GRADE_NA: -1}


def _grade_emoji(grade: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "—"}.get(grade, "—")


def _ahi_card(ahi_summary: pd.DataFrame) -> dict:
    if ahi_summary is None or ahi_summary.empty or "Metric" not in ahi_summary.columns:
        return _na_card("AHI", source="AASM 2012")
    row = ahi_summary[ahi_summary["Metric"] == "Total Events"]
    if row.empty:
        row = ahi_summary[ahi_summary["Metric"] == "AHI Score"]
    value = float(row["Value"].iloc[0]) if not row.empty else float("nan")
    if not np.isfinite(value):
        return _na_card("AHI", source="AASM 2012")
    if value < 5:
        grade = GRADE_PASS
    elif value < 15:
        grade = GRADE_WARN
    else:
        grade = GRADE_FAIL
    return {
        "name": "AHI",
        "value": round(value, 2),
        "unit": "/h",
        "threshold": "<5 정상 / 5–15 경증 / ≥15 중등증·중증",
        "source": "AASM 2012 (Berry et al.)",
        "grade": grade,
        "emoji": _grade_emoji(grade),
    }


def _leak_card(leakrate: pd.DataFrame) -> dict:
    if leakrate is None or leakrate.empty or "LeakRate_Lpm" not in leakrate.columns:
        return _na_card("Leak (95p)", source="ResMed Clinical Guideline")
    p95 = float(np.nanpercentile(leakrate["LeakRate_Lpm"], 95))
    grade = GRADE_PASS if p95 < 24 else GRADE_FAIL
    return {
        "name": "Leak (95p)",
        "value": round(p95, 1),
        "unit": "L/min",
        "threshold": "95p < 24 L/min",
        "source": "ResMed Clinical Guideline",
        "grade": grade,
        "emoji": _grade_emoji(grade),
    }


def _usage_card(duration_hours: Optional[float]) -> dict:
    if duration_hours is None:
        return _na_card("Usage", source="CMS Medicare 2008")
    grade = GRADE_PASS if duration_hours >= 4.0 else GRADE_FAIL
    h = int(duration_hours)
    m = int(round((duration_hours - h) * 60))
    return {
        "name": "Usage",
        "value": f"{h}h {m:02d}m",
        "unit": "",
        "threshold": "≥4 h/night",
        "source": "CMS Medicare 2008 (CAG-00093R2)",
        "grade": grade,
        "emoji": _grade_emoji(grade),
    }


def _pressure_card(pressure: pd.DataFrame, statistics: pd.DataFrame) -> dict:
    p95 = None
    if statistics is not None and not statistics.empty and "Metric" in statistics.columns:
        row = statistics[statistics["Metric"] == "Pressure"]
        if not row.empty and "95th_pct" in row.columns:
            p95 = float(row["95th_pct"].iloc[0])
    if p95 is None and pressure is not None and "Pressure_cmH2O" in pressure.columns and not pressure.empty:
        p95 = float(np.nanpercentile(pressure["Pressure_cmH2O"], 95))
    if p95 is None or not np.isfinite(p95):
        return _na_card("Pressure (95p)", source="ResMed AutoSet manual")
    # Heuristic: AutoSet default max 20 cmH2O. We flag warn if p95 is unusually high vs typical 5-12 range.
    grade = GRADE_WARN if p95 / 20.0 >= 0.9 else GRADE_PASS
    return {
        "name": "Pressure (95p)",
        "value": round(p95, 1),
        "unit": "cmH₂O",
        "threshold": "95p / max < 0.9",
        "source": "ResMed AutoSet manual",
        "grade": grade,
        "emoji": _grade_emoji(grade),
    }


def _na_card(name: str, source: str) -> dict:
    return {
        "name": name,
        "value": "—",
        "unit": "",
        "threshold": "—",
        "source": source,
        "grade": GRADE_NA,
        "emoji": _grade_emoji(GRADE_NA),
    }


def evaluate_quality(session: dict, duration_hours: Optional[float]) -> dict:
    cards = [
        _ahi_card(session.get("ahi_summary")),
        _leak_card(session.get("leakrate")),
        _usage_card(duration_hours),
        _pressure_card(session.get("pressure"), session.get("statistics")),
    ]
    measurable = [c for c in cards if c["grade"] != GRADE_NA]
    if not measurable:
        overall_grade = GRADE_NA
        reason = "측정 가능한 지표 없음"
    else:
        worst = max(measurable, key=lambda c: GRADE_RANK[c["grade"]])
        overall_grade = worst["grade"]
        reason = f"{worst['name']} 카드가 종합 등급을 결정 ({worst['emoji']})"
    return {
        "cards": cards,
        "overall": {
            "grade": overall_grade,
            "emoji": _grade_emoji(overall_grade),
            "reason": reason,
        },
    }


def compare_with_sleephq(
    detected: pd.DataFrame,
    ahi_summary: pd.DataFrame,
    duration_hours: Optional[float],
) -> dict:
    """Count-level agreement between detector output and SleepHQ-reported AHI counts.

    SleepHQ-style counts come from AHI_Summary.csv which stores per-hour rates.
    We multiply by session duration to get absolute counts for fair comparison.
    """
    if duration_hours is None or duration_hours <= 0:
        return {"detected_count": int(len(detected)) if detected is not None else 0,
                "sleephq_count": None, "agreement": None}

    sleephq_count = None
    if ahi_summary is not None and not ahi_summary.empty and "Metric" in ahi_summary.columns:
        row = ahi_summary[ahi_summary["Metric"] == "Total Events"]
        if not row.empty:
            rate = float(row["Value"].iloc[0])
            sleephq_count = int(round(rate * duration_hours))

    detected_count = int(len(detected)) if detected is not None else 0
    if sleephq_count is None or sleephq_count == 0:
        agreement = None
    else:
        agreement = round(min(detected_count, sleephq_count) / max(detected_count, sleephq_count), 3)
    return {
        "detected_count": detected_count,
        "sleephq_count": sleephq_count,
        "agreement": agreement,
    }
