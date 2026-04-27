"""CPAP usage quality evaluator.

Cards (each with measured value + threshold + academic source + grade):
- AHI:   events / total recording hours          (AASM 2012)
- Leak:  LeakRate 95th percentile                (ResMed Clinical Guideline)
- Usage: total recording duration                 (CMS Medicare 2008)
- Pressure: Pressure 95th percentile             (ResMed AutoSet)
- ODI3%: oxygen desaturations >=3% per hour     (AASM 2012)
- T90:   time with SpO2 below 90%               (Punjabi 2009)
- Lowest SpO2 / Mean SpO2                        (PSG standard)

Overall grade follows an AND-gate: the worst card determines the overall grade.
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

APNEA_HYPOPNEA_TYPES = {"CA", "OA", "MA", "H"}  # Hypopnea + all Apnea variants


def _emoji(grade: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "—"}.get(grade, "—")


def _na_card(name: str, source: str, threshold: str = "—") -> dict:
    return {"name": name, "value": "—", "unit": "", "threshold": threshold,
            "source": source, "grade": GRADE_NA, "emoji": _emoji(GRADE_NA)}


def _ahi_card(events: Optional[pd.DataFrame], duration_hours: Optional[float]) -> dict:
    if events is None or events.empty or duration_hours is None or duration_hours <= 0:
        return _na_card("AHI (무호흡-저호흡 지수)", "AASM 2012", "<5 / 5–15 / ≥15")
    n = events[events["event_type"].isin(APNEA_HYPOPNEA_TYPES)].shape[0]
    value = n / duration_hours
    if value < 5:    grade = GRADE_PASS
    elif value < 15: grade = GRADE_WARN
    else:            grade = GRADE_FAIL
    return {"name": "AHI (무호흡-저호흡 지수)", "value": round(value, 2), "unit": "/h",
            "threshold": "<5 정상 / 5–15 경증 / ≥15 중등증·중증",
            "source": "AASM 2012 (Berry et al.)",
            "grade": grade, "emoji": _emoji(grade)}


def _leak_card(leakrate: Optional[pd.DataFrame]) -> dict:
    if leakrate is None or leakrate.empty or "LeakRate_Lpm" not in leakrate.columns:
        return _na_card("Leak 95p (누설량 95백분위)", "ResMed Clinical Guideline", "<24 L/min")
    p95 = float(np.nanpercentile(leakrate["LeakRate_Lpm"], 95))
    grade = GRADE_PASS if p95 < 24 else GRADE_FAIL
    return {"name": "Leak 95p (누설량 95백분위)", "value": round(p95, 1), "unit": "L/min",
            "threshold": "95p < 24 L/min", "source": "ResMed Clinical Guideline",
            "grade": grade, "emoji": _emoji(grade)}


def _usage_card(duration_hours: Optional[float]) -> dict:
    if duration_hours is None:
        return _na_card("Usage (사용 시간)", "CMS Medicare 2008", "≥4 h/night")
    grade = GRADE_PASS if duration_hours >= 4.0 else GRADE_FAIL
    h = int(duration_hours)
    m = int(round((duration_hours - h) * 60))
    return {"name": "Usage (사용 시간)", "value": f"{h}h {m:02d}m", "unit": "",
            "threshold": "≥4 h/night", "source": "CMS Medicare 2008 (CAG-00093R2)",
            "grade": grade, "emoji": _emoji(grade)}


def _pressure_card(pressure: Optional[pd.DataFrame]) -> dict:
    if pressure is None or pressure.empty or "Pressure_cmH2O" not in pressure.columns:
        return _na_card("Pressure 95p (압력 95백분위)", "ResMed AutoSet", "95p / max < 0.9")
    p95 = float(np.nanpercentile(pressure["Pressure_cmH2O"], 95))
    grade = GRADE_WARN if p95 / 20.0 >= 0.9 else GRADE_PASS
    return {"name": "Pressure 95p (압력 95백분위)", "value": round(p95, 1), "unit": "cmH₂O",
            "threshold": "95p < 18 cmH₂O (default max 20)",
            "source": "ResMed AutoSet manual",
            "grade": grade, "emoji": _emoji(grade)}


def _odi_card(spo2: Optional[pd.DataFrame], duration_hours: Optional[float]) -> dict:
    """Oxygen Desaturation Index (≥3%) per hour."""
    if (spo2 is None or spo2.empty or duration_hours is None or duration_hours <= 0
            or "SpO2_pct" not in spo2.columns):
        return _na_card("ODI 3% (산소 탈포화 지수)", "AASM 2012", "<5 정상 / 5–15 경증 / ≥15 중등 이상")

    s = spo2.sort_values("Timestamp_ET")
    vals = s["SpO2_pct"].to_numpy()
    if vals.size < 2:
        return _na_card("ODI 3% (산소 탈포화 지수)", "AASM 2012")

    # Detect drops of >=3 percentage points within ~120-second windows from a local baseline.
    desat_count = 0
    baseline = vals[0]
    for v in vals[1:]:
        if v <= baseline - 3:
            desat_count += 1
            baseline = v
        elif v > baseline:
            baseline = v

    odi = desat_count / duration_hours
    if odi < 5:    grade = GRADE_PASS
    elif odi < 15: grade = GRADE_WARN
    else:          grade = GRADE_FAIL
    return {"name": "ODI 3% (산소 탈포화 지수)", "value": round(odi, 2), "unit": "/h",
            "threshold": "<5 / 5–15 / ≥15", "source": "AASM 2012",
            "grade": grade, "emoji": _emoji(grade)}


def _t90_card(spo2: Optional[pd.DataFrame]) -> dict:
    """Time spent below 90% SpO2 (cumulative minutes + percentage of recording)."""
    if spo2 is None or spo2.empty or "SpO2_pct" not in spo2.columns:
        return _na_card("T90 (SpO₂<90% 누적 시간)", "Punjabi NM 2009", "<5% 정상")
    s = spo2.sort_values("Timestamp_ET").copy()
    s["dt_sec"] = s["Timestamp_ET"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=10)
    below = s[s["SpO2_pct"] < 90]
    sec_below = float(below["dt_sec"].sum())
    sec_total = float(s["dt_sec"].sum())
    pct = (sec_below / sec_total * 100) if sec_total > 0 else 0.0
    if pct < 1:    grade = GRADE_PASS
    elif pct < 5:  grade = GRADE_WARN
    else:          grade = GRADE_FAIL
    return {"name": "T90 (SpO₂<90% 누적 시간)", "value": f"{sec_below/60:.1f} min ({pct:.1f}%)", "unit": "",
            "threshold": "<1% 정상 / <5% 경계 / ≥5% 위험",
            "source": "Punjabi NM, Sleep Med 2009",
            "grade": grade, "emoji": _emoji(grade)}


def _lowest_spo2_card(spo2: Optional[pd.DataFrame]) -> dict:
    if spo2 is None or spo2.empty or "SpO2_pct" not in spo2.columns:
        return _na_card("Lowest SpO₂ (최저 산소포화도)", "PSG standard", "≥90% 정상")
    lo = float(spo2["SpO2_pct"].min())
    if lo >= 90:    grade = GRADE_PASS
    elif lo >= 85:  grade = GRADE_WARN
    else:           grade = GRADE_FAIL
    return {"name": "Lowest SpO₂ (최저 산소포화도)", "value": round(lo, 1), "unit": "%",
            "threshold": "≥90% 정상 / 85–89% 경계 / <85% 저산소",
            "source": "PSG standard",
            "grade": grade, "emoji": _emoji(grade)}


def evaluate_quality(session: dict, duration_hours: Optional[float]) -> dict:
    cards = [
        _ahi_card(session.get("events"), duration_hours),
        _leak_card(session.get("leakrate")),
        _usage_card(duration_hours),
        _pressure_card(session.get("pressure")),
        _odi_card(session.get("spo2"), duration_hours),
        _t90_card(session.get("spo2")),
        _lowest_spo2_card(session.get("spo2")),
    ]
    measurable = [c for c in cards if c["grade"] != GRADE_NA]
    if not measurable:
        return {"cards": cards, "overall": {"grade": GRADE_NA, "emoji": _emoji(GRADE_NA),
                                            "reason": "측정 가능한 지표 없음"}}
    worst = max(measurable, key=lambda c: GRADE_RANK[c["grade"]])
    return {"cards": cards, "overall": {
        "grade": worst["grade"], "emoji": _emoji(worst["grade"]),
        "reason": f"{worst['name']} 카드가 종합 등급을 결정 ({worst['emoji']})"}}


def compare_with_sleephq(detected: pd.DataFrame, events: Optional[pd.DataFrame]) -> dict:
    """Timestamp-level Precision/Recall/F1 between detector output and SleepHQ events.

    A detected event is a TP if it overlaps any SleepHQ apnea/hypopnea event.
    """
    sleephq = events[events["event_type"].isin(APNEA_HYPOPNEA_TYPES)] if events is not None else None
    if sleephq is None or sleephq.empty:
        return {"detected_count": int(len(detected)) if detected is not None else 0,
                "sleephq_count": 0, "precision": None, "recall": None, "f1": None}

    if detected is None or detected.empty:
        return {"detected_count": 0, "sleephq_count": int(len(sleephq)),
                "precision": None, "recall": 0.0, "f1": 0.0}

    det_intervals = list(zip(detected["start_ts"].tolist(), detected["end_ts"].tolist()))
    ref_intervals = list(zip(sleephq["start_ts"].tolist(), sleephq["end_ts"].tolist()))

    def _overlaps(a_s, a_e, b_s, b_e) -> bool:
        return a_s <= b_e and b_s <= a_e

    tp = sum(1 for d_s, d_e in det_intervals if any(_overlaps(d_s, d_e, r_s, r_e)
                                                     for r_s, r_e in ref_intervals))
    matched_ref = sum(1 for r_s, r_e in ref_intervals if any(_overlaps(d_s, d_e, r_s, r_e)
                                                              for d_s, d_e in det_intervals))
    precision = tp / len(det_intervals) if det_intervals else None
    recall    = matched_ref / len(ref_intervals) if ref_intervals else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else 0.0)
    return {
        "detected_count": len(det_intervals),
        "sleephq_count":  len(ref_intervals),
        "precision": round(precision, 3) if precision is not None else None,
        "recall":    round(recall, 3)    if recall    is not None else None,
        "f1":        round(f1, 3),
    }
