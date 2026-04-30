"""Stage 4 — Respiratory pattern analysis.

Implements the algorithms defined in 03_신호처리_사양서 v0.4 §4.2~§4.8:

  §4.2 Flow signal post-processing
        - D01 envelope (2s RMS)
        - D02 baseline (rolling 60s 60th percentile)
        - D03 ratio (envelope / baseline)
  §4.3 Inspiration / Expiration separation (Hilbert + zero-crossing)
  §4.4 Tidal Volume (D04, per-breath inspiration integral)
  §4.5 Respiratory Rate (D05, zero-crossing + STFT ensemble)
  §4.6 Minute Ventilation (D06, TV × RR)
  §4.7 Inspiratory / Expiratory Time (D07, D08, D09)
  §4.8 Flow Limitation Index (D10, Schwartz 1994)

References
----------
* Berry RB et al. J Clin Sleep Med 2012;8(5):597-619
* Schwartz AR et al. J Appl Physiol 1991;70(1):425-31 (FL index)
* Cole RJ et al. Sleep 1992;15(5):461-9 (sleep latency)
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import signal as sps


class BreathCycle(TypedDict, total=False):
    """One breath cycle parameters (per §4.3~§4.7)."""
    start_idx: int           # zero-crossing into inspiration
    inspiration_end_idx: int # zero-crossing exiting inspiration
    end_idx: int             # zero-crossing into next inspiration (= cycle end)
    start_s: float
    end_s: float
    TV_ml: float             # D04
    IT_s: float              # D07
    ET_s: float              # D08
    IE_ratio: float          # D09 (IT / ET)
    peak_flow_lpm: float
    FL_index: float          # D10


# ---------------------------------------------------------------------------
# §4.2 Flow signal post-processing
# ---------------------------------------------------------------------------


def compute_envelope(flow_lpm: np.ndarray, fs: float, window_s: float = 2.0) -> np.ndarray:
    """D01 — Flow envelope as 2-second sliding RMS.

    Sliding-window RMS gives a smooth amplitude estimate that is robust to
    breath-by-breath variation. AASM uses 2 s as the standard window.
    """
    win = max(int(round(window_s * fs)), 1)
    sq = np.square(flow_lpm.astype(float))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    means = (csum[win:] - csum[:-win]) / win
    pad = np.full(win - 1, means[0] if means.size else 0.0)
    rms = np.sqrt(np.maximum(np.concatenate([pad, means]), 0.0))
    return rms


def compute_baseline(envelope: np.ndarray, fs: float, window_s: float = 60.0,
                     percentile: float = 60.0) -> np.ndarray:
    """D02 — Rolling baseline as 60th percentile over 60-second window.

    ResMed AutoSet style — uses high percentile rather than mean to be
    robust against multiple consecutive apneas which would pull mean low.
    """
    win = max(int(round(window_s * fs)), 1)
    s = pd.Series(envelope)
    baseline = (
        s.rolling(win, min_periods=max(win // 4, 1))
        .quantile(percentile / 100.0)
        .bfill()
        .ffill()
        .to_numpy()
    )
    return baseline


def compute_ratio(envelope: np.ndarray, baseline: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
    """D03 — Flow ratio = envelope / baseline (AASM event detection input)."""
    base_safe = np.where(baseline > eps, baseline, eps)
    return envelope / base_safe


# ---------------------------------------------------------------------------
# §4.3 Breath cycle extraction
# ---------------------------------------------------------------------------


def detect_zero_crossings(flow_lpm: np.ndarray) -> np.ndarray:
    """Return sample indices where flow crosses zero."""
    sign = np.sign(flow_lpm)
    return np.where(np.diff(sign) != 0)[0]


def extract_breath_cycles(flow_lpm: np.ndarray, fs: float,
                          min_cycle_s: float = 1.5,
                          max_cycle_s: float = 8.0) -> list[BreathCycle]:
    """Extract breath cycles from a (centered, leak-subtracted) flow signal.

    A cycle = (start_zc → inspiration_end_zc → end_zc) where the cycle
    duration is between min/max bounds. Inspiration is the positive-flow
    half cycle; expiration is the negative-flow half.
    """
    zc = detect_zero_crossings(flow_lpm)
    if len(zc) < 3:
        return []

    cycles: list[BreathCycle] = []
    # Iterate over consecutive pairs of (rising) zero-crossings as cycle bounds.
    # Use sign at the start to make sure each "cycle" starts on inspiration.
    for i in range(len(zc) - 2):
        start = int(zc[i])
        # First positive-going crossing only
        if flow_lpm[start + 1] <= 0:
            continue
        insp_end = int(zc[i + 1])
        end = int(zc[i + 2])

        cycle_s = (end - start) / fs
        if not (min_cycle_s <= cycle_s <= max_cycle_s):
            continue

        # Tidal volume: integrate inspiratory flow (L/min × s → L → mL)
        # ∫(L/min) dt(s) → L if we divide by 60
        insp_segment = flow_lpm[start:insp_end + 1]
        tv_ml = float(np.trapezoid(insp_segment, dx=1.0 / fs) / 60.0 * 1000.0)
        tv_ml = max(tv_ml, 0.0)

        it_s = (insp_end - start) / fs
        et_s = (end - insp_end) / fs
        ie_ratio = it_s / et_s if et_s > 0 else float("nan")

        peak = float(np.max(insp_segment)) if len(insp_segment) else 0.0
        fl_idx = compute_flow_limitation_index(insp_segment)

        cycles.append(BreathCycle(
            start_idx=start,
            inspiration_end_idx=insp_end,
            end_idx=end,
            start_s=start / fs,
            end_s=end / fs,
            TV_ml=tv_ml,
            IT_s=it_s,
            ET_s=et_s,
            IE_ratio=ie_ratio,
            peak_flow_lpm=peak,
            FL_index=fl_idx,
        ))
    return cycles


# ---------------------------------------------------------------------------
# §4.5 Respiratory Rate
# ---------------------------------------------------------------------------


def compute_respiratory_rate_zc(cycles: list[BreathCycle]) -> float:
    """RR estimate from mean cycle duration (zero-crossing method)."""
    if not cycles:
        return float("nan")
    durations = np.asarray([c["end_s"] - c["start_s"] for c in cycles])
    durations = durations[durations > 0]
    if len(durations) == 0:
        return float("nan")
    return float(60.0 / np.median(durations))


def compute_respiratory_rate_stft(flow_lpm: np.ndarray, fs: float,
                                  window_s: float = 60.0) -> float:
    """RR estimate from STFT peak in 0.1~0.5 Hz band (6~30 BPM)."""
    nperseg = min(int(window_s * fs), len(flow_lpm))
    if nperseg < int(8 * fs):
        return float("nan")
    f, pxx = sps.welch(flow_lpm, fs=fs, nperseg=nperseg)
    band = (f >= 0.1) & (f <= 0.5)
    if not np.any(band):
        return float("nan")
    f_band = f[band]
    p_band = pxx[band]
    f_peak = float(f_band[np.argmax(p_band)])
    return f_peak * 60.0


def compute_respiratory_rate(flow_lpm: np.ndarray, fs: float,
                             cycles: list[BreathCycle] | None = None) -> dict:
    """Ensemble RR — combines zero-crossing and STFT.

    Returns dict with rr_zc, rr_stft, rr (final), and confidence (0~1).
    """
    if cycles is None:
        cycles = extract_breath_cycles(flow_lpm, fs)
    rr_zc = compute_respiratory_rate_zc(cycles)
    rr_stft = compute_respiratory_rate_stft(flow_lpm, fs)

    # Ensemble: mean of valid estimates; if both valid and close, high confidence.
    valid = [r for r in (rr_zc, rr_stft) if np.isfinite(r)]
    if not valid:
        return {"rr_zc": rr_zc, "rr_stft": rr_stft, "rr": float("nan"),
                "confidence": 0.0}
    rr_final = float(np.mean(valid))

    if len(valid) == 2 and np.isfinite(rr_zc) and np.isfinite(rr_stft):
        agreement = 1.0 - min(abs(rr_zc - rr_stft) / max(rr_final, 1.0), 1.0)
        confidence = float(0.5 + 0.5 * agreement)
    else:
        confidence = 0.5

    return {"rr_zc": rr_zc, "rr_stft": rr_stft, "rr": rr_final,
            "confidence": confidence}


# ---------------------------------------------------------------------------
# §4.6 Minute Ventilation
# ---------------------------------------------------------------------------


def compute_minute_ventilation(cycles: list[BreathCycle], rr_bpm: float) -> float:
    """D06 — Minute Ventilation = mean(TV) × RR in L/min."""
    if not cycles or not np.isfinite(rr_bpm):
        return float("nan")
    mean_tv_ml = float(np.median([c["TV_ml"] for c in cycles]))
    return mean_tv_ml / 1000.0 * rr_bpm


# ---------------------------------------------------------------------------
# §4.8 Flow Limitation Index (Schwartz 1994)
# ---------------------------------------------------------------------------


def compute_flow_limitation_index(inspiratory_flow: np.ndarray,
                                  plateau_threshold_ratio: float = 0.7) -> float:
    """D10 — Flow limitation: how flat is the inspiratory plateau?

    FL = clip((plateau_ratio - 0.3) / 0.5, 0, 1)
    where plateau_ratio = fraction of samples ≥ 70% of peak.

    A perfectly rounded sinusoidal breath returns ~0; a heavily flattened
    inspiration (UARS / RERA precursor) returns toward 1.
    """
    if len(inspiratory_flow) == 0:
        return 0.0
    peak = float(np.max(inspiratory_flow))
    if peak <= 0.0:
        return 0.0
    plateau_threshold = plateau_threshold_ratio * peak
    plateau_samples = int(np.sum(inspiratory_flow >= plateau_threshold))
    plateau_ratio = plateau_samples / len(inspiratory_flow)
    fl = (plateau_ratio - 0.3) / 0.5
    return float(np.clip(fl, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §4.9 Leak estimation (mask-type-aware)
# ---------------------------------------------------------------------------


# Leak thresholds from 02_LCD_UI_사양서 v0.2 §6.4 D.9
MASK_LEAK_THRESHOLDS = {
    "nasal_pillow": (24.0, 36.0),
    "nasal":        (30.0, 42.0),
    "full_face":    (36.0, 48.0),
    "hybrid":       (30.0, 42.0),
}


def estimate_leak(total_flow_lpm: np.ndarray, patient_flow_lpm: np.ndarray,
                  ) -> np.ndarray:
    """Unintentional leak estimate.

    leak_unintentional = total_outflow - patient_inhalation - intentional_leak
    For PoC: returns total - patient (net leak including intentional).
    """
    return total_flow_lpm - patient_flow_lpm


def classify_leak_severity(leak_lpm: float, mask_type: str = "nasal_pillow") -> str:
    """Return one of: 'good' | 'caution' | 'high'."""
    warn, alarm = MASK_LEAK_THRESHOLDS.get(mask_type, (24.0, 36.0))
    if leak_lpm < warn:
        return "good"
    if leak_lpm < alarm:
        return "caution"
    return "high"


# ---------------------------------------------------------------------------
# Convenience: complete Stage 4 pipeline in one call
# ---------------------------------------------------------------------------


def stage4_analyze(flow_lpm: np.ndarray, fs: float) -> dict:
    """Run §4.2 ~ §4.8 in one call. Returns a dict bundling all derived signals.

    Output keys:
      'envelope', 'baseline', 'ratio'  (1 Hz time-aligned arrays at fs)
      'cycles'                         (list of BreathCycle dicts)
      'rr'                             (dict from compute_respiratory_rate)
      'mv_lpm'                         (float, minute ventilation)
    """
    envelope = compute_envelope(flow_lpm, fs)
    baseline = compute_baseline(envelope, fs)
    ratio = compute_ratio(envelope, baseline)
    cycles = extract_breath_cycles(flow_lpm, fs)
    rr = compute_respiratory_rate(flow_lpm, fs, cycles=cycles)
    mv = compute_minute_ventilation(cycles, rr["rr"])
    return {
        "envelope": envelope,
        "baseline": baseline,
        "ratio": ratio,
        "cycles": cycles,
        "rr": rr,
        "mv_lpm": mv,
    }
