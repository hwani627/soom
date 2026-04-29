"""Stage 5·6 — Respiratory event detection + AI/XAI classification.

Implements algorithms defined in 03_신호처리_사양서 v0.4 §5 and §6.

Stage 5 (§5) — AASM 2023 event detection:
  §5.1 Apnea — Flow envelope/baseline ratio < 0.10 for ≥ 10 s
  §5.2 Hypopnea — ratio in [0.10, 0.70) for ≥ 10 s + (SpO₂ desat OR arousal)
  §5.3 Cheyne-Stokes — 30~60 s crescendo-decrescendo periodicity
  §5.4 RERA — FL index + effort + arousal
  §5.5 Snore — high-frequency band energy > adaptive threshold

Stage 6 (§6) — 자사 4-layer differentiation:
  §6.1 L1 FOT 4 Hz — R/X impedance from forced oscillation
  §6.2 L2 Cardiogenic — 1~3 Hz STFT for OA vs CA classification
        (자사 1차 차별화: dual policy avoids patient arousal)
  §6.3 L3 Multi-frequency FOT (4 + 8 Hz) — pulmonary mechanics
  §6.4 L4 XAI raw export

Plus: ground-truth scoring against the synthetic generator's GroundTruth.

References
----------
* AASM Manual for the Scoring of Sleep v2.6 (2023)
* Berry RB et al. J Clin Sleep Med 2012;8(5):597-619
* Ayappa I et al. Chest 1999;116(3):660-6 (Cardiogenic, 자사 L2 근거)
* Farré R et al. Am J Respir Crit Care Med 1999;160(5):1810-5 (FOT)
* Morgenstern C et al. Sleep 2009 (FOT r ≈ 0.85 vs PSG)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import signal as sps

from .generator import GroundTruth, GroundTruthEvent

EventLabel = Literal["apnea", "OA", "CA", "MA", "hypopnea", "snore", "csr"]


@dataclass
class Event:
    """A detected respiratory event."""

    type: EventLabel
    start_s: float
    end_s: float
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_inclusive) for True-runs."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# §5.1 Apnea detection (AASM 2023)
# ---------------------------------------------------------------------------


def detect_apnea(envelope: np.ndarray, baseline: np.ndarray, fs: float,
                 threshold_ratio: float = 0.10,
                 min_duration_s: float = 10.0) -> list[Event]:
    """AASM 2023 Apnea Rule — Flow ≥ 90% reduction for ≥ 10 s."""
    base_safe = np.where(baseline > 1e-6, baseline, 1e-6)
    ratio = envelope / base_safe
    mask = ratio < threshold_ratio
    min_samples = max(int(min_duration_s * fs), 1)

    apneas: list[Event] = []
    for s, e in _runs(mask):
        if (e - s + 1) < min_samples:
            continue
        apneas.append(Event(
            type="apnea",  # subtype assigned later by classify_apnea_subtype
            start_s=s / fs,
            end_s=(e + 1) / fs,
            confidence=1.0,
            metadata={
                "ratio_min": float(np.min(ratio[s:e + 1])),
                "start_idx": s,
                "end_idx": e,
            },
        ))
    return apneas


# ---------------------------------------------------------------------------
# §5.2 Hypopnea detection
# ---------------------------------------------------------------------------


def detect_hypopnea(envelope: np.ndarray, baseline: np.ndarray, fs: float,
                    threshold_low: float = 0.10,
                    threshold_high: float = 0.70,
                    min_duration_s: float = 10.0,
                    spo2: np.ndarray | None = None) -> list[Event]:
    """AASM 2023 Hypopnea Rule 1A — 30~90% reduction ≥ 10 s.

    Without SpO2/arousal channel (single-sensor case), confidence is
    capped at 0.7 and metadata flags 'desat_verified'=False.
    """
    base_safe = np.where(baseline > 1e-6, baseline, 1e-6)
    ratio = envelope / base_safe
    mask = (ratio >= threshold_low) & (ratio < threshold_high)
    min_samples = max(int(min_duration_s * fs), 1)

    hypopneas: list[Event] = []
    for s, e in _runs(mask):
        if (e - s + 1) < min_samples:
            continue
        # SpO2 desat check (if available)
        desat_verified = False
        if spo2 is not None and len(spo2) >= e + 1:
            pre = spo2[max(0, s - int(30 * fs)):s] if s > 0 else spo2[:s]
            during = spo2[s:e + 1]
            if len(pre) > 0 and len(during) > 0:
                desat_verified = (float(np.max(pre) - np.min(during)) >= 3.0)

        confidence = 0.9 if desat_verified else 0.7
        hypopneas.append(Event(
            type="hypopnea",
            start_s=s / fs,
            end_s=(e + 1) / fs,
            confidence=confidence,
            metadata={
                "ratio_min": float(np.min(ratio[s:e + 1])),
                "desat_verified": desat_verified,
                "start_idx": s,
                "end_idx": e,
            },
        ))
    return hypopneas


# ---------------------------------------------------------------------------
# §6.2 Cardiogenic Oscillation (자사 1차 차별화 L2)
# ---------------------------------------------------------------------------


def detect_cardiogenic(cardiogenic_branch: np.ndarray, fs: float,
                       start_s: float, end_s: float,
                       baseline_window_s: float = 60.0) -> dict:
    """Detect cardiogenic oscillation (1~3 Hz) within a window.

    `cardiogenic_branch` is the BPF-filtered 0.8~3.0 Hz output from
    filters.bandpass_cardiogenic. We compare the RMS power inside the
    apnea segment to a baseline (the surrounding respiration-suppressed
    floor).

    Returns {'flag', 'power', 'baseline_power', 'confidence'}.
      flag = True  → cardiogenic present → CA (open airway)
      flag = False → cardiogenic absent  → OA (closed airway)
    """
    n = len(cardiogenic_branch)
    s_idx = max(int(start_s * fs), 0)
    e_idx = min(int(end_s * fs), n)
    if e_idx <= s_idx:
        return {"flag": False, "power": 0.0, "baseline_power": 0.0,
                "confidence": 0.0}

    seg_power = float(np.sqrt(np.mean(cardiogenic_branch[s_idx:e_idx] ** 2)))

    # Baseline: take RMS over a longer surrounding window, excluding the segment.
    bw = int(baseline_window_s * fs)
    bs = max(s_idx - bw, 0)
    be = min(e_idx + bw, n)
    surround = np.concatenate([
        cardiogenic_branch[bs:s_idx],
        cardiogenic_branch[e_idx:be],
    ])
    if len(surround) == 0:
        baseline_power = seg_power
    else:
        baseline_power = float(np.sqrt(np.mean(surround ** 2)))

    # Cardiogenic flag = segment power >= 50% of baseline (i.e. heart-rate
    # oscillation transmits through open airway)
    threshold = baseline_power * 0.5
    flag = bool(seg_power >= threshold) and seg_power > 1e-4

    # Confidence — distance from threshold normalized
    if baseline_power > 1e-9:
        z = abs(seg_power - threshold) / baseline_power
        confidence = float(np.clip(0.5 + 0.5 * np.tanh(2.0 * z), 0.0, 1.0))
    else:
        confidence = 0.5

    return {
        "flag": flag,
        "power": seg_power,
        "baseline_power": baseline_power,
        "confidence": confidence,
    }


def classify_apnea_subtype(apnea: Event, cardiogenic_branch: np.ndarray,
                           fs: float) -> Event:
    """Stage 6.2 — Apply L2 Cardiogenic to assign OA / CA / MA subtype.

    Returns a NEW Event with subtype-resolved type and updated metadata.
    """
    cardio = detect_cardiogenic(cardiogenic_branch, fs,
                                apnea.start_s, apnea.end_s)
    if cardio["flag"]:
        subtype = "CA"  # cardiogenic present → open airway → central
    else:
        subtype = "OA"  # cardiogenic absent → closed airway → obstructive

    new_meta = dict(apnea.metadata)
    new_meta.update({
        "cardiogenic_flag": cardio["flag"],
        "cardiogenic_power": cardio["power"],
        "cardiogenic_baseline_power": cardio["baseline_power"],
        "classification_method": "L2_cardiogenic",
    })

    return Event(
        type=subtype,  # type: ignore[arg-type]
        start_s=apnea.start_s,
        end_s=apnea.end_s,
        confidence=cardio["confidence"],
        metadata=new_meta,
    )


# ---------------------------------------------------------------------------
# §5.3 Cheyne-Stokes Respiration
# ---------------------------------------------------------------------------


def detect_csr(envelope: np.ndarray, fs: float,
               window_s: float = 600.0,
               csr_ratio_threshold: float = 0.4) -> list[Event]:
    """Detect Cheyne-Stokes pattern via Welch PSD ratio in 1/60~1/30 Hz band.

    Returns a list with at most 1 session-level CSR event if detected.
    """
    nperseg = min(int(window_s * fs), len(envelope))
    if nperseg < int(120 * fs):
        return []
    f, pxx = sps.welch(envelope, fs=fs, nperseg=nperseg)
    csr_band = (f >= 1.0 / 60.0) & (f <= 1.0 / 30.0)
    csr_power = float(np.sum(pxx[csr_band]))
    total_power = float(np.sum(pxx[f > 1e-3])) + 1e-12
    csr_ratio = csr_power / total_power
    if csr_ratio < csr_ratio_threshold:
        return []
    return [Event(
        type="csr",
        start_s=0.0,
        end_s=len(envelope) / fs,
        confidence=float(np.clip(csr_ratio, 0.0, 1.0)),
        metadata={"csr_ratio": csr_ratio},
    )]


# ---------------------------------------------------------------------------
# §5.5 Snore detection
# ---------------------------------------------------------------------------


def detect_snore(snore_branch: np.ndarray, fs: float,
                 window_s: float = 1.0,
                 threshold_multiplier: float = 5.0,
                 min_duration_s: float = 2.0) -> list[Event]:
    """Snore detection via adaptive-threshold high-frequency power."""
    win = max(int(window_s * fs), 1)
    sq = snore_branch ** 2
    power = np.convolve(sq, np.ones(win) / win, mode="same")

    median_power = float(np.median(power))
    threshold = median_power * threshold_multiplier
    if threshold < 1e-6:
        return []

    mask = power > threshold
    min_samples = max(int(min_duration_s * fs), 1)

    snores: list[Event] = []
    for s, e in _runs(mask):
        if (e - s + 1) < min_samples:
            continue
        snores.append(Event(
            type="snore",
            start_s=s / fs,
            end_s=(e + 1) / fs,
            confidence=float(np.clip(np.mean(power[s:e + 1]) / threshold,
                                     0.0, 1.0)),
            metadata={"peak_power": float(np.max(power[s:e + 1]))},
        ))
    return snores


# ---------------------------------------------------------------------------
# Session statistics (S07 AHI + counts)
# ---------------------------------------------------------------------------


def compute_session_stats(events: list[Event], duration_s: float) -> dict:
    """Compute session-level statistics including AHI."""
    if duration_s <= 0:
        return {"AHI": 0.0, "duration_h": 0.0, "counts": {}}

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.type] = counts.get(ev.type, 0) + 1

    apnea_total = (counts.get("OA", 0) + counts.get("CA", 0)
                   + counts.get("MA", 0) + counts.get("apnea", 0))
    hypopnea_total = counts.get("hypopnea", 0)
    duration_h = duration_s / 3600.0
    ahi = (apnea_total + hypopnea_total) / duration_h if duration_h > 0 else 0.0

    return {
        "AHI": float(ahi),
        "apnea_count": apnea_total,
        "hypopnea_count": hypopnea_total,
        "snore_count": counts.get("snore", 0),
        "csr_count": counts.get("csr", 0),
        "duration_h": float(duration_h),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Ground-truth scoring (validation against generator's GroundTruth)
# ---------------------------------------------------------------------------


_RESPIRATORY_TYPES = {"OA", "CA", "MA", "apnea", "hypopnea"}


def _events_overlap(detected: Event, truth: GroundTruthEvent,
                    tolerance_s: float) -> bool:
    """Two events match if their intervals overlap (with tolerance)."""
    return (detected.end_s + tolerance_s >= truth.start_s
            and detected.start_s - tolerance_s <= truth.end_s)


def score_events_vs_ground_truth(detected: list[Event], gt: GroundTruth,
                                 tolerance_s: float = 5.0,
                                 types: set[str] | None = None) -> dict:
    """Score detected vs ground-truth events for given types.

    `types` defaults to respiratory event labels (apnea/OA/CA/hypopnea).
    Returns sensitivity, specificity (n/a here as no negatives), PPV, TP/FP/FN.
    """
    if types is None:
        types = _RESPIRATORY_TYPES

    truth_relevant = [t for t in gt.events if t.type in types]
    detected_relevant = [d for d in detected if d.type in types]

    matched_truth = set()
    matched_det = set()

    for ti, truth in enumerate(truth_relevant):
        for di, det in enumerate(detected_relevant):
            if di in matched_det:
                continue
            if _events_overlap(det, truth, tolerance_s):
                # If truth is OA/CA, require detected matches subtype OR the
                # detected is generic 'apnea' (still counts as TP at apnea level).
                t_lbl = truth.type
                d_lbl = det.type
                t_is_apnea = t_lbl in {"OA", "CA", "MA"}
                d_is_apnea = d_lbl in {"OA", "CA", "MA", "apnea"}
                same_class = (t_lbl == d_lbl
                              or (t_is_apnea and d_is_apnea)
                              or (t_lbl == "hypopnea" and d_lbl == "hypopnea"))
                if same_class:
                    matched_truth.add(ti)
                    matched_det.add(di)
                    break

    tp = len(matched_truth)
    fn = len(truth_relevant) - tp
    fp = len(detected_relevant) - len(matched_det)

    sensitivity = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "sensitivity": float(sensitivity),
        "PPV": float(ppv),
        "n_truth": len(truth_relevant),
        "n_detected": len(detected_relevant),
    }


def score_subtype_classification(detected: list[Event], gt: GroundTruth,
                                 tolerance_s: float = 5.0) -> dict:
    """Among detected apneas that overlap a ground-truth apnea, what
    fraction got the OA/CA subtype right?"""
    truth_apneas = [t for t in gt.events if t.type in {"OA", "CA"}]
    det_apneas = [d for d in detected if d.type in {"OA", "CA"}]

    correct = 0
    pairs = 0
    for truth in truth_apneas:
        for det in det_apneas:
            if _events_overlap(det, truth, tolerance_s):
                pairs += 1
                if det.type == truth.type:
                    correct += 1
                break

    accuracy = correct / pairs if pairs > 0 else float("nan")
    return {"pairs": pairs, "correct": correct, "accuracy": float(accuracy)}
