"""Stage 2 — Raw pressure signal refinement and 4-branch BPF.

Implements the digital filter chain defined in 03_신호처리_사양서 v0.4 §3.2:

   Raw Pressure (1 kHz, after MCU acquisition)
       ↓ decimate by 10 → 100 Hz
       ↓ HPF 0.05 Hz   (DC drift removal)
       ↓ Notch 50 Hz   (Korean grid)
       ↓ FOUR PARALLEL BRANCHES:
           (a) Breath path:     LPF 30 Hz Bessel 4th
           (b) FOT path:        BPF 3.5~8.5 Hz
           (c) Cardiogenic path: BPF 0.8~3.0 Hz
           (d) Snore path:      HPF 60 Hz + LPF 200 Hz (limited at fs=100)

Note on Nyquist: with fs=100 Hz, max usable frequency is 50 Hz, so the snore
path here is approximated as HPF 30 Hz to capture what's available.
For full snore analysis (60-200 Hz), increase fs to 1000 Hz upstream.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def decimate_signal(raw: np.ndarray, factor: int = 10) -> np.ndarray:
    """Anti-aliased decimation. Pass-through if factor <= 1."""
    if factor <= 1:
        return raw.astype(float)
    return sps.decimate(raw, factor, ftype="iir", zero_phase=True)


def highpass(signal_in: np.ndarray, fs: float, cutoff_hz: float = 0.05,
             order: int = 2) -> np.ndarray:
    """Butterworth HPF for DC drift removal."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return signal_in.astype(float)
    sos = sps.butter(order, cutoff_hz / nyq, btype="highpass", output="sos")
    return sps.sosfiltfilt(sos, signal_in)


def notch_filter(signal_in: np.ndarray, fs: float, freq_hz: float = 50.0,
                 q_factor: float = 30.0) -> np.ndarray:
    """IIR notch filter for power-line interference (50 Hz Korea / 60 Hz US)."""
    nyq = fs / 2.0
    if freq_hz >= nyq:
        return signal_in.astype(float)
    b, a = sps.iirnotch(freq_hz / nyq, q_factor)
    return sps.filtfilt(b, a, signal_in)


def lowpass_breath(signal_in: np.ndarray, fs: float,
                   cutoff_hz: float = 30.0, order: int = 4) -> np.ndarray:
    """Bessel LPF for breathing-band signal (preserves breath shape)."""
    nyq = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.95)
    sos = sps.bessel(order, cutoff / nyq, btype="lowpass", output="sos", norm="phase")
    return sps.sosfiltfilt(sos, signal_in)


def bandpass_fot(signal_in: np.ndarray, fs: float,
                 low_hz: float = 3.5, high_hz: float = 8.5,
                 order: int = 4) -> np.ndarray:
    """BPF for FOT response (4 Hz / 8 Hz forced oscillation analysis)."""
    nyq = fs / 2.0
    high = min(high_hz, nyq * 0.95)
    sos = sps.butter(order, [low_hz / nyq, high / nyq], btype="bandpass",
                     output="sos")
    return sps.sosfiltfilt(sos, signal_in)


def bandpass_cardiogenic(signal_in: np.ndarray, fs: float,
                         low_hz: float = 0.8, high_hz: float = 3.0,
                         order: int = 4) -> np.ndarray:
    """BPF for cardiogenic oscillation (heart rate band, 1~3 Hz)."""
    nyq = fs / 2.0
    sos = sps.butter(order, [low_hz / nyq, high_hz / nyq],
                     btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, signal_in)


def bandpass_snore(signal_in: np.ndarray, fs: float,
                   low_hz: float = 30.0, high_hz: float | None = None) -> np.ndarray:
    """HPF for snore detection. With fs=100 Hz, limited to ~30 Hz upper bound.

    For full 60~200 Hz analysis, run on raw 1 kHz upstream signal.
    """
    nyq = fs / 2.0
    if low_hz >= nyq * 0.95:
        return np.zeros_like(signal_in, dtype=float)
    if high_hz is None or high_hz >= nyq * 0.95:
        # HPF only
        sos = sps.butter(4, low_hz / nyq, btype="highpass", output="sos")
    else:
        sos = sps.butter(4, [low_hz / nyq, high_hz / nyq],
                         btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, signal_in)


def stage2_refine(raw_signal: np.ndarray, fs_in: float,
                  fs_out: float = 100.0,
                  notch_hz: float = 50.0) -> np.ndarray:
    """Apply Stage 2 baseline refinement: decimate → HPF → Notch.

    Returns the cleaned signal at `fs_out` ready for 4-branch BPF.
    """
    factor = max(int(round(fs_in / fs_out)), 1)
    refined = decimate_signal(raw_signal, factor)
    refined = highpass(refined, fs_out, cutoff_hz=0.05)
    if notch_hz < fs_out / 2.0:
        refined = notch_filter(refined, fs_out, freq_hz=notch_hz)
    return refined


def apply_4branch_bpf(refined_signal: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Stage 2 — 4 parallel band-pass branches.

    Returns dict with keys:
      'breath'      : 0.05 ~ 30 Hz   (respiration shape)
      'fot'         : 3.5  ~ 8.5 Hz  (FOT 4Hz/8Hz response)
      'cardiogenic' : 0.8  ~ 3.0 Hz  (heart rate band)
      'snore'       : 30+  Hz        (snore proxy at fs=100)
    """
    return {
        "breath": lowpass_breath(refined_signal, fs),
        "fot": bandpass_fot(refined_signal, fs),
        "cardiogenic": bandpass_cardiogenic(refined_signal, fs),
        "snore": bandpass_snore(refined_signal, fs),
    }
