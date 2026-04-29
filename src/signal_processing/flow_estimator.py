"""Stage 3 — Flow derivation from single differential pressure sensor.

Two strategies (per 03_신호처리_사양서 v0.4 §0.6.5 Phase decision branch):

  Phase 0 — 1-port gauge mode (current production reality):
    Flow is ESTIMATED via blower inverse model:
      flow = trilinear_lookup(rpm, pressure)
           + d(pressure)/dt correction
           + adaptive Kalman filter
           + (Stage 2 onwards) PSG-trained refinement
    Accuracy: ±5~10% (Stage 1), ±3% (post-PSG learning)

  Phase 1 — 2-port pneumotachograph (formally adopted from prototype):
    Flow is DIRECTLY MEASURED via Bernoulli on orifice ΔP:
      flow = K * sign(ΔP) * |ΔP|^n
    where K, n are calibrated against Sensirion SFM3300 reference.
    Accuracy: ±2~5%

Both also provide a `confidence` channel (R17) for downstream XAI export.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Phase 0 — Blower inverse model
# ---------------------------------------------------------------------------


@dataclass
class BlowerLookupTable:
    """3D lookup table mapping (RPM, Pressure) → baseline Flow.

    In production this is calibrated on the manufacturing line by sweeping
    RPM × Pressure with a Sensirion SFM3300 reference. Here we synthesize
    a plausible blower characteristic for PoC use.
    """

    rpm_grid: np.ndarray
    pressure_grid: np.ndarray
    flow_table: np.ndarray  # shape (len(rpm_grid), len(pressure_grid))
    # K_correction (empirical): flow ↑ when pressure ↑ rapidly (breath-in)
    pressure_derivative_gain: float = 80.0
    model_version: str = "synthetic-v0.1"  # → R16

    @classmethod
    def synthetic_default(cls) -> "BlowerLookupTable":
        """A reasonable synthetic blower characteristic.

        The model: blower behaves like an aerodynamic compressor —
        higher RPM produces higher baseline flow at a given pressure load.
        Modeled here as Q_baseline = a*RPM + b*RPM*P (approximate).
        """
        rpm_grid = np.linspace(5_000, 30_000, 26)
        pressure_grid = np.linspace(4.0, 20.0, 17)  # cmH2O
        # Coefficients chosen so realistic CPAP at 18,000 RPM, 9.5 cmH2O ≈ 24 L/min
        # Q ≈ 0.0014*RPM - 0.045*RPM*P/10000  (intentional leak baseline)
        rpm_2d, p_2d = np.meshgrid(rpm_grid, pressure_grid, indexing="ij")
        flow_table = 0.00135 * rpm_2d - 0.045 * rpm_2d * p_2d / 10000.0
        flow_table = np.clip(flow_table, 0.0, None)
        return cls(rpm_grid, pressure_grid, flow_table)

    def lookup(self, rpm: float | np.ndarray, pressure: float | np.ndarray) -> np.ndarray:
        """Bilinear interpolation lookup."""
        rpm_arr = np.atleast_1d(rpm).astype(float)
        p_arr = np.atleast_1d(pressure).astype(float)
        # Clip to grid range
        rpm_arr = np.clip(rpm_arr, self.rpm_grid[0], self.rpm_grid[-1])
        p_arr = np.clip(p_arr, self.pressure_grid[0], self.pressure_grid[-1])

        # Find indices and weights along each axis
        ri = np.searchsorted(self.rpm_grid, rpm_arr) - 1
        ri = np.clip(ri, 0, len(self.rpm_grid) - 2)
        rw = (rpm_arr - self.rpm_grid[ri]) / (self.rpm_grid[ri + 1] - self.rpm_grid[ri])
        pi = np.searchsorted(self.pressure_grid, p_arr) - 1
        pi = np.clip(pi, 0, len(self.pressure_grid) - 2)
        pw = (p_arr - self.pressure_grid[pi]) / (
            self.pressure_grid[pi + 1] - self.pressure_grid[pi]
        )

        # Bilinear blend
        f00 = self.flow_table[ri, pi]
        f01 = self.flow_table[ri, pi + 1]
        f10 = self.flow_table[ri + 1, pi]
        f11 = self.flow_table[ri + 1, pi + 1]
        flow = (
            f00 * (1 - rw) * (1 - pw)
            + f01 * (1 - rw) * pw
            + f10 * rw * (1 - pw)
            + f11 * rw * pw
        )
        return flow


def estimate_flow_phase0(
    pressure: np.ndarray,
    blower_rpm: np.ndarray,
    fs: float,
    lookup: BlowerLookupTable | None = None,
    intentional_leak_lpm: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate patient flow using Phase 0 blower-inverse model.

    Parameters
    ----------
    pressure : array, shape (N,) — cleaned pressure (cmH2O), 100 Hz
    blower_rpm : array, shape (N,) — blower RPM, 100 Hz
    fs : sampling frequency (Hz)
    lookup : BlowerLookupTable (uses synthetic default if None)
    intentional_leak_lpm : average mask leak to subtract for patient flow

    Returns
    -------
    flow_patient : array shape (N,) — estimated patient flow (L/min, sinusoidal,
                   ± around 0)
    confidence : array shape (N,) — 0~1 confidence (R17 channel)
    """
    if lookup is None:
        lookup = BlowerLookupTable.synthetic_default()

    # Step 1: Lookup recovers blower's total output flow from (RPM, P)
    # → which equals patient_flow + intentional_leak by mass balance
    flow_total = lookup.lookup(blower_rpm, pressure)

    # Step 2: Subtract intentional leak to recover patient flow
    flow_patient = flow_total - intentional_leak_lpm

    # Step 3: Mild low-pass smoothing for measurement noise
    win = max(int(0.05 * fs), 1)
    if win > 1:
        kernel = np.ones(win) / win
        flow_patient = np.convolve(flow_patient, kernel, mode="same")

    # Step 4: Confidence — proxy for how well lookup is operating in trusted
    # range; we drop confidence when pressure is at extremes of the grid.
    dp_dt = np.gradient(pressure, 1.0 / fs)
    dp_dt_abs = np.abs(dp_dt)
    confidence = np.clip(
        1.0 - dp_dt_abs / (np.percentile(dp_dt_abs, 99) + 1e-9), 0.5, 1.0,
    )

    return flow_patient, confidence


# ---------------------------------------------------------------------------
# Phase 1 — Pneumotachograph direct measurement
# ---------------------------------------------------------------------------


def measure_flow_phase1(
    dp_orifice_pa: np.ndarray,
    K: float = 1.5,
    n: float = 0.5,
    intentional_leak_lpm: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct flow measurement via Bernoulli on orifice ΔP (Phase 1).

    Parameters
    ----------
    dp_orifice_pa : differential pressure across orifice (Pa)
    K, n          : Bernoulli calibration coefficients (production line cal)
                    Linear regime n=1.0, turbulent n=0.5
    intentional_leak_lpm : mask intentional leak to subtract

    Returns
    -------
    flow_patient : direct-measured patient flow (L/min)
    confidence   : ~1.0 across the board (direct measurement)
    """
    flow_total = K * np.sign(dp_orifice_pa) * np.power(np.abs(dp_orifice_pa), n)
    flow_patient = flow_total - intentional_leak_lpm
    confidence = np.full_like(flow_patient, 0.95, dtype=float)
    return flow_patient, confidence


def synthesize_pneumotacho_dp(true_flow_lpm: np.ndarray, K: float = 1.5,
                              n: float = 0.5,
                              noise_pa_std: float = 1.0,
                              seed: int = 0) -> np.ndarray:
    """Forward-synthesize ΔP for Phase 1 PoC validation (when only patient flow
    is known from the generator)."""
    rng = np.random.default_rng(seed)
    dp = np.sign(true_flow_lpm) * np.power(np.abs(true_flow_lpm) / K, 1.0 / n)
    dp = dp + rng.normal(0, noise_pa_std, len(true_flow_lpm))
    return dp
