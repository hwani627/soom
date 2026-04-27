"""One-shot generator for the synthetic EDF fixtures used in tests.

Run once:
    python tests/fixtures/_make_synth.py

Outputs (committed to git):
    tests/fixtures/synth_psg.edf         — 30 s × 3 channels
    tests/fixtures/synth_hypnogram.edf   — single 30 s "W" epoch
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib

OUT_DIR = Path(__file__).parent
PSG_PATH = OUT_DIR / "synth_psg.edf"
HYP_PATH = OUT_DIR / "synth_hypnogram.edf"

DURATION_SEC = 30
START_DT = datetime(2026, 4, 27, 22, 0, 0)


def _make_psg() -> None:
    rng = np.random.default_rng(42)
    fs_eeg = 100  # Hz
    fs_eog = 100
    fs_spo2 = 1
    t_eeg = np.arange(0, DURATION_SEC, 1 / fs_eeg)
    t_spo2 = np.arange(0, DURATION_SEC, 1 / fs_spo2)

    eeg = (50.0 * np.sin(2 * np.pi * 10 * t_eeg)
           + 5.0 * rng.standard_normal(len(t_eeg))).astype(np.float32)
    eog = (100.0 * np.sin(2 * np.pi * 1.0 * t_eeg)).astype(np.float32)
    spo2 = (97.0 + rng.standard_normal(len(t_spo2)) * 0.5).astype(np.float32)

    signals = [eeg, eog, spo2]
    headers = [
        {"label": "EEG Fpz-Cz", "dimension": "uV",
         "sample_frequency": fs_eeg,
         "physical_min": -200.0, "physical_max": 200.0,
         "digital_min": -32768, "digital_max": 32767,
         "transducer": "", "prefilter": ""},
        {"label": "EOG horizontal", "dimension": "uV",
         "sample_frequency": fs_eog,
         "physical_min": -200.0, "physical_max": 200.0,
         "digital_min": -32768, "digital_max": 32767,
         "transducer": "", "prefilter": ""},
        {"label": "SpO2", "dimension": "%",
         "sample_frequency": fs_spo2,
         "physical_min": 0.0, "physical_max": 100.0,
         "digital_min": -32768, "digital_max": 32767,
         "transducer": "", "prefilter": ""},
    ]

    writer = pyedflib.EdfWriter(str(PSG_PATH), len(signals),
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(START_DT)
        writer.setSignalHeaders(headers)
        writer.writeSamples(signals)
    finally:
        writer.close()


def _make_hypnogram() -> None:
    # EDF+ file with one annotation: a single "Sleep stage W" epoch covering 30 s.
    writer = pyedflib.EdfWriter(str(HYP_PATH), 0,
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(START_DT)
        writer.writeAnnotation(0.0, 30.0, "Sleep stage W")
    finally:
        writer.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _make_psg()
    _make_hypnogram()
    print(f"Wrote {PSG_PATH} ({PSG_PATH.stat().st_size} bytes)")
    print(f"Wrote {HYP_PATH} ({HYP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
