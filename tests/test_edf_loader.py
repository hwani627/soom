"""Tests for src/edf_loader.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import edf_loader

REPO = Path(__file__).resolve().parents[1]
SYNTH_PSG = REPO / "tests" / "fixtures" / "synth_psg.edf"
SYNTH_HYP = REPO / "tests" / "fixtures" / "synth_hypnogram.edf"


def test_classify_channel_eeg():
    assert edf_loader.classify_channel("EEG Fpz-Cz") == "EEG"
    assert edf_loader.classify_channel("EEG Pz-Oz") == "EEG"


def test_classify_channel_eog():
    assert edf_loader.classify_channel("EOG horizontal") == "EOG"


def test_classify_channel_emg():
    assert edf_loader.classify_channel("EMG submental") == "EMG"


def test_classify_channel_resp():
    assert edf_loader.classify_channel("Resp oro-nasal") == "Resp"


def test_classify_channel_spo2():
    assert edf_loader.classify_channel("SpO2") == "SpO2"
    assert edf_loader.classify_channel("Sa02") == "SpO2"


def test_classify_channel_falls_back_to_other():
    assert edf_loader.classify_channel("Random Sensor X") == "Other"
    assert edf_loader.classify_channel("") == "Other"
