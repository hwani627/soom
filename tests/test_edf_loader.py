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


def test_classify_channel_resp_variants():
    assert edf_loader.classify_channel("Respiratory effort") == "Resp"
    assert edf_loader.classify_channel("Respiration") == "Resp"
    assert edf_loader.classify_channel("Thoracic belt") == "Resp"
    assert edf_loader.classify_channel("Abdominal belt") == "Resp"


def test_classify_channel_spo2():
    assert edf_loader.classify_channel("SpO2") == "SpO2"
    assert edf_loader.classify_channel("Sa02") == "SpO2"


def test_classify_channel_falls_back_to_other():
    assert edf_loader.classify_channel("Random Sensor X") == "Other"
    assert edf_loader.classify_channel("") == "Other"


def test_list_edf_files_excludes_hypnogram(tmp_path: Path):
    (tmp_path / "a-PSG.edf").touch()
    (tmp_path / "a-Hypnogram.edf").touch()
    (tmp_path / "ignored.txt").touch()
    files = edf_loader.list_edf_files(tmp_path)
    assert [p.name for p in files] == ["a-PSG.edf"]


def test_list_edf_files_returns_empty_when_no_dir(tmp_path: Path):
    assert edf_loader.list_edf_files(tmp_path / "missing") == []


def test_list_edf_files_sorted_alphabetically(tmp_path: Path):
    (tmp_path / "b-PSG.edf").touch()
    (tmp_path / "a-PSG.edf").touch()
    files = edf_loader.list_edf_files(tmp_path)
    assert [p.name for p in files] == ["a-PSG.edf", "b-PSG.edf"]


def test_pair_hypnogram_exact_stem_swap(tmp_path: Path):
    psg = tmp_path / "a-PSG.edf"; psg.touch()
    hyp = tmp_path / "a-Hypnogram.edf"; hyp.touch()
    assert edf_loader.pair_hypnogram(psg) == hyp


def test_pair_hypnogram_sleep_edf_cassette_prefix(tmp_path: Path):
    """Sleep-EDF Cassette files differ in the last char of the stem."""
    psg = tmp_path / "SC4001E0-PSG.edf"; psg.touch()
    hyp = tmp_path / "SC4001EC-Hypnogram.edf"; hyp.touch()
    assert edf_loader.pair_hypnogram(psg) == hyp


def test_pair_hypnogram_returns_none_when_missing(tmp_path: Path):
    psg = tmp_path / "x-PSG.edf"; psg.touch()
    assert edf_loader.pair_hypnogram(psg) is None


def test_pair_hypnogram_only_searches_same_dir(tmp_path: Path):
    nested = tmp_path / "nested"; nested.mkdir()
    psg = tmp_path / "a-PSG.edf"; psg.touch()
    (nested / "a-Hypnogram.edf").touch()  # wrong dir, must not match
    assert edf_loader.pair_hypnogram(psg) is None


def test_load_meta_returns_three_channels():
    meta = edf_loader.load_meta(SYNTH_PSG)
    assert len(meta.channels) == 3
    assert meta.duration_sec == 30
    assert meta.path == SYNTH_PSG


def test_load_meta_classifies_channels():
    meta = edf_loader.load_meta(SYNTH_PSG)
    by_label = {ch.label: ch.group for ch in meta.channels}
    assert by_label["EEG Fpz-Cz"] == "EEG"
    assert by_label["EOG horizontal"] == "EOG"
    assert by_label["SpO2"] == "SpO2"


def test_load_meta_populates_sample_rates():
    meta = edf_loader.load_meta(SYNTH_PSG)
    rates = {ch.label: ch.sample_rate for ch in meta.channels}
    assert rates["EEG Fpz-Cz"] == 100.0
    assert rates["SpO2"] == 1.0


def test_load_signal_shape_matches_meta():
    meta = edf_loader.load_meta(SYNTH_PSG)
    eeg = edf_loader.load_signal(SYNTH_PSG, ch_idx=0)
    assert isinstance(eeg, np.ndarray)
    assert eeg.shape == (meta.channels[0].n_samples,)
    assert eeg.shape == (3000,)  # 100 Hz × 30 s


def test_load_signal_spo2_is_one_hertz():
    sig = edf_loader.load_signal(SYNTH_PSG, ch_idx=2)
    assert sig.shape == (30,)
    assert np.all((sig > 90) & (sig < 100))


def test_load_meta_raises_on_corrupt_file(tmp_path: Path):
    bad = tmp_path / "not_real.edf"
    bad.write_bytes(b"NOT AN EDF HEADER")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        edf_loader.load_meta(bad)
