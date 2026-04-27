"""EDF (European Data Format) parsing layer.

Pure-function loader on top of pyedflib. No Streamlit imports — testable
without any UI runtime.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib

# Channel-group classification rules.
# Patterns are word-boundary anchored so substring collisions are impossible
# (e.g., "EEOG" → Other, not EOG). The rule order doesn't affect correctness;
# it just reflects test ordering and how often each group appears in PSG data.
# Note: "Sa02" (with digit 0) is treated as a common transcription variant of
# "SaO2"; the pattern covers both the letter O and the digit 0.
_GROUP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("SpO2", re.compile(r"\b(spo2|sp\s*[o0]\s*2|sa[o0]2)\b", re.IGNORECASE)),
    ("EEG",  re.compile(r"\beeg\b",  re.IGNORECASE)),
    ("EOG",  re.compile(r"\beog\b",  re.IGNORECASE)),
    ("EMG",  re.compile(r"\bemg\b",  re.IGNORECASE)),
    ("Resp", re.compile(
        r"\b(resp(ir(ation|atory)?)?|airflow|nasal|oro-?nasal|thor(ax|acic)?|abd(om(en|inal)?)?)\b",
        re.IGNORECASE)),
)


def classify_channel(label: str) -> str:
    """Map a channel label to one of: EEG / EOG / EMG / Resp / SpO2 / Other."""
    if not label:
        return "Other"
    for group, pattern in _GROUP_RULES:
        if pattern.search(label):
            return group
    return "Other"


@dataclass(frozen=True)
class ChannelInfo:
    """One signal channel's metadata (ordering matches the EDF file)."""

    label: str
    sample_rate: float
    n_samples: int
    physical_dim: str
    group: str  # Result of classify_channel()


@dataclass(frozen=True)
class EdfMeta:
    """Header-level summary of an EDF file."""

    path: Path
    subject_id: str
    start_datetime: datetime
    duration_sec: float
    channels: tuple[ChannelInfo, ...] = field(default_factory=tuple)


def list_edf_files(root: Path) -> list[Path]:
    """All `*.edf` files in *root*, excluding `*-Hypnogram.edf`. Alphabetical.

    Returns an empty list if *root* does not exist.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.glob("*.edf")
        if not p.name.endswith("-Hypnogram.edf")
    )


def pair_hypnogram(psg_path: Path) -> Path | None:
    """Locate the Hypnogram EDF that pairs with *psg_path*, if any.

    Two-stage match in the same directory:
      1. Exact stem swap: ``<stem with '-PSG' replaced by '-Hypnogram'>.edf``.
      2. Subject-night prefix: first 7 chars of the stem (Sleep-EDF Cassette
         convention; PSG and Hypnogram differ only in the last char of stem).

    Returns ``None`` if neither produces a hit.
    """
    psg_path = Path(psg_path)
    parent = psg_path.parent

    # Stage 1: exact stem swap
    if "-PSG" in psg_path.stem:
        candidate = parent / (psg_path.stem.replace("-PSG", "-Hypnogram") + ".edf")
        if candidate.is_file():
            return candidate

    # Stage 2: subject-night prefix glob (Sleep-EDF Cassette pattern)
    if len(psg_path.stem) >= 7:
        prefix = psg_path.stem[:7]
        for cand in sorted(parent.glob(f"{prefix}*-Hypnogram.edf")):
            return cand  # first match wins

    return None


def load_meta(path: Path) -> EdfMeta:
    """Read header-level info from an EDF file.

    Raises:
        OSError / ValueError / RuntimeError on a corrupt or non-EDF file
        (the underlying pyedflib exceptions, surfaced as-is).
    """
    path = Path(path)
    reader = pyedflib.EdfReader(str(path))
    try:
        n = reader.signals_in_file
        labels = reader.getSignalLabels()
        sample_rates = reader.getSampleFrequencies()
        n_samples_arr = reader.getNSamples()
        physical_dims = [reader.getPhysicalDimension(i) for i in range(n)]
        start_dt = reader.getStartdatetime()
        duration_sec = float(reader.getFileDuration())
        # Subject identifier is in the header's local patient field.
        subject_id = (reader.getPatientCode() or reader.getPatientName()
                      or path.stem)
        channels = tuple(
            ChannelInfo(
                label=labels[i],
                sample_rate=float(sample_rates[i]),
                n_samples=int(n_samples_arr[i]),
                physical_dim=physical_dims[i],
                group=classify_channel(labels[i]),
            )
            for i in range(n)
        )
    finally:
        reader.close()
    return EdfMeta(
        path=path,
        subject_id=subject_id,
        start_datetime=start_dt,
        duration_sec=duration_sec,
        channels=channels,
    )


def load_signal(path: Path, ch_idx: int) -> np.ndarray:
    """Load one channel's full signal as a float32 numpy array."""
    path = Path(path)
    reader = pyedflib.EdfReader(str(path))
    try:
        sig = reader.readSignal(ch_idx).astype(np.float32, copy=False)
    finally:
        reader.close()
    return sig
