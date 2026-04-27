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
