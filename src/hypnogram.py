"""EDF+ Hypnogram annotation parser (Sleep-EDF format).

Independent of `edf_loader`; only depends on pyedflib + numpy + pandas.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib


class Stage(Enum):
    W = "W"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    REM = "REM"
    MOVE = "MOVE"
    UNK = "UNK"


# Mapping from EDF+ annotation strings to Stage.
# Sleep-EDF uses "Sleep stage W", "Sleep stage 1", ..., "Sleep stage R",
# "Sleep stage ?", "Movement time".
_ANNOT_TO_STAGE: dict[str, Stage] = {
    "Sleep stage W": Stage.W,
    "Sleep stage 1": Stage.N1,
    "Sleep stage 2": Stage.N2,
    "Sleep stage 3": Stage.N3,
    "Sleep stage 4": Stage.N3,   # legacy R&K stage 4 → AASM N3
    "Sleep stage R": Stage.REM,
    "Sleep stage ?": Stage.UNK,
    "Movement time": Stage.MOVE,
}


@dataclass(frozen=True)
class HypnogramEpoch:
    start_sec: float
    duration_sec: float
    stage: Stage


def load_hypnogram(path: Path) -> list[HypnogramEpoch]:
    """Parse an EDF+ Hypnogram file into a list of epoch records.

    Unknown annotation strings fall back to ``Stage.UNK``.
    """
    path = Path(path)
    reader = pyedflib.EdfReader(str(path))
    try:
        starts, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()

    epochs: list[HypnogramEpoch] = []
    for s, d, desc in zip(starts, durations, descriptions):
        stage = _ANNOT_TO_STAGE.get(desc.strip(), Stage.UNK)
        epochs.append(HypnogramEpoch(
            start_sec=float(s),
            duration_sec=float(d),
            stage=stage,
        ))
    return epochs


def to_stage_series(
    epochs: list[HypnogramEpoch], total_dur_sec: float
) -> pd.DataFrame:
    """Densify epochs into a 1 Hz DataFrame with columns ``t_sec`` and ``stage``.

    Rows are 1 second apart, indexed 0..floor(total_dur_sec) - 1. Any second
    not covered by an epoch is filled with ``"UNK"``. Epochs extending past
    *total_dur_sec* are truncated.
    """
    n = int(total_dur_sec)
    stages = np.full(n, Stage.UNK.value, dtype=object)
    for e in epochs:
        a = max(0, int(e.start_sec))
        b = min(n, int(e.start_sec + e.duration_sec))
        if b > a:
            stages[a:b] = e.stage.value
    return pd.DataFrame({"t_sec": np.arange(n, dtype=np.float64),
                         "stage": stages})
