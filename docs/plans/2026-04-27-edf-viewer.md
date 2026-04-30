# EDF Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent EDF viewer page to the existing Streamlit app, supporting Sleep-EDF format PSG files with channel grouping, hypnogram overlay, free-zoom + 30s epoch modes, and dynamic LTTB downsampling.

**Architecture:** New Streamlit page `pages/2_EDF_Viewer.py` (UI only) → `src/edf_plotting.py` (Plotly + plotly-resampler) → `src/edf_loader.py` (pyedflib parsing) + `src/hypnogram.py` (EDF+ annotations). The existing CSV pipeline (`src/loader.py`, `src/plotting.py`, etc.) is untouched. All `src/edf_*` and `src/hypnogram.py` modules are pure functions with no Streamlit imports for testability.

**Tech Stack:** Python 3.11+, Streamlit ≥1.32, Plotly ≥5.18, **pyedflib ≥0.1.36 (NEW)**, **plotly-resampler ≥0.10 (NEW)**, NumPy, Pandas, pytest.

**Spec:** `docs/specs/2026-04-27-edf-viewer-design.md`

---

## Task Map

| #   | Task                                             | Files                                                                                                 | TDD                 |
| --- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------- |
| 1   | Add dependencies                                 | `requirements.txt`                                                                                    | smoke import        |
| 2   | Build synthetic EDF fixtures                     | `tests/fixtures/_make_synth.py`, `tests/fixtures/synth_psg.edf`, `tests/fixtures/synth_hypnogram.edf` | self-test in script |
| 3   | `edf_loader`: dataclasses + `classify_channel`   | `src/edf_loader.py`, `tests/test_edf_loader.py`                                                       | unit                |
| 4   | `edf_loader`: file discovery + hypnogram pairing | `src/edf_loader.py`, `tests/test_edf_loader.py`                                                       | unit                |
| 5   | `edf_loader`: `load_meta` + `load_signal`        | `src/edf_loader.py`, `tests/test_edf_loader.py`                                                       | unit                |
| 6   | `hypnogram.py` full module                       | `src/hypnogram.py`, `tests/test_hypnogram.py`                                                         | unit                |
| 7   | `edf_plotting.make_freezoom_figure`              | `src/edf_plotting.py`, `tests/test_edf_plotting.py`                                                   | smoke               |
| 8   | `edf_plotting.make_epoch_figure`                 | `src/edf_plotting.py`, `tests/test_edf_plotting.py`                                                   | smoke               |
| 9   | Streamlit page                                   | `pages/2_EDF_Viewer.py`                                                                               | manual              |

---

## Task 1: Add dependencies and verify imports

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append the two new dependencies**

Append to `requirements.txt`:

```
pyedflib>=0.1.36
plotly-resampler>=0.10
```

Resulting full file:

```
streamlit>=1.32
pandas>=2.1
numpy>=1.26
scipy>=1.11
plotly>=5.18
scikit-learn>=1.3
pyedflib>=0.1.36
plotly-resampler>=0.10
```

- [ ] **Step 2: Install into the active environment**

Run from repo root:

```bash
pip install -r requirements.txt
```

Expected: `pyedflib` and `plotly-resampler` install without C-compile errors. `dash` is pulled in as a transitive dep of `plotly-resampler` — that's expected and harmless.

- [ ] **Step 3: Smoke test the imports**

Run from repo root:

```bash
python -c "import pyedflib; import plotly_resampler; print('OK', pyedflib.__version__, plotly_resampler.__version__)"
```

Expected output: a single line `OK 0.1.x 0.10.x`. Any ImportError means the install didn't take — fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add pyedflib and plotly-resampler for EDF viewer"
```

---

## Task 2: Synthetic EDF fixtures

A 30-second synthetic EDF with 3 channels (EEG-like, EOG-like, SpO2-like) plus a paired Hypnogram annotation file. ~10 KB committed to git. Used by all subsequent tests.

**Files:**
- Create: `tests/fixtures/_make_synth.py` (generator script — runs once)
- Create: `tests/fixtures/synth_psg.edf` (output, committed)
- Create: `tests/fixtures/synth_hypnogram.edf` (output, committed)
- Create: `tests/fixtures/__init__.py` (empty, makes the dir a package)

- [ ] **Step 1: Create the empty fixture package marker**

Create `tests/fixtures/__init__.py`:

```python
```

(empty file — pytest discovery aid)

- [ ] **Step 2: Write the generator script**

Create `tests/fixtures/_make_synth.py`:

```python
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
```

- [ ] **Step 3: Run the generator**

```bash
python tests/fixtures/_make_synth.py
```

Expected output: two `Wrote ...` lines, both with positive byte counts (PSG ≈ 30 KB, Hypnogram ≈ 1 KB).

- [ ] **Step 4: Sanity check the fixtures with a one-liner**

```bash
python -c "import pyedflib; r = pyedflib.EdfReader('tests/fixtures/synth_psg.edf'); print(r.signals_in_file, r.getSignalLabels(), r.getFileDuration()); r.close()"
```

Expected: `3 ['EEG Fpz-Cz', 'EOG horizontal', 'SpO2'] 30`.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/_make_synth.py tests/fixtures/synth_psg.edf tests/fixtures/synth_hypnogram.edf
git commit -m "test: add synthetic EDF fixtures (30s × 3ch + hypnogram)"
```

---

## Task 3: `edf_loader` — dataclasses and `classify_channel`

Pure-function building blocks. No filesystem yet.

**Files:**
- Create: `src/edf_loader.py`
- Create: `tests/test_edf_loader.py`

- [ ] **Step 1: Write the failing tests for `classify_channel`**

Create `tests/test_edf_loader.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: collection error (`ModuleNotFoundError: No module named 'src.edf_loader'`).

- [ ] **Step 3: Implement the dataclasses and `classify_channel`**

Create `src/edf_loader.py`:

```python
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
# Order matters: SpO2 comes before generic O-checks, EEG before EOG/EMG so
# "EOG" doesn't accidentally match "EEG" patterns.
_GROUP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("SpO2", re.compile(r"\b(spo2|sao2|sp\s*o\s*2)\b", re.IGNORECASE)),
    ("EEG",  re.compile(r"\beeg\b",  re.IGNORECASE)),
    ("EOG",  re.compile(r"\beog\b",  re.IGNORECASE)),
    ("EMG",  re.compile(r"\bemg\b",  re.IGNORECASE)),
    ("Resp", re.compile(r"\b(resp|respir|airflow|nasal|oro-?nasal|thor|abd)\b",
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/edf_loader.py tests/test_edf_loader.py
git commit -m "feat(edf_loader): dataclasses and channel classifier"
```

---

## Task 4: `edf_loader` — file discovery and hypnogram pairing

Build the directory-level helpers. Both are pure-function and use only `pathlib`.

**Files:**
- Modify: `src/edf_loader.py` (append)
- Modify: `tests/test_edf_loader.py` (append)

- [ ] **Step 1: Write the failing tests for `list_edf_files` and `pair_hypnogram`**

Append to `tests/test_edf_loader.py`:

```python
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
```

- [ ] **Step 2: Run and confirm they fail**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: 7 new failures (`AttributeError: module 'src.edf_loader' has no attribute 'list_edf_files'` and similar).

- [ ] **Step 3: Implement `list_edf_files` and `pair_hypnogram`**

Append to `src/edf_loader.py`:

```python
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
```

- [ ] **Step 4: Run and confirm they pass**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: 13 passed total (6 from Task 3 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/edf_loader.py tests/test_edf_loader.py
git commit -m "feat(edf_loader): list_edf_files and pair_hypnogram"
```

---

## Task 5: `edf_loader` — `load_meta` and `load_signal`

The actual pyedflib-touching functions. They use the synthetic fixture from Task 2.

**Files:**
- Modify: `src/edf_loader.py` (append)
- Modify: `tests/test_edf_loader.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_edf_loader.py`:

```python
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
```

- [ ] **Step 2: Run and confirm they fail**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: 6 new failures (`AttributeError: module 'src.edf_loader' has no attribute 'load_meta'`).

- [ ] **Step 3: Implement `load_meta` and `load_signal`**

Append to `src/edf_loader.py`. **At the top of the file, add `import pyedflib` to the import block** (right after `import numpy as np`):

```python
import pyedflib
```

Then append at the bottom:

```python
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
```

- [ ] **Step 4: Run and confirm they pass**

```bash
pytest tests/test_edf_loader.py -v
```

Expected: 19 passed total (13 from prior tasks + 6 new).

- [ ] **Step 5: Commit**

```bash
git add src/edf_loader.py tests/test_edf_loader.py
git commit -m "feat(edf_loader): load_meta and load_signal via pyedflib"
```

---

## Task 6: `hypnogram.py` — Stage enum and annotation parsing

Independent module; no dependency on `edf_loader`.

**Files:**
- Create: `src/hypnogram.py`
- Create: `tests/test_hypnogram.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hypnogram.py`:

```python
"""Tests for src/hypnogram.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import hypnogram

REPO = Path(__file__).resolve().parents[1]
SYNTH_HYP = REPO / "tests" / "fixtures" / "synth_hypnogram.edf"


def test_stage_enum_covers_aasm_five_stages():
    names = {s.name for s in hypnogram.Stage}
    assert {"W", "N1", "N2", "N3", "REM"}.issubset(names)


def test_load_hypnogram_returns_one_w_epoch():
    epochs = hypnogram.load_hypnogram(SYNTH_HYP)
    assert len(epochs) == 1
    e = epochs[0]
    assert e.start_sec == 0.0
    assert e.duration_sec == 30.0
    assert e.stage == hypnogram.Stage.W


def test_to_stage_series_truncates_to_total_dur():
    eps = [
        hypnogram.HypnogramEpoch(0.0, 60.0, hypnogram.Stage.W),
        hypnogram.HypnogramEpoch(60.0, 60.0, hypnogram.Stage.N1),
    ]
    df = hypnogram.to_stage_series(eps, total_dur_sec=90.0)
    assert isinstance(df, pd.DataFrame)
    assert df["t_sec"].max() == 89.0
    # First minute should all be W; last 30 s should all be N1.
    assert (df.loc[df["t_sec"] < 60, "stage"] == "W").all()
    assert (df.loc[df["t_sec"] >= 60, "stage"] == "N1").all()


def test_to_stage_series_handles_empty_epoch_list():
    df = hypnogram.to_stage_series([], total_dur_sec=10.0)
    assert (df["stage"] == "UNK").all()
    assert len(df) == 10
```

- [ ] **Step 2: Run and confirm they fail**

```bash
pytest tests/test_hypnogram.py -v
```

Expected: collection error (`ModuleNotFoundError: No module named 'src.hypnogram'`).

- [ ] **Step 3: Implement `src/hypnogram.py`**

Create `src/hypnogram.py`:

```python
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
```

- [ ] **Step 4: Run and confirm they pass**

```bash
pytest tests/test_hypnogram.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full test suite to confirm nothing else broke**

```bash
pytest -v
```

Expected: existing tests + the 19 from `test_edf_loader` + the 4 new ones all pass.

- [ ] **Step 6: Commit**

```bash
git add src/hypnogram.py tests/test_hypnogram.py
git commit -m "feat(hypnogram): EDF+ annotation parser with AASM Stage enum"
```

---

## Task 7: `edf_plotting.make_freezoom_figure`

Wrapper around `plotly_resampler.FigureResampler` that produces stacked subplots and an optional hypnogram band. Smoke-tested only — visual fidelity verified manually in Task 9.

**Files:**
- Create: `src/edf_plotting.py`
- Create: `tests/test_edf_plotting.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_edf_plotting.py`:

```python
"""Smoke tests for src/edf_plotting.py — verify Figures build without error."""
from __future__ import annotations

from pathlib import Path

from src import edf_loader, edf_plotting, hypnogram

REPO = Path(__file__).resolve().parents[1]
SYNTH_PSG = REPO / "tests" / "fixtures" / "synth_psg.edf"
SYNTH_HYP = REPO / "tests" / "fixtures" / "synth_hypnogram.edf"


def test_freezoom_figure_builds_without_hypnogram():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {
        0: edf_loader.load_signal(SYNTH_PSG, 0),
        1: edf_loader.load_signal(SYNTH_PSG, 1),
    }
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=None, channels=[0, 1],
    )
    assert fig.data, "expected at least one trace"


def test_freezoom_figure_builds_with_hypnogram():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {0: edf_loader.load_signal(SYNTH_PSG, 0)}
    hypno = hypnogram.load_hypnogram(SYNTH_HYP)
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=hypno, channels=[0],
    )
    assert fig.data
```

- [ ] **Step 2: Run and confirm they fail**

```bash
pytest tests/test_edf_plotting.py -v
```

Expected: collection error (`ModuleNotFoundError: No module named 'src.edf_plotting'`).

- [ ] **Step 3: Implement `make_freezoom_figure`**

Create `src/edf_plotting.py`:

```python
"""Plotly figures for the EDF viewer page.

`make_freezoom_figure` wraps plotly_resampler.FigureResampler so that the
plot dynamically downsamples (LTTB) on zoom — see Steinarsson 2013 and
Van Der Donckt et al., SoftwareX 2022.

All callers must pass numpy arrays already loaded via edf_loader.load_signal.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

from src.edf_loader import EdfMeta
from src.hypnogram import HypnogramEpoch, Stage

# Group → line color (consistent across both freezoom and epoch modes).
CHANNEL_GROUP_COLORS: dict[str, str] = {
    "EEG":   "#0F62FE",
    "EOG":   "#6929C4",
    "EMG":   "#198038",
    "Resp":  "#FF832B",
    "SpO2":  "#EE5396",
    "Other": "#525252",
}

# Sleep stage band colors for the hypnogram strip.
_STAGE_COLORS: dict[Stage, str] = {
    Stage.W:    "#5cb85c",
    Stage.N1:   "#7ab5e0",
    Stage.N2:   "#3a8fc8",
    Stage.N3:   "#c4a13a",
    Stage.REM:  "#c43a3a",
    Stage.MOVE: "#888888",
    Stage.UNK:  "#dddddd",
}


def make_freezoom_figure(
    meta: EdfMeta,
    signals: dict[int, np.ndarray],
    hypno: list[HypnogramEpoch] | None,
    channels: Iterable[int],
) -> go.Figure:
    """Stacked-subplot view of the whole recording, with dynamic LTTB.

    Args:
        meta: Result of `edf_loader.load_meta`.
        signals: Mapping ``ch_idx -> numpy array`` (one entry per visible channel).
        hypno: Optional list of HypnogramEpoch — drawn as a colored band below.
        channels: Channel indices to plot, in display order (top to bottom).
    """
    chans = [c for c in channels if c in signals]
    n_chan_rows = len(chans)
    if n_chan_rows == 0:
        return go.Figure()

    has_hypno = bool(hypno)
    total_rows = n_chan_rows + (1 if has_hypno else 0)
    row_heights = [1.0] * n_chan_rows + ([0.25] if has_hypno else [])
    titles = [meta.channels[c].label for c in chans] + (
        ["Sleep stage (Hypnogram)"] if has_hypno else []
    )

    base = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=titles,
    )
    fig = FigureResampler(base, default_n_shown_samples=4000)

    for row_idx, ch in enumerate(chans, start=1):
        info = meta.channels[ch]
        sig = signals[ch]
        t = np.arange(len(sig), dtype=np.float64) / max(info.sample_rate, 1.0)
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line=dict(width=1.0,
                          color=CHANNEL_GROUP_COLORS.get(info.group,
                                                         CHANNEL_GROUP_COLORS["Other"])),
                name=info.label, showlegend=False,
            ),
            hf_x=t, hf_y=sig,
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text=info.physical_dim or "", row=row_idx, col=1)

    if has_hypno:
        _add_hypnogram_band(fig, hypno, row=total_rows)

    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    fig.update_layout(
        height=180 * n_chan_rows + (60 if has_hypno else 0) + 80,
        margin=dict(l=55, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _add_hypnogram_band(
    fig: go.Figure, epochs: list[HypnogramEpoch], row: int
) -> None:
    """Render the hypnogram as horizontal colored rectangles in the given row."""
    for e in epochs:
        fig.add_shape(
            type="rect", xref=f"x{row}", yref=f"y{row} domain",
            x0=e.start_sec, x1=e.start_sec + e.duration_sec,
            y0=0, y1=1,
            fillcolor=_STAGE_COLORS.get(e.stage, _STAGE_COLORS[Stage.UNK]),
            line=dict(width=0), layer="below",
        )
    fig.update_yaxes(visible=False, row=row, col=1)
```

- [ ] **Step 4: Run and confirm they pass**

```bash
pytest tests/test_edf_plotting.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/edf_plotting.py tests/test_edf_plotting.py
git commit -m "feat(edf_plotting): make_freezoom_figure with FigureResampler"
```

---

## Task 8: `edf_plotting.make_epoch_figure`

30-second epoch viewer — no LTTB needed, just slice the arrays. Adds a minimap row showing the hypnogram with a current-position marker.

**Files:**
- Modify: `src/edf_plotting.py` (append)
- Modify: `tests/test_edf_plotting.py` (append)

- [ ] **Step 1: Write the failing smoke tests**

Append to `tests/test_edf_plotting.py`:

```python
def test_epoch_figure_first_epoch_builds():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {0: edf_loader.load_signal(SYNTH_PSG, 0)}
    fig = edf_plotting.make_epoch_figure(
        meta, signals, hypno=None, channels=[0], epoch_idx=0,
    )
    assert fig.data


def test_epoch_figure_with_hypno_builds():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {0: edf_loader.load_signal(SYNTH_PSG, 0)}
    hypno = hypnogram.load_hypnogram(SYNTH_HYP)
    fig = edf_plotting.make_epoch_figure(
        meta, signals, hypno=hypno, channels=[0], epoch_idx=0,
    )
    assert fig.data
```

- [ ] **Step 2: Run and confirm they fail**

```bash
pytest tests/test_edf_plotting.py -v
```

Expected: 2 new failures (`AttributeError: module 'src.edf_plotting' has no attribute 'make_epoch_figure'`).

- [ ] **Step 3: Implement `make_epoch_figure`**

Append to `src/edf_plotting.py`:

```python
def make_epoch_figure(
    meta: EdfMeta,
    signals: dict[int, np.ndarray],
    hypno: list[HypnogramEpoch] | None,
    channels: Iterable[int],
    epoch_idx: int,
    epoch_sec: float = 30.0,
) -> go.Figure:
    """30-second epoch view (PSG clinical standard).

    No LTTB — one epoch holds at most ~10k samples per channel which Plotly
    renders fine. A minimap row at the bottom shows the full-night hypnogram
    plus a vertical marker at the current epoch.
    """
    chans = [c for c in channels if c in signals]
    if not chans:
        return go.Figure()

    t0 = epoch_idx * epoch_sec
    t1 = t0 + epoch_sec

    has_minimap = bool(hypno)
    total_rows = len(chans) + (1 if has_minimap else 0)
    row_heights = [1.0] * len(chans) + ([0.35] if has_minimap else [])
    titles = [meta.channels[c].label for c in chans] + (
        ["Hypnogram (current epoch marked)"] if has_minimap else []
    )

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=False,  # minimap uses full-night x-axis, channels use epoch slice
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    for row_idx, ch in enumerate(chans, start=1):
        info = meta.channels[ch]
        sig = signals[ch]
        fs = max(info.sample_rate, 1.0)
        a = int(t0 * fs)
        b = int(min(len(sig), t1 * fs))
        t = np.arange(a, b, dtype=np.float64) / fs
        fig.add_trace(
            go.Scatter(
                x=t, y=sig[a:b], mode="lines",
                line=dict(width=1.0,
                          color=CHANNEL_GROUP_COLORS.get(info.group,
                                                         CHANNEL_GROUP_COLORS["Other"])),
                name=info.label, showlegend=False,
            ),
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text=info.physical_dim or "", row=row_idx, col=1)
        fig.update_xaxes(range=[t0, t1], row=row_idx, col=1)

    if has_minimap:
        _add_hypnogram_minimap(fig, hypno, current_t=t0,
                               total_dur=meta.duration_sec, row=total_rows)

    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    fig.update_layout(
        height=180 * len(chans) + (80 if has_minimap else 0) + 80,
        margin=dict(l=55, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _add_hypnogram_minimap(
    fig: go.Figure, epochs: list[HypnogramEpoch],
    current_t: float, total_dur: float, row: int,
) -> None:
    """Render full-night hypnogram band + a vertical marker at current_t."""
    xref = f"x{row}"
    yref = f"y{row} domain"
    for e in epochs:
        fig.add_shape(
            type="rect", xref=xref, yref=yref,
            x0=e.start_sec, x1=e.start_sec + e.duration_sec,
            y0=0, y1=1,
            fillcolor=_STAGE_COLORS.get(e.stage, _STAGE_COLORS[Stage.UNK]),
            line=dict(width=0), layer="below",
        )
    # Current-epoch marker
    fig.add_shape(
        type="line", xref=xref, yref=yref,
        x0=current_t, x1=current_t, y0=0, y1=1,
        line=dict(color="black", width=2),
    )
    fig.update_yaxes(visible=False, row=row, col=1)
    fig.update_xaxes(range=[0, total_dur], row=row, col=1)
```

- [ ] **Step 4: Run and confirm they pass**

```bash
pytest tests/test_edf_plotting.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full suite once more to verify no regressions**

```bash
pytest -v
```

Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add src/edf_plotting.py tests/test_edf_plotting.py
git commit -m "feat(edf_plotting): make_epoch_figure with hypnogram minimap"
```

---

## Task 9: Streamlit page `pages/2_EDF_Viewer.py`

UI assembly. Tested only via the manual checklist below — no automated test (per spec section 8).

**Files:**
- Create: `pages/__init__.py` (Streamlit doesn't strictly need it, but pytest collection benefits)
- Create: `pages/2_EDF_Viewer.py`

- [ ] **Step 1: Add a packaging marker for `pages/`**

Create `pages/__init__.py`:

```python
```

(empty file)

- [ ] **Step 2: Write the page**

Create `pages/2_EDF_Viewer.py`:

```python
"""EDF Viewer (PSG) — independent page in the Soom Streamlit app.

Loads `*.edf` files from the repo's CAPA_Data folder, lets the user pick
channels and a viewing mode (free zoom / 30-second epoch), and overlays
the paired Hypnogram annotation when available.

Spec: docs/specs/2026-04-27-edf-viewer-design.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import edf_loader, edf_plotting, hypnogram

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "CAPA_Data"

st.set_page_config(
    page_title="EDF Viewer · Soom",
    page_icon="🧠",
    layout="wide",
)

# Same compact CSS as app.py — Streamlit multipage applies CSS per page.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      h1 { font-size: 1.4rem !important; margin-bottom: 0.3rem; }
      h2 { font-size: 1.05rem !important; margin-top: 0.8rem; margin-bottom: 0.3rem; }
      h3 { font-size: 0.95rem !important; }
      .stMarkdown p, .stCaption, label { font-size: 0.85rem !important; }
      [data-testid="stCaptionContainer"] { font-size: 0.78rem !important; }
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] label { font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _list_files_cached(root_str: str) -> list[str]:
    return [str(p) for p in edf_loader.list_edf_files(Path(root_str))]


@st.cache_data(show_spinner="📥 EDF 헤더 읽는 중...")
def _load_meta_cached(path_str: str):
    return edf_loader.load_meta(Path(path_str))


@st.cache_data(show_spinner=False, max_entries=10)
def _load_signal_cached(path_str: str, ch_idx: int) -> np.ndarray:
    return edf_loader.load_signal(Path(path_str), ch_idx)


@st.cache_data(show_spinner=False)
def _load_hypnogram_cached(path_str: str):
    return hypnogram.load_hypnogram(Path(path_str))


# ─────────────────────────────────────────────────────────────────────
# Sidebar — file + channel + mode selectors
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 EDF Viewer")
st.sidebar.caption("PSG 표준 EDF/EDF+ · v1")

paths = _list_files_cached(str(DATA_ROOT))
if not paths:
    st.sidebar.error(f"EDF 파일을 찾을 수 없습니다: {DATA_ROOT}")
    st.info("`CAPA_Data/` 폴더에 `*.edf` 파일을 넣어주세요. (`*-Hypnogram.edf`는 자동 페어링되며 목록에 표시되지 않습니다.)")
    st.stop()

selected_path_str = st.sidebar.selectbox(
    "EDF 파일",
    options=paths,
    format_func=lambda p: Path(p).name,
)

try:
    meta = _load_meta_cached(selected_path_str)
except (OSError, ValueError, RuntimeError) as e:
    st.error(f"EDF 헤더를 읽을 수 없습니다: {e}")
    st.stop()

# Spec §7 — large-file guard (warning only in v1; no slicing)
if meta.duration_sec > 86400 or len(meta.channels) > 50:
    st.warning(
        f"⚠️ 대용량 EDF: duration={meta.duration_sec / 3600:.1f}h, "
        f"channels={len(meta.channels)}. 렌더링이 느릴 수 있습니다."
    )

# Channel group toggles
all_groups = sorted({ch.group for ch in meta.channels})
st.sidebar.markdown("**채널 그룹 (Channel Groups)**")
group_state: dict[str, bool] = {}
for g in all_groups:
    group_state[g] = st.sidebar.checkbox(g, value=(g != "Other"), key=f"grp_{g}")

# Filtered individual-channel multiselect
all_ch_indices = [i for i, ch in enumerate(meta.channels) if group_state.get(ch.group, False)]
selected_ch_indices = st.sidebar.multiselect(
    "표시할 채널",
    options=all_ch_indices,
    default=all_ch_indices,
    format_func=lambda i: f"{meta.channels[i].label} ({meta.channels[i].group}, {meta.channels[i].sample_rate:.0f} Hz)",
)

# Mode toggle
st.sidebar.divider()
mode = st.sidebar.radio(
    "보기 모드",
    options=["자유 줌 (Free Zoom)", "30s 에포크 (Epoch)"],
    index=0,
)

# Epoch slider (only in epoch mode)
if mode.startswith("30s"):
    total_epochs = max(1, int(meta.duration_sec // 30))
    epoch_idx = st.sidebar.slider(
        "에포크 (Epoch)",
        min_value=0, max_value=total_epochs - 1, value=0, step=1,
        format="%d",
    )
else:
    epoch_idx = 0

st.sidebar.divider()
st.sidebar.caption("ⓘ Hypnogram 파일이 같은 폴더에 있으면 자동 페어링됩니다.")


# ─────────────────────────────────────────────────────────────────────
# Header panel
# ─────────────────────────────────────────────────────────────────────
st.title("🧠 EDF Viewer (PSG)")
header_df = pd.DataFrame([{
    "파일": Path(meta.path).name,
    "환자 ID": meta.subject_id,
    "시작": meta.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "길이 (시간)": f"{meta.duration_sec / 3600:.2f}",
    "채널 수": len(meta.channels),
}])
st.dataframe(header_df, hide_index=True, use_container_width=True)

# Channel breakdown table
ch_df = pd.DataFrame([
    {"#": i, "라벨": ch.label, "그룹": ch.group,
     "샘플레이트 (Hz)": ch.sample_rate, "단위": ch.physical_dim,
     "샘플 수": ch.n_samples}
    for i, ch in enumerate(meta.channels)
])
with st.expander("채널 상세 (Channel details)"):
    st.dataframe(ch_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Hypnogram pairing
# ─────────────────────────────────────────────────────────────────────
hyp_path = edf_loader.pair_hypnogram(Path(selected_path_str))
hypno = None
if hyp_path is not None:
    try:
        hypno = _load_hypnogram_cached(str(hyp_path))
    except (OSError, ValueError, RuntimeError) as e:
        st.warning(f"Hypnogram 파일을 읽을 수 없어 오버레이를 비활성화합니다: {e}")
        hypno = None
    else:
        st.caption(f"✅ Hypnogram 페어링: `{hyp_path.name}` ({len(hypno)} epochs)")
else:
    st.caption("ℹ️ Hypnogram 파일이 없어 sleep stage 오버레이가 비활성화되었습니다.")


# ─────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 시계열 (Time-series)")

if not selected_ch_indices:
    st.warning("채널을 1개 이상 선택하세요.")
    st.stop()

# Lazy load only the selected channels
signals = {i: _load_signal_cached(selected_path_str, i)
           for i in selected_ch_indices}

if mode.startswith("자유"):
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=hypno, channels=selected_ch_indices,
    )
else:
    fig = edf_plotting.make_epoch_figure(
        meta, signals, hypno=hypno, channels=selected_ch_indices,
        epoch_idx=epoch_idx, epoch_sec=30.0,
    )

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "본 도구는 **시연·연구 목적**의 EDF/EDF+ 뷰어이며 임상 진단을 대체하지 않습니다. "
    "다운샘플링은 plotly-resampler의 LTTB 알고리즘(Steinarsson 2013)을 사용합니다."
)
```

- [ ] **Step 3: Run the full test suite (no regressions)**

```bash
pytest -v
```

Expected: every test passes; no test references `pages/`.

- [ ] **Step 4: Manual verification — start the app**

```bash
streamlit run app.py
```

Expected: the app starts. Open the URL Streamlit prints (typically `http://localhost:8501`).

- [ ] **Step 5: Manual verification — checklist**

In the running app, walk the checklist below. **Mark each item as you confirm it.**

- [ ] Sidebar shows two pages: the original Soom Analyzer (`app`) and **EDF Viewer**.
- [ ] Click **EDF Viewer**. The page loads without error.
- [ ] Sidebar selectbox shows `SC4001E0-PSG.edf` (and `*-Hypnogram.edf` is **NOT** in the list).
- [ ] Header panel renders with start time, duration ≈ 8 hours, channel count ≈ 7.
- [ ] "채널 상세" expander shows 7 channels with their sample rates and groups (EEG / EOG / EMG / Resp / SpO2).
- [ ] A `✅ Hypnogram 페어링: SC4001EC-Hypnogram.edf` caption appears below the channel table.
- [ ] By default, channel groups EEG/EOG/EMG/Resp/SpO2 are checked and "Other" is unchecked (if present).
- [ ] In **자유 줌** mode: charts render, drag-zoom on a region produces a smooth refresh (LTTB resampling). Hypnogram colored band is visible at the bottom.
- [ ] Switch to **30s 에포크** mode: a slider appears, charts now show a 30-second window. The minimap row at the bottom shows the full-night hypnogram with a black vertical marker at the current epoch. Moving the slider moves the marker.
- [ ] Uncheck all channel groups → "채널을 1개 이상 선택하세요" warning appears.
- [ ] Go back to the original Soom Analyzer page → it still works exactly as before.

- [ ] **Step 6: Commit**

```bash
git add pages/__init__.py pages/2_EDF_Viewer.py
git commit -m "feat: EDF Viewer Streamlit page (free zoom + 30s epoch)"
```

---

## Done

After Task 9, the EDF viewer is feature-complete per the v1 spec. Next steps (out of scope):

- v2: signal filters (band-pass / notch via `scipy.signal`)
- v2: range-based DataFrame export
- v2: user EDF upload via `st.file_uploader`
- v2: side-by-side multi-EDF comparison
