"""Package synthesized signals + ground truth + metadata into a ZIP archive."""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from signal_processing.generator import GroundTruth, ScenarioConfig

SCHEMA_VERSION = "1.0"

_README = """soom_Project Synthetic CPAP Pressure Signal — Validation Dataset
================================================================
1. Read signal.csv into your analysis tool
   -> Use 'pressure_cmh2o' column as the input signal (fs given in metadata.json)
2. Run your apnea/hypopnea detector on that single column
3. Compare your detections with ground_truth.csv
   (event_type, start_s, end_s) tuples are the labels
4. metadata.json describes the synthesis parameters

Note: 'flow_lpm' / 'flow_patient_lpm' are GROUND TRUTH only.
A real CPAP device with a single DP sensor cannot directly measure flow.
"""


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


_REQUIRED_SIGNAL_KEYS = frozenset(
    {"t_s", "pressure_cmh2o", "flow_lpm", "flow_patient_lpm", "blower_rpm"}
)


def _signal_dataframe(signals: dict) -> pd.DataFrame:
    missing = _REQUIRED_SIGNAL_KEYS - signals.keys()
    if missing:
        raise ValueError(f"signals dict missing keys: {sorted(missing)}")
    return pd.DataFrame({
        "time_s": signals["t_s"],
        "pressure_cmh2o": signals["pressure_cmh2o"],
        "flow_lpm": signals["flow_lpm"],
        "flow_patient_lpm": signals["flow_patient_lpm"],
        "blower_rpm": signals["blower_rpm"],
    })


def _ground_truth_dataframe(gt: GroundTruth) -> pd.DataFrame:
    rows = []
    for ev in gt.events:
        meta = dict(ev.metadata) if ev.metadata else {}
        rows.append({
            "event_type": ev.type,
            "start_s": ev.start_s,
            "end_s": ev.end_s,
            "duration_s": ev.duration_s,
            "severity": meta.pop("severity", ""),
            "metadata": json.dumps(meta) if meta else "",
        })
    return pd.DataFrame(rows, columns=[
        "event_type", "start_s", "end_s", "duration_s", "severity", "metadata",
    ])


def _build_metadata(cfg: ScenarioConfig, gt: GroundTruth, seed: int) -> dict:
    counts: dict[str, int] = {}
    for ev in gt.events:
        counts[ev.type] = counts.get(ev.type, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": int(seed),
        "fs_hz": float(cfg.fs_hz),
        "duration_s": float(cfg.duration_s),
        "scenario": {
            "rr_bpm": float(cfg.rr_bpm),
            "tv_ml": float(cfg.tv_ml),
            "ie_ratio": float(cfg.ie_ratio),
            "base_pressure_cmh2o": float(cfg.base_pressure_cmh2o),
            "intentional_leak_lpm": float(cfg.intentional_leak_lpm),
            "hr_bpm": float(cfg.heart_rate_bpm),
            "cardiogenic_amplitude_cmh2o": float(cfg.cardiogenic_amplitude_cmh2o),
            "measurement_noise_std_cmh2o": float(cfg.measurement_noise_std_cmh2o),
            "epr_enabled": bool(cfg.epr_enabled),
            "epr_relief_cmh2o": float(cfg.epr_relief_cmh2o),
            "power_line_50hz_amplitude_cmh2o": float(cfg.power_line_50hz_amplitude_cmh2o),
            "unintentional_leak_lpm": float(cfg.unintentional_leak_lpm),
            "unintentional_leak_profile": str(cfg.unintentional_leak_profile),
        },
        "event_summary": counts,
        "soom_version": _git_short_hash(),
        "spec_reference": "03_신호처리_사양서.md v0.4 §0.6",
    }


def build_zip(signals: dict, gt: GroundTruth, cfg: ScenarioConfig, seed: int) -> bytes:
    """Return ZIP bytes: signal.csv + ground_truth.csv + metadata.json + README.txt."""
    sig_df = _signal_dataframe(signals)
    gt_df = _ground_truth_dataframe(gt)
    meta = _build_metadata(cfg, gt, seed)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("signal.csv", sig_df.to_csv(index=False, lineterminator="\n"))
        z.writestr("ground_truth.csv", gt_df.to_csv(index=False, lineterminator="\n"))
        z.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))
        z.writestr("README.txt", _README)
    return buf.getvalue()


def filename_for(seed: int, ts: datetime | None = None) -> str:
    ts = ts or datetime.now()
    return f"soom_simulator_{ts.strftime('%Y-%m-%d_%H%M%S')}_{int(seed)}.zip"
