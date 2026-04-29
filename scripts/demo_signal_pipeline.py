"""End-to-end demo of the 7-Stage signal processing pipeline.

Run with:
    python scripts/demo_signal_pipeline.py

Generates a 5-minute synthetic CPAP session with controlled events
(2× OA, 1× CA, 1× hypopnea, 1× snore episode), pushes it through
Stage 2~6 of the pipeline (filters → flow estimation → breath analysis
→ event detection → subtype classification), then prints accuracy
metrics and saves a 6-subplot diagnostic figure.

Output files:
    soom_Project/demo_pipeline_output.png   — 6-subplot visualization
    (console output: AHI, sensitivity, OA/CA classification accuracy, RMSE)

References
----------
* 03_신호처리_사양서.md v0.4 §0.6 (7-Stage view)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is importable when the script is run from project root or scripts/.
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signal_processing import (  # noqa: E402
    breath_analyzer,
    event_detector,
    filters,
    flow_estimator,
    generator,
)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(seed: int = 42) -> dict:
    """Run Stage 0~6 of the pipeline. Returns a result bundle for plotting."""
    fs = 100.0

    # -------- Stage 0 — Synthesize input signals --------
    cfg = generator.default_demo_scenario()
    signals, gt = generator.synthesize_session(cfg, seed=seed)

    pressure = signals["pressure_cmh2o"]
    blower_rpm = signals["blower_rpm"]
    flow_truth = signals["flow_patient_lpm"]   # ground truth (validation only)
    t = signals["t_s"]

    # -------- Stage 2 — Refine + 4-branch BPF --------
    refined = filters.stage2_refine(pressure, fs_in=fs, fs_out=fs, notch_hz=50.0)
    branches = filters.apply_4branch_bpf(refined, fs=fs)

    # -------- Stage 3 — Flow derivation (Phase 0 blower-inverse) --------
    lookup = flow_estimator.BlowerLookupTable.synthetic_default()
    flow_est, flow_conf = flow_estimator.estimate_flow_phase0(
        refined, blower_rpm, fs=fs, lookup=lookup,
        intentional_leak_lpm=cfg.intentional_leak_lpm,
    )

    # Flow estimation accuracy (validation only, never available in production)
    flow_rmse_lpm = float(np.sqrt(np.mean((flow_est - flow_truth) ** 2)))
    flow_rmse_pct = flow_rmse_lpm / max(np.std(flow_truth), 1e-6) * 100.0

    # -------- Stage 4 — Breath analysis --------
    stage4 = breath_analyzer.stage4_analyze(flow_est, fs=fs)
    envelope = stage4["envelope"]
    baseline = stage4["baseline"]
    cycles = stage4["cycles"]
    rr_info = stage4["rr"]

    # -------- Stage 5/6 — Event detection + subtype classification --------
    apneas_raw = event_detector.detect_apnea(envelope, baseline, fs=fs)
    cardio_branch = branches["cardiogenic"]

    # Apply L2 Cardiogenic dual policy: each apnea → OA or CA
    apneas_classified = [
        event_detector.classify_apnea_subtype(a, cardio_branch, fs=fs)
        for a in apneas_raw
    ]
    hypopneas = event_detector.detect_hypopnea(envelope, baseline, fs=fs)
    snores = event_detector.detect_snore(branches["snore"], fs=fs)

    all_events = apneas_classified + hypopneas + snores

    # -------- Stage 7 (subset) — Stats + scoring --------
    stats = event_detector.compute_session_stats(
        all_events, duration_s=cfg.duration_s
    )
    score_resp = event_detector.score_events_vs_ground_truth(
        all_events, gt, tolerance_s=5.0,
    )
    score_subtype = event_detector.score_subtype_classification(
        all_events, gt, tolerance_s=5.0,
    )

    return {
        "t": t,
        "pressure": pressure,
        "refined": refined,
        "branches": branches,
        "flow_truth": flow_truth,
        "flow_est": flow_est,
        "flow_rmse_lpm": flow_rmse_lpm,
        "flow_rmse_pct": flow_rmse_pct,
        "envelope": envelope,
        "baseline": baseline,
        "cycles": cycles,
        "rr_info": rr_info,
        "events_apnea": apneas_classified,
        "events_hypopnea": hypopneas,
        "events_snore": snores,
        "all_events": all_events,
        "stats": stats,
        "score_resp": score_resp,
        "score_subtype": score_subtype,
        "cfg": cfg,
        "gt": gt,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _shade_event(ax, ev, color, label=None):
    ax.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.2,
               label=label, zorder=0)


def make_figure(result: dict, save_path: Path) -> None:
    """6-subplot diagnostic figure."""
    fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
    t = result["t"]
    cfg = result["cfg"]
    gt = result["gt"]

    # ----- 1. Raw pressure with ground-truth event overlay -----
    ax = axes[0]
    ax.plot(t, result["pressure"], color="#1B6FB8", linewidth=0.6, label="Raw P")
    seen = set()
    for ev in gt.events:
        color_map = {"OA": "#E11D2C", "CA": "#1F77B4", "hypopnea": "#F3B100",
                     "snore": "#16A34A", "csr": "#9333EA"}
        color = color_map.get(ev.type, "#999")
        label = ev.type if ev.type not in seen else None
        seen.add(ev.type)
        ax.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.18,
                   label=f"GT {label}" if label else None)
    ax.set_ylabel("Pressure\n(cmH₂O)")
    ax.set_title(
        f"Stage 1 — Raw pressure with ground truth events "
        f"(synthetic, fs={cfg.fs_hz:.0f} Hz, {cfg.duration_s/60:.1f} min)"
    )
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 2. 4-branch BPF (overlay) -----
    ax = axes[1]
    branches = result["branches"]
    ax.plot(t, branches["breath"] - 9.5, color="#1B6FB8",
            linewidth=0.5, alpha=0.7, label="Breath (LPF 30Hz)")
    ax.plot(t, branches["cardiogenic"] * 4, color="#E11D2C",
            linewidth=0.5, alpha=0.7, label="Cardiogenic (×4)")
    ax.plot(t, branches["fot"] * 4, color="#F3B100",
            linewidth=0.5, alpha=0.7, label="FOT band (×4)")
    ax.plot(t, branches["snore"] * 2, color="#16A34A",
            linewidth=0.4, alpha=0.5, label="Snore band (×2)")
    ax.set_ylabel("Branch\noutputs")
    ax.set_title("Stage 2 — 4-branch BPF (single raw signal → 4 spectral bands)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 3. Flow truth vs estimated -----
    ax = axes[2]
    ax.plot(t, result["flow_truth"], color="#1B6FB8", linewidth=0.6,
            label="Flow ground truth")
    ax.plot(t, result["flow_est"], color="#E11D2C", linewidth=0.5,
            alpha=0.8, label="Flow estimate (Phase 0)")
    ax.set_ylabel("Flow\n(L/min)")
    ax.set_title(
        f"Stage 3 — Flow derivation "
        f"(Phase 0 blower-inverse, RMSE = {result['flow_rmse_lpm']:.1f} L/min)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 4. Envelope + baseline + ratio -----
    ax = axes[3]
    ax.plot(t, result["envelope"], color="#1B6FB8", linewidth=0.7,
            label="Envelope (D01)")
    ax.plot(t, result["baseline"], color="#F3B100", linewidth=0.7,
            label="Baseline (D02)")
    # threshold lines as fraction of baseline
    ax.plot(t, result["baseline"] * 0.10, "r--", linewidth=0.5, alpha=0.7,
            label="Apnea threshold (10%)")
    ax.plot(t, result["baseline"] * 0.70, "y--", linewidth=0.5, alpha=0.7,
            label="Hypopnea threshold (70%)")
    ax.set_ylabel("Envelope\n(L/min)")
    ax.set_title("Stage 4 — Flow envelope · baseline · AASM thresholds")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 5. Detected events overlay -----
    ax = axes[4]
    ax.plot(t, result["envelope"], color="#999", linewidth=0.5, alpha=0.6)
    seen_d = set()
    for ev in result["all_events"]:
        color_map = {"OA": "#E11D2C", "CA": "#1F77B4", "apnea": "#9333EA",
                     "hypopnea": "#F3B100", "snore": "#16A34A"}
        color = color_map.get(ev.type, "#999")
        lbl = f"Detected {ev.type}" if ev.type not in seen_d else None
        seen_d.add(ev.type)
        ax.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.35, label=lbl)
    ax.set_ylabel("Detected\nevents")
    ax.set_title(
        f"Stage 5/6 — Detected events with L2 Cardiogenic OA/CA classification"
    )
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- 6. Scorecard (text panel) -----
    ax = axes[5]
    ax.axis("off")
    stats = result["stats"]
    score_r = result["score_resp"]
    score_s = result["score_subtype"]
    rr = result["rr_info"]
    truth_apnea = len(gt.of_type("OA")) + len(gt.of_type("CA"))
    truth_hypop = len(gt.of_type("hypopnea"))
    text = (
        f"  SESSION RESULTS\n"
        f"  ───────────────────────────────────────────────────────────\n"
        f"  Duration: {stats['duration_h']*60:.1f} min   "
        f"AHI = {stats['AHI']:.1f}/h   "
        f"RR = {rr['rr']:.1f} bpm  (conf {rr['confidence']:.2f})\n"
        f"  Cycles found: {len(result['cycles'])}   "
        f"Flow estimation RMSE: {result['flow_rmse_lpm']:.2f} L/min "
        f"({result['flow_rmse_pct']:.1f}%)\n"
        f"\n"
        f"  GROUND TRUTH:    {truth_apnea} apnea + {truth_hypop} hypopnea + "
        f"{len(gt.of_type('snore'))} snore\n"
        f"  DETECTED:        {stats['apnea_count']} apnea + "
        f"{stats['hypopnea_count']} hypopnea + {stats['snore_count']} snore\n"
        f"\n"
        f"  RESPIRATORY-EVENT DETECTION (apnea+hypopnea):\n"
        f"     TP = {score_r['TP']}   FP = {score_r['FP']}   "
        f"FN = {score_r['FN']}   "
        f"Sensitivity = {score_r['sensitivity']:.1%}   "
        f"PPV = {score_r['PPV']:.1%}\n"
        f"\n"
        f"  L2 CARDIOGENIC OA/CA SUBTYPE CLASSIFICATION:\n"
        f"     Pairs scored = {score_s['pairs']}   "
        f"Correct = {score_s['correct']}   "
        f"Accuracy = {score_s['accuracy']:.1%}"
    )
    ax.text(0.01, 0.98, text, family="monospace", fontsize=9,
            verticalalignment="top", transform=ax.transAxes)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "soom_Project — Signal-processing pipeline demo (single ΔP sensor)",
        fontsize=12, y=0.995, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(save_path, dpi=120)
    print(f"[demo] Saved figure → {save_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(result: dict) -> None:
    cfg = result["cfg"]
    stats = result["stats"]
    score_r = result["score_resp"]
    score_s = result["score_subtype"]
    rr = result["rr_info"]

    print()
    print("=" * 68)
    print("  soom_Project Signal-Processing Pipeline Demo")
    print("=" * 68)
    print(f"  Synthetic session   : {cfg.duration_s/60:.1f} minutes "
          f"@ {cfg.fs_hz:.0f} Hz")
    print(f"  Scheduled events    : "
          f"{len(cfg.obstructive_apneas)} OA, "
          f"{len(cfg.central_apneas)} CA, "
          f"{len(cfg.hypopneas)} hypopnea, "
          f"{len(cfg.snore_episodes)} snore")
    print()
    print(f"  Flow estimation RMSE   : {result['flow_rmse_lpm']:.2f} L/min "
          f"({result['flow_rmse_pct']:.1f}%)")
    print(f"  Respiratory rate (RR)  : {rr['rr']:.1f} bpm "
          f"(confidence {rr['confidence']:.2f})")
    print(f"  Breath cycles detected : {len(result['cycles'])}")
    print()
    print(f"  AHI                    : {stats['AHI']:.1f}/h")
    print(f"  Detected counts        : "
          f"{stats['apnea_count']} apnea, "
          f"{stats['hypopnea_count']} hypopnea, "
          f"{stats['snore_count']} snore")
    print()
    print("  Respiratory event detection (vs ground truth):")
    print(f"     TP={score_r['TP']}  FP={score_r['FP']}  FN={score_r['FN']}")
    print(f"     Sensitivity = {score_r['sensitivity']:.1%}   "
          f"PPV = {score_r['PPV']:.1%}")
    print()
    print("  L2 Cardiogenic OA/CA subtype classification:")
    print(f"     Pairs={score_s['pairs']}  Correct={score_s['correct']}  "
          f"Accuracy={score_s['accuracy']:.1%}")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    result = run_pipeline(seed=42)
    print_summary(result)
    out_path = ROOT / "demo_pipeline_output.png"
    make_figure(result, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
