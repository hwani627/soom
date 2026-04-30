"""Multi-panel plotly figure for the Signal Simulator page."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signal_processing.generator import GroundTruth

EVENT_COLOR = {
    "OA": "#E11D2C",
    "CA": "#1F77B4",
    "MA": "#9333EA",
    "hypopnea": "#F3B100",
    "snore": "#16A34A",
    "cough": "#FB923C",
    "csr": "#7C3AED",
}


def _add_event_shapes(fig: go.Figure, gt: GroundTruth) -> None:
    for ev in gt.events:
        color = EVENT_COLOR.get(ev.type, "#888")
        fig.add_vrect(
            x0=ev.start_s, x1=ev.end_s,
            fillcolor=color, opacity=0.18,
            line_width=0, layer="below",
            annotation_text=ev.type, annotation_position="top left",
            annotation_font_size=10,
        )


def build_figure(signals: dict, gt: GroundTruth,
                 show_advanced: bool = False) -> go.Figure:
    """Build the Signal Simulator main plot.

    Basic: 2 panels (pressure, flow).
    Advanced: 4 panels (pressure, flow, RPM, 4-branch BPF overlay).
    """
    n_panels = 4 if show_advanced else 2
    titles = ["Pressure (cmH₂O)", "Flow ground truth (L/min)"]
    if show_advanced:
        titles += ["Blower RPM", "4-branch BPF (overlay)"]

    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=titles,
    )
    t = signals["t_s"]

    fig.add_trace(go.Scatter(x=t, y=signals["pressure_cmh2o"],
                             name="pressure", line=dict(color="#1B6FB8", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=signals["flow_patient_lpm"],
                             name="flow_patient", line=dict(color="#16A34A", width=1)),
                  row=2, col=1)

    if show_advanced:
        fig.add_trace(go.Scatter(x=t, y=signals["blower_rpm"],
                                 name="rpm", line=dict(color="#9333EA", width=1)),
                      row=3, col=1)
        from signal_processing import filters
        fs = float(1.0 / np.mean(np.diff(t)))
        refined = filters.stage2_refine(signals["pressure_cmh2o"], fs_in=fs, fs_out=fs)
        branches = filters.apply_4branch_bpf(refined, fs=fs)
        for name, color in [("breath", "#1B6FB8"), ("cardiogenic", "#E11D2C"),
                             ("fot", "#F3B100"), ("snore", "#16A34A")]:
            fig.add_trace(go.Scatter(x=t, y=branches[name],
                                     name=name, line=dict(color=color, width=1)),
                          row=4, col=1)

    _add_event_shapes(fig, gt)
    fig.update_layout(
        height=200 * n_panels + 80,
        showlegend=True,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
    return fig
