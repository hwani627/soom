"""Plotly figures for time-series + event overlays."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHANNEL_META = {
    "breathing":   {"col": "Breathing_Lpm",   "title": "Breathing",      "y": "Flow (L/min)",  "color": "#0F62FE"},
    "pressure":    {"col": "Pressure_cmH2O",  "title": "Pressure",       "y": "cmH₂O",         "color": "#DA1E28"},
    "epap":        {"col": "EPAP_cmH2O",      "title": "EPAP",           "y": "cmH₂O",         "color": "#FA4D56"},
    "leakrate":    {"col": "LeakRate_Lpm",    "title": "Leak Rate",      "y": "L/min",         "color": "#FF832B"},
    "flowlimit":   {"col": "FlowLimit",       "title": "Flow Limit",     "y": "FL idx",        "color": "#198038"},
    "snore":       {"col": "Snore",           "title": "Snore",          "y": "idx",           "color": "#6929C4"},
    "spo2":        {"col": "SpO2_pct",        "title": "SpO₂",           "y": "%",             "color": "#EE5396"},
    "pulserate":   {"col": "PulseRate_bpm",   "title": "Pulse Rate",     "y": "bpm",           "color": "#08BDBA"},
    "movement":    {"col": "Movement_idx",    "title": "Movement",       "y": "idx",           "color": "#33B1FF"},
    "resprate":    {"col": "RespRate_bpm",    "title": "Resp Rate",      "y": "br/min",        "color": "#9F1853"},
    "minutevent":  {"col": "MinuteVent_Lpm",  "title": "Minute Vent",    "y": "L/min",         "color": "#005D5D"},
    "tidalvolume": {"col": "TidalVolume_mL",  "title": "Tidal Volume",   "y": "mL",            "color": "#A56EFF"},
    "sleep_stage": {"col": "sleep_stage",     "title": "Sleep Stage",    "y": "stage",         "color": "#393939"},
}

# Default order for the multichannel display (top to bottom).
CHANNEL_ORDER = (
    "breathing", "pressure", "leakrate", "flowlimit", "snore",
    "spo2", "pulserate", "movement",
    "tidalvolume", "resprate", "minutevent",
    "sleep_stage",
)

OVERLAY_COLORS = {
    "aasm":    "rgba(220,30,30,0.38)",       # crimson
    "resmed":  "rgba(138,63,252,0.38)",      # purple (avoids clash with Breathing's blue)
    "sleephq": "rgba(255,176,0,0.38)",       # amber (high visibility on white bg)
}

OVERLAY_LEGEND_COLORS = {
    "aasm":    "rgba(220,30,30,0.95)",
    "resmed":  "rgba(138,63,252,0.95)",
    "sleephq": "rgba(255,176,0,0.95)",
}


def _build_event_shapes(events: pd.DataFrame, color: str) -> list[dict]:
    if events is None or events.empty:
        return []
    return [
        dict(
            type="rect", xref="x", yref="paper",
            x0=row["start_ts"], x1=row["end_ts"], y0=0, y1=1,
            fillcolor=color, opacity=1.0, line=dict(width=0), layer="below",
        )
        for _, row in events.iterrows()
    ]


def _legend_dummy(method: str, label: str) -> go.Scatter:
    legend_color = OVERLAY_LEGEND_COLORS.get(method, OVERLAY_COLORS.get(method, "rgba(0,0,0,0.8)"))
    return go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=legend_color, symbol="square"),
        name=label, showlegend=True,
    )


def _downsample(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    """Stride-based downsampling for fast SVG rendering on dense series (e.g. Breathing)."""
    if df is None or len(df) <= max_points:
        return df
    stride = len(df) // max_points + 1
    return df.iloc[::stride].reset_index(drop=True)


def plot_timeseries(
    df: Optional[pd.DataFrame],
    channel: str,
    overlays: Optional[dict[str, pd.DataFrame]] = None,
) -> go.Figure:
    """Single-channel chart (kept for backward compatibility / individual exports)."""
    meta = CHANNEL_META[channel]
    fig = go.Figure()
    if df is not None and not df.empty and meta["col"] in df.columns:
        plot_df = _downsample(df, max_points=8000)
        fig.add_trace(go.Scatter(
            x=plot_df["Timestamp_ET"], y=plot_df[meta["col"]],
            mode="lines", line=dict(width=1, color=meta["color"]),
            name=meta["title"], showlegend=False,
        ))
    if overlays:
        all_shapes: list[dict] = []
        for method, evs in overlays.items():
            if method not in OVERLAY_COLORS:
                continue
            all_shapes.extend(_build_event_shapes(evs, OVERLAY_COLORS[method]))
            fig.add_trace(_legend_dummy(method, method.upper()))
        if all_shapes:
            fig.update_layout(shapes=all_shapes)
    fig.update_layout(
        title=meta["title"],
        xaxis_title="Time",
        yaxis_title=meta["y"],
        height=380,
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_multichannel(
    session: dict,
    overlays: Optional[dict[str, pd.DataFrame]] = None,
    channels: Optional[tuple[str, ...]] = None,
    height_per_row: int = 180,
) -> go.Figure:
    """Stacked multi-row chart with shared x-axis.

    `channels` lets the caller pick the subset/order of rows to display.
    Event overlays are applied to every row so a vertical band crosses all panels.
    """
    chans = tuple(channels) if channels else CHANNEL_ORDER
    chans = tuple(c for c in chans if c in CHANNEL_META)
    if not chans:
        return go.Figure()

    titles = [f"{CHANNEL_META[c]['title']}  ({CHANNEL_META[c]['y']})" for c in chans]
    fig = make_subplots(
        rows=len(chans), cols=1,
        shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=titles,
    )

    for i, ch in enumerate(chans, start=1):
        meta = CHANNEL_META[ch]
        df = session.get(ch)
        if df is not None and not df.empty and meta["col"] in df.columns:
            plot_df = _downsample(df, max_points=8000)
            mode = "lines"
            line_shape = "hv" if ch == "sleep_stage" else "linear"
            fig.add_trace(
                go.Scatter(
                    x=plot_df["Timestamp_ET"], y=plot_df[meta["col"]],
                    mode=mode, line=dict(width=1.2, color=meta["color"], shape=line_shape),
                    name=meta["title"], showlegend=False,
                    hovertemplate=f"%{{x}}<br>%{{y}} {meta['y']}<extra></extra>",
                ),
                row=i, col=1,
            )
        fig.update_yaxes(title_text=meta["y"], row=i, col=1)

    if overlays:
        all_shapes: list[dict] = []
        for method, evs in overlays.items():
            if method not in OVERLAY_COLORS or evs is None or evs.empty:
                continue
            for r in range(1, len(chans) + 1):
                xref = "x" if r == 1 else f"x{r}"
                yref = "y domain" if r == 1 else f"y{r} domain"
                for _, row in evs.iterrows():
                    all_shapes.append(dict(
                        type="rect", xref=xref, yref=yref,
                        x0=row["start_ts"], x1=row["end_ts"], y0=0, y1=1,
                        fillcolor=OVERLAY_COLORS[method], opacity=1.0,
                        line=dict(width=0), layer="below",
                    ))
            fig.add_trace(_legend_dummy(method, method.upper()), row=1, col=1)
        if all_shapes:
            fig.update_layout(shapes=all_shapes)

    fig.update_xaxes(title_text="Time", row=len(chans), col=1)
    fig.update_layout(
        height=height_per_row * len(chans) + 80,
        margin=dict(l=55, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def sleephq_events_as_overlay(events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Convert the parsed SleepHQ events DataFrame into the overlay shape
    (columns: start_ts, end_ts) used by `plot_multichannel`."""
    if events is None or events.empty:
        return None
    apnea_hypop = {"CA", "OA", "MA", "H"}
    sub = events[events["event_type"].isin(apnea_hypop)].copy()
    if sub.empty:
        return None
    return sub[["start_ts", "end_ts"]].reset_index(drop=True)
