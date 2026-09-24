"""Shared chart theme (dataviz reference palette, validated): dismissed = slot 1 blue, escalated = slot 2 orange.

Figures are built with LIGHT colors; the website swaps each light hex for its dark step on theme change
(see DARK_MAP), so both themes use validated steps rather than an automatic inversion.
"""
from __future__ import annotations

import plotly.graph_objects as go

LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e6e5e1",
         "dismissed": "#2a78d6", "escalated": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
         "neutral": "#f0efec", "red": "#e34948"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f",
        "dismissed": "#3987e5", "escalated": "#d95926", "s3": "#199e70", "s4": "#c98500",
        "neutral": "#383835", "red": "#e66767"}
DARK_MAP = {LIGHT[k]: DARK[k] for k in LIGHT}
TYPE_COLORS = {"karta": LIGHT["dismissed"], "bank_otkazmasi": LIGHT["escalated"], "naqd": LIGHT["s3"],
               "xalqaro": LIGHT["s4"]}
TARGET_NAMES = {0: "Dismissed / Rad etilgan", 1: "Escalated / Eskalatsiya"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 420) -> go.Figure:
    """Recessive grid/axes, thin marks, legend on top, no dual axes."""
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": LIGHT["text"]}},
        paper_bgcolor=LIGHT["surface"], plot_bgcolor=LIGHT["surface"],
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "size": 13, "color": LIGHT["text2"]},
        margin={"l": 60, "r": 20, "t": 60, "b": 50}, height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        hoverlabel={"font": {"family": "IBM Plex Sans, system-ui, sans-serif"}}, bargap=0.25,
    )
    if fig.layout.annotations:  # subplot titles occupy the top edge -> legend goes under the plot
        fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
                          margin={"b": 90})
    fig.update_xaxes(title=x, gridcolor=LIGHT["grid"], zerolinecolor=LIGHT["grid"], linecolor=LIGHT["grid"])
    fig.update_yaxes(title=y, gridcolor=LIGHT["grid"], zerolinecolor=LIGHT["grid"], linecolor=LIGHT["grid"])
    return fig
