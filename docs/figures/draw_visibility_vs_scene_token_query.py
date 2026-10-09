#!/usr/bin/env python3
"""Draw a focused, paper-style comparison of the two spatial query branches."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
STEM = OUT_DIR / "visibility_vs_scene_token_query_architecture"

BG = "#F7F8FA"
WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#5D6975"
LINE = "#85919D"
SOFT = "#E7EBEF"
SHARED = "#EFF2F5"
ORANGE = "#D97A21"
ORANGE_DARK = "#8E4810"
ORANGE_LIGHT = "#FFF0DF"
TEAL = "#168B82"
TEAL_DARK = "#0A5B56"
TEAL_LIGHT = "#DFF4F1"


def box(ax, x, y, w, h, title, detail="", *, face=WHITE, edge=LINE, fs=9.2, lw=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.62 if detail else 0.50), title,
            ha="center", va="center", color=INK, fontsize=fs,
            fontweight="bold", zorder=5)
    if detail:
        ax.text(x + w / 2, y + h * 0.27, detail,
                ha="center", va="center", color=MUTED,
                fontsize=fs - 1.5, zorder=5)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=LINE, lw=1.45, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
    )
    ax.add_patch(patch)
    return patch


def matrix_icon(ax, x, y, w, h, *, color, rows=5, cols=7):
    """Small raster/token-grid glyph used instead of another text box."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=color,
                           linewidth=1.2, zorder=4))
    for i in range(1, cols):
        ax.plot([x + w * i / cols] * 2, [y, y + h], color=color,
                linewidth=0.45, alpha=0.55, zorder=4)
    for i in range(1, rows):
        ax.plot([x, x + w], [y + h * i / rows] * 2, color=color,
                linewidth=0.45, alpha=0.55, zorder=4)
    for r, c in ((0, 0), (1, 2), (2, 4), (3, 1), (4, 5)):
        if r < rows and c < cols:
            ax.add_patch(Rectangle(
                (x + w * c / cols, y + h * r / rows), w / cols, h / rows,
                facecolor=color, edgecolor="none", alpha=0.25, zorder=4,
            ))


def token_stack(ax, x, y, w, h, *, color):
    for offset in (0.16, 0.08, 0.0):
        ax.add_patch(Rectangle((x + offset, y + offset), w, h,
                               facecolor=WHITE, edgecolor=color,
                               linewidth=1.0, zorder=4))
    for i in range(1, 6):
        ax.plot([x, x + w], [y + h * i / 6] * 2, color=color,
                linewidth=0.45, alpha=0.6, zorder=5)


def row_background(ax, y, color, title, run, tag):
    ax.add_patch(FancyBboxPatch(
        (0.65, y), 16.70, 3.15,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=WHITE, edgecolor="#CCD3DA", linewidth=1.0, zorder=0,
    ))
    ax.add_patch(Rectangle((0.65, y), 0.12, 3.15, facecolor=color,
                           edgecolor="none", zorder=1))
    ax.text(0.98, y + 2.73, title, ha="left", va="center", color=color,
            fontsize=13.5, fontweight="bold")
    ax.text(0.98, y + 2.36, run, ha="left", va="center", color=MUTED,
            fontsize=7.8)
    ax.text(16.98, y + 2.73, tag, ha="right", va="center", color=color,
            fontsize=8.4, fontweight="bold")


def draw_visibility_row(ax):
    y = 5.62
    cy = y + 1.30
    row_background(
        ax, y, ORANGE,
        "A   Visibility Query",
        "2026-07-20_21-14-26_g1_sru_visibility_query_v1",
        "LEARNED DENSE-MAP ENCODING",
    )

    matrix_icon(ax, 1.10, cy - 0.46, 0.82, 0.92, color=ORANGE)
    ax.text(1.51, cy - 0.70, "Visibility BEV", ha="center", va="center",
            color=INK, fontsize=8.4, fontweight="bold")
    ax.text(1.51, cy - 0.93, "1 x 64 x 64  |  4096D", ha="center", va="center",
            color=MUTED, fontsize=7.2)

    box(ax, 2.43, cy - 0.52, 2.18, 1.04, "3-layer CNN",
        "1->16->32->64  |  stride 2", face=ORANGE_LIGHT, edge=ORANGE)
    arrow(ax, 1.95, cy, 2.43, cy, color=ORANGE)

    token_stack(ax, 5.16, cy - 0.43, 0.62, 0.78, color=ORANGE)
    ax.text(5.49, cy - 0.70, "CNN tokens", ha="center", va="center",
            color=INK, fontsize=8.4, fontweight="bold")
    ax.text(5.49, cy - 0.93, "64 x 64D", ha="center", va="center",
            color=MUTED, fontsize=7.2)
    arrow(ax, 4.61, cy, 5.13, cy, color=ORANGE)

    box(ax, 6.55, cy - 0.52, 2.08, 1.04, "Cross-attention",
        "4 heads  |  Q attends K,V", face=ORANGE_LIGHT, edge=ORANGE)
    arrow(ax, 5.95, cy, 6.55, cy, color=ORANGE)

    box(ax, 9.20, cy - 0.52, 1.70, 1.04, "Context", "64D", face=ORANGE_LIGHT, edge=ORANGE)
    arrow(ax, 8.63, cy, 9.20, cy, color=ORANGE)

    box(ax, 11.48, cy - 0.52, 2.12, 1.04, "Late fusion", "[h 512 ; context 64]",
        face=SHARED, edge=LINE)
    arrow(ax, 10.90, cy, 11.48, cy, color=ORANGE)

    box(ax, 14.16, cy - 0.52, 2.40, 1.04, "Actor head", "576->512->256->128->3",
        face=SHARED, edge=LINE)
    arrow(ax, 13.60, cy, 14.16, cy)

    box(ax, 6.64, cy + 0.73, 1.90, 0.62, "Shared query Q", "[h512 ; proprio16] -> 64D",
        face=SHARED, edge=LINE, fs=8.0)
    arrow(ax, 7.59, cy + 0.73, 7.59, cy + 0.52, color=LINE)

    ax.text(3.52, y + 0.20,
            "The policy learns spatial features directly from every BEV cell.",
            ha="center", va="center", color=ORANGE_DARK, fontsize=7.6)
    ax.text(16.94, y + 0.20, "branch 99,264 params  |  actor obs 6672D",
            ha="right", va="center", color=ORANGE_DARK, fontsize=7.6,
            fontweight="bold")


def draw_scene_row(ax):
    y = 1.74
    cy = y + 1.30
    row_background(
        ax, y, TEAL,
        "B   Scene Token Query v1",
        "2026-07-23_20-49-31_g1_sru_scene_token_query_v1",
        "ENGINEERED COMPACT REPRESENTATION",
    )

    matrix_icon(ax, 1.10, cy - 0.46, 0.82, 0.92, color=TEAL)
    ax.text(1.51, cy - 0.70, "Visibility BEV", ha="center", va="center",
            color=INK, fontsize=8.4, fontweight="bold")
    ax.text(1.51, cy - 0.93, "1 x 64 x 64", ha="center", va="center",
            color=MUTED, fontsize=7.2)

    box(ax, 2.30, cy - 0.52, 2.45, 1.04, "8 x 8 patch builder",
        "pool visibility + geometry", face=TEAL_LIGHT, edge=TEAL)
    arrow(ax, 1.95, cy, 2.30, cy, color=TEAL)

    token_stack(ax, 5.16, cy - 0.43, 0.62, 0.78, color=TEAL)
    ax.text(5.49, cy - 0.70, "Scene tokens", ha="center", va="center",
            color=INK, fontsize=8.4, fontweight="bold")
    ax.text(5.49, cy - 0.93, "64 x 6D", ha="center", va="center",
            color=MUTED, fontsize=7.2)
    arrow(ax, 4.75, cy, 5.13, cy, color=TEAL)

    box(ax, 6.28, cy - 0.52, 1.70, 1.04, "Token MLP", "6->64->64",
        face=TEAL_LIGHT, edge=TEAL)
    arrow(ax, 5.95, cy, 6.28, cy, color=TEAL)

    box(ax, 8.52, cy - 0.52, 2.08, 1.04, "Cross-attention",
        "4 heads  |  Q attends K,V", face=TEAL_LIGHT, edge=TEAL)
    arrow(ax, 7.98, cy, 8.52, cy, color=TEAL)

    box(ax, 11.05, cy - 0.52, 1.42, 1.04, "Context", "64D",
        face=TEAL_LIGHT, edge=TEAL)
    arrow(ax, 10.60, cy, 11.05, cy, color=TEAL)

    box(ax, 12.92, cy - 0.52, 1.70, 1.04, "Late fusion", "same as A",
        face=SHARED, edge=LINE)
    arrow(ax, 12.47, cy, 12.92, cy, color=TEAL)

    box(ax, 15.07, cy - 0.52, 1.49, 1.04, "Actor head", "same as A",
        face=SHARED, edge=LINE)
    arrow(ax, 14.62, cy, 15.07, cy)

    box(ax, 8.61, cy + 0.73, 1.90, 0.62, "Shared query Q", "[h512 ; proprio16] -> 64D",
        face=SHARED, edge=LINE, fs=8.0)
    arrow(ax, 9.56, cy + 0.73, 9.56, cy + 0.52, color=LINE)

    ax.text(4.20, y + 0.20,
            "6D/token = (x, y, visibility, goal distance, goal cos, goal sin)",
            ha="center", va="center", color=TEAL_DARK, fontsize=7.4)
    ax.text(16.94, y + 0.20, "branch 80,320 params  |  actor obs 2960D",
            ha="right", va="center", color=TEAL_DARK, fontsize=7.6,
            fontweight="bold")


def draw():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(0.68, 9.53, "Spatial Query Branch Ablation", ha="left", va="center",
            color=INK, fontsize=19, fontweight="bold")
    ax.text(0.69, 9.13,
            "The recurrent backbone and policy head are identical; only the highlighted spatial representation changes.",
            ha="left", va="center", color=MUTED, fontsize=10.0)
    ax.text(17.30, 9.40, "instinct_nav  |  July 2026", ha="right", va="center",
            color=MUTED, fontsize=8.3)

    draw_visibility_row(ax)
    draw_scene_row(ax)

    ax.text(0.76, 1.22, "KEY DIFFERENCE", ha="left", va="center", color=INK,
            fontsize=8.2, fontweight="bold")
    ax.text(2.28, 1.22,
            "dense learned CNN features (A)  vs.  compact, explicitly structured 6D tokens (B)",
            ha="left", va="center", color=MUTED, fontsize=9.0)
    ax.plot([0.68, 17.30], [0.88, 0.88], color=SOFT, linewidth=1.0)
    ax.text(0.68, 0.51,
            "Not shown: identical depth encoder, Actor SRU (80->512), critic, and frozen G1 low-level controller.",
            ha="left", va="center", color=MUTED, fontsize=8.2)
    ax.text(17.30, 0.51,
            "Both branches are queried after the SRU; neither is written into recurrent memory.",
            ha="right", va="center", color=MUTED, fontsize=8.2)

    plt.tight_layout(pad=0.3)
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=BG)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    draw()
