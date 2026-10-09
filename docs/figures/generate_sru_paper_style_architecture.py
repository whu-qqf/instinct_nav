"""Generate an InstinctNav architecture figure in the visual language of SRU Figure 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = Path(__file__).resolve().parent

INK = "#222222"
MUTED = "#666666"
PANEL = "#F6F6F6"
PRETRAIN = "#FFF7E5"
PRETRAIN_EDGE = "#E5D8B8"
DEPTH = "#66727B"
DEPTH_LIGHT = "#D9DEE2"
QUERY = "#EA756B"
QUERY_LIGHT = "#F8C4BE"
DYNAMIC = "#D88419"
DYNAMIC_LIGHT = "#FFF0D2"
MAP = "#3A7896"
MAP_LIGHT = "#DFEEF4"
SRU = "#8C79A5"
HEAD = "#8196AD"
CONTROL = "#4D8C66"
CRITIC = "#777777"


def rounded_panel(ax, x, y, w, h, face, edge="#D0D0D0", radius=0.30, lw=1.0, dashed=False, z=0):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=INK, lw=1.7, dashed=False, rad=0.0, z=4, scale=12):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def route(ax, points, color=INK, lw=1.5, dashed=False, z=3):
    style = "--" if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=style, zorder=z)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed, z=z)


def vertical_module(ax, x, y, w, h, label, face, edge=None, text_color="white", fontsize=9.0):
    edge = edge or face
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
        zorder=5,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        rotation=90,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        zorder=6,
    )
    return p


def small_box(ax, x, y, w, h, title, detail="", face="white", edge="#777777", title_size=8.8, detail_size=6.7):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
        zorder=5,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * (0.63 if detail else 0.5), title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK, zorder=6)
    if detail:
        ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", fontsize=detail_size, color=MUTED, linespacing=1.05, zorder=6)
    return p


def cuboid(ax, x, y, w, h, depth=0.14, face="#CED5DA", edge="#666666", label=None, label_size=7.0, z=5):
    front = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.9, zorder=z)
    top = Polygon([(x, y + h), (x + depth, y + h + depth), (x + w + depth, y + h + depth), (x + w, y + h)], facecolor="white", edgecolor=edge, linewidth=0.8, zorder=z)
    side = Polygon([(x + w, y), (x + w + depth, y + depth), (x + w + depth, y + h + depth), (x + w, y + h)], facecolor="#B7C0C6", edgecolor=edge, linewidth=0.8, zorder=z)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)
    if label:
        ax.text(x + w / 2, y - 0.12, label, ha="center", va="top", fontsize=label_size, color=MUTED, zorder=z + 1)
    return front


def feature_stack(ax, x, y, cols=4, rows=5, cell=0.10, depth=0.06, color="#D9DEE2"):
    for k in range(3, -1, -1):
        ox = x + k * depth
        oy = y + k * depth
        for i in range(cols):
            for j in range(rows):
                shade = 0.92 - 0.05 * ((i + j + k) % 3)
                fc = tuple(np.array(plt.matplotlib.colors.to_rgb(color)) * shade)
                ax.add_patch(Rectangle((ox + i * cell, oy + j * cell), cell, cell, facecolor=fc, edgecolor="#8A8A8A", linewidth=0.35, zorder=5 + k))


def depth_thumbnail(ax, x, y, w, h, offset=0.0, z=5):
    xx, yy = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 70))
    img = 0.25 + 0.75 * (1.0 - yy)
    img -= 0.5 * np.exp(-((xx - 0.52 - offset) ** 2 / 0.018 + (yy - 0.55) ** 2 / 0.05))
    img -= 0.28 * (np.abs(xx - 0.18) < 0.05) * (yy > 0.25)
    ax.imshow(img, extent=(x, x + w, y, y + h), origin="lower", cmap="cividis", vmin=0, vmax=1, zorder=z, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#777777", linewidth=0.7, zorder=z + 1))


def bev_grid(ax, x, y, size, dynamic=True, z=6):
    n = 8
    cell = size / n
    for i in range(n):
        for j in range(n):
            val = 0.17 + 0.65 * ((i * 3 + j * 5) % 13) / 12
            fc = plt.cm.Blues(val)
            ax.add_patch(Rectangle((x + i * cell, y + j * cell), cell - 0.004, cell - 0.004, facecolor=fc, edgecolor="white", linewidth=0.25, zorder=z))
    if dynamic:
        for i, j in ((2, 5), (3, 5), (5, 2), (6, 2)):
            ax.add_patch(Rectangle((x + i * cell, y + j * cell), cell - 0.004, cell - 0.004, facecolor=DYNAMIC, edgecolor="white", linewidth=0.25, zorder=z + 1))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.35, 6.75, "SRU Navigation with Dynamic BEV Recurrent Fusion", fontsize=18, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(17.65, 6.75, "InstinctNav · G1", fontsize=9.5, fontweight="bold", color=SRU, ha="right", va="center")

    # Paper-like major panels.
    rounded_panel(ax, 0.35, 1.25, 4.05, 5.15, PRETRAIN, PRETRAIN_EDGE, radius=0.38)
    rounded_panel(ax, 4.55, 1.25, 13.10, 5.15, PANEL, "#D5D5D5", radius=0.38)
    ax.text(0.68, 6.12, "Pretraining Encoder–Decoder", color="#A84820", fontsize=10, fontstyle="italic", fontweight="bold")
    ax.text(4.83, 6.12, "BASE SRU VISUAL PATH", color=MUTED, fontsize=8.2, fontweight="bold")
    ax.text(17.20, 1.47, "Dynamic Navigation RL Training", color=MUTED, fontsize=10, fontstyle="italic", fontweight="bold", ha="right")

    # Pretraining panel: visually follows the original encoder-decoder composition.
    for k in range(3, -1, -1):
        depth_thumbnail(ax, 0.72 + 0.08 * k, 4.15 + 0.08 * k, 0.88, 0.72, offset=0.015 * k, z=5 + k)
    ax.text(1.20, 3.91, "synthetic depth + sensor noise", fontsize=7.0, color=MUTED, ha="center")

    # RegNet stages.
    stages = [(1.85, 4.12, 0.13, 1.22), (2.05, 4.22, 0.17, 1.02), (2.29, 4.32, 0.21, 0.82), (2.58, 4.42, 0.25, 0.62)]
    for x, y, w, h in stages:
        cuboid(ax, x, y, w, h, depth=0.09, face="#F7F7F7", edge="#555555", z=7)
    ax.text(2.33, 5.55, "RegNet", fontsize=9, color=MUTED, fontstyle="italic", fontweight="bold", ha="center")

    # FPN pyramid and reconstruction branch.
    for idx, (w, y) in enumerate(((1.15, 4.92), (0.92, 4.62), (0.69, 4.32))):
        cuboid(ax, 3.02 + idx * 0.08, y, w, 0.12, depth=0.08, face="#FCFCFC", edge="#555555", z=7)
    ax.text(3.55, 5.50, "FPN", fontsize=9, color=MUTED, fontstyle="italic", fontweight="bold", ha="center")
    arrow(ax, (1.65, 4.65), (1.85, 4.70), lw=1.5)
    arrow(ax, (2.86, 4.72), (3.05, 4.72), lw=1.5)
    route(ax, [(3.65, 4.30), (3.65, 3.36), (2.73, 3.36)], color=INK, lw=1.2)
    for i in range(4):
        cuboid(ax, 2.05 + 0.09 * i, 2.52 + 0.09 * i, 1.05, 0.16, depth=0.06, face="#EEEEEE", edge="#666666", z=6 + i)
    arrow(ax, (2.10, 2.68), (1.38, 2.68), color=INK, lw=1.2)
    ax.text(1.20, 2.68, "Lrec", fontsize=8.2, fontweight="bold", ha="center", va="center")
    ax.text(3.91, 5.95, "❄", fontsize=20, color="#67BCE8", ha="center", va="center")
    ax.text(2.42, 1.70, "Pretrained and frozen during PPO", fontsize=8.0, color=MUTED, ha="center")

    # Baseline SRU path in the navigation panel.
    feature_stack(ax, 4.92, 4.45, cols=4, rows=6, cell=0.105, depth=0.055, color=DEPTH_LIGHT)
    ax.text(5.23, 4.18, "Fₜ  64×5×8", fontsize=7.0, color=MUTED, ha="center")
    arrow(ax, (4.12, 4.85), (4.92, 4.85), lw=1.7)

    vertical_module(ax, 5.62, 4.18, 0.38, 1.64, "Self-Attention", DEPTH, fontsize=8.0)
    arrow(ax, (5.42, 4.86), (5.62, 4.86), lw=1.4)
    feature_stack(ax, 6.16, 4.45, cols=4, rows=6, cell=0.105, depth=0.055, color=DEPTH_LIGHT)
    ax.text(6.47, 4.18, "F′ₜ", fontsize=7.5, color=MUTED, ha="center")
    arrow(ax, (6.00, 4.86), (6.16, 4.86), lw=1.4)

    vertical_module(ax, 6.86, 4.18, 0.38, 1.64, "Cross-Attention", DEPTH, fontsize=8.0)
    arrow(ax, (6.66, 4.86), (6.86, 4.86), lw=1.4)
    cuboid(ax, 7.47, 4.48, 0.25, 0.88, depth=0.13, face="#9CA3A8", edge="#5F666A", label="depth context 64D", label_size=6.6)
    arrow(ax, (7.24, 4.86), (7.47, 4.86), lw=1.4)

    # State query, echoing the red query cube and dashed path from the paper.
    cuboid(ax, 6.18, 3.72, 0.23, 0.25, depth=0.10, face=QUERY_LIGHT, edge=QUERY, z=7)
    ax.text(6.30, 3.48, "oᵖʳᵒᵖₜ ∥ goalₜ\n16D", fontsize=6.8, color=MUTED, ha="center")
    route(ax, [(6.42, 3.86), (6.70, 3.86), (6.70, 4.46), (6.86, 4.46)], color=QUERY, lw=1.2, dashed=True, z=4)
    ax.text(6.67, 4.05, "Query", fontsize=6.6, color=QUERY, ha="right")

    # Pre-SRU concat, SRU, late scene query, and policy head.
    small_box(ax, 8.03, 4.35, 0.90, 1.10, "Concat", "state 16\ndepth 64\ndynamic 64\n= 144D", face="white", edge="#777777", title_size=8.2, detail_size=6.1)
    arrow(ax, (7.85, 4.86), (8.03, 4.86), lw=1.4)
    vertical_module(ax, 9.18, 4.03, 0.48, 1.92, "LSTM–SRU  ·  h=512", SRU, fontsize=8.2)
    arrow(ax, (8.93, 4.86), (9.18, 4.86), lw=1.5)
    route(ax, [(9.42, 4.02), (9.42, 3.72), (9.83, 3.72), (9.83, 5.48), (9.66, 5.48)], color=INK, lw=1.25, dashed=True, z=3)
    ax.text(9.85, 4.62, "hₜ₋₁", fontsize=7.0, color=MUTED, rotation=90, va="center")
    cuboid(ax, 9.96, 4.50, 0.58, 0.66, depth=0.12, face="#4D5052", edge="#333333", label="hₜ  512D", label_size=6.7)
    arrow(ax, (9.66, 4.86), (9.96, 4.86), lw=1.5)

    vertical_module(ax, 10.82, 4.03, 0.48, 1.92, "Scene Query Attention", MAP, fontsize=7.8)
    arrow(ax, (10.67, 4.86), (10.82, 4.86), lw=1.5)
    cuboid(ax, 11.58, 4.50, 0.36, 0.66, depth=0.12, face="#5D7886", edge="#344C58", label="scene 64D", label_size=6.7)
    arrow(ax, (11.30, 4.86), (11.58, 4.86), lw=1.5)
    small_box(ax, 12.20, 4.35, 0.82, 1.10, "Concat", "hₜ 512\nscene 64\n= 576D", face="white", edge="#777777", title_size=8.0, detail_size=6.2)
    arrow(ax, (12.06, 4.86), (12.20, 4.86), lw=1.4)
    vertical_module(ax, 13.28, 4.03, 0.50, 1.92, "MLP + TC-Dropout", HEAD, fontsize=8.0)
    arrow(ax, (13.02, 4.86), (13.28, 4.86), lw=1.5)
    small_box(ax, 14.05, 4.43, 0.75, 0.88, "aₜ", "SE(2)\n3D", face="white", edge="#777777", title_size=10, detail_size=6.4)
    arrow(ax, (13.78, 4.86), (14.05, 4.86), lw=1.5)
    small_box(ax, 15.07, 4.24, 1.25, 1.25, "Frozen Instinct", "Official Parkour\nONNX · 50 Hz", face="#E8F4EC", edge=CONTROL, title_size=8.5, detail_size=6.5)
    arrow(ax, (14.80, 4.86), (15.07, 4.86), color=CONTROL, lw=1.6)
    cuboid(ax, 16.65, 4.48, 0.34, 0.76, depth=0.14, face="#77A88A", edge=CONTROL, label="29 joint targets", label_size=6.7)
    arrow(ax, (16.32, 4.86), (16.65, 4.86), color=CONTROL, lw=1.6)

    # Our dynamic-scene extension in a distinct inset, while retaining the paper's style.
    rounded_panel(ax, 4.82, 1.68, 8.25, 1.90, "#FFFDFC", DYNAMIC, radius=0.18, lw=1.4, dashed=True, z=1)
    ax.text(5.02, 3.38, "OUR EXTENSION  ·  DynamicMapMask-RecurrentFusion", fontsize=8.5, color=DYNAMIC, fontweight="bold", ha="left")

    small_box(ax, 5.02, 2.06, 1.12, 1.00, "Dynamic mask", "TRAIN: GT + noise\nDEPLOY: YOLO11-seg", face=DYNAMIC_LIGHT, edge=DYNAMIC, title_size=8.0, detail_size=5.9)
    small_box(ax, 6.42, 2.06, 1.08, 1.00, "Depth → BEV", "camera → world\n→ robot local", face=MAP_LIGHT, edge=MAP, title_size=8.0, detail_size=6.0)
    arrow(ax, (6.14, 2.56), (6.42, 2.56), color=DYNAMIC, lw=1.4)
    route(ax, [(5.22, 4.43), (5.22, 3.78), (6.94, 3.78), (6.94, 3.06)], color=MAP, lw=1.2, dashed=True, z=3)
    ax.text(6.04, 3.82, "same delayed depth", fontsize=6.4, color=MAP, ha="center")

    # VisScore and patch token depiction.
    bev_grid(ax, 7.78, 2.18, 0.72, dynamic=True)
    ax.text(8.14, 3.02, "64×6 BEV tokens", fontsize=7.0, color=INK, fontweight="bold", ha="center")
    arrow(ax, (7.50, 2.56), (7.78, 2.56), color=MAP, lw=1.4)
    small_box(ax, 7.72, 1.78, 0.84, 0.22, "VisScore  Mₜ", face="#EDF4F6", edge=MAP, title_size=6.2)

    vertical_module(ax, 8.82, 2.03, 0.42, 1.02, "DynamicMapCross", MAP, fontsize=6.8)
    arrow(ax, (8.50, 2.67), (8.82, 2.67), color=MAP, lw=1.4)
    cuboid(ax, 9.51, 2.52, 0.36, 0.42, depth=0.10, face="#7AA2B5", edge=MAP, label="64×64D scene", label_size=6.0)
    arrow(ax, (9.24, 2.67), (9.51, 2.67), color=MAP, lw=1.4)
    route(ax, [(9.98, 2.75), (10.55, 2.75), (10.55, 4.35), (10.82, 4.35)], color=MAP, lw=1.35, dashed=True, z=4)
    ax.text(10.58, 3.52, "K,V", color=MAP, fontsize=6.5, ha="left")

    vertical_module(ax, 10.18, 1.91, 0.42, 0.94, "Presence Pool", DYNAMIC, fontsize=6.9)
    route(ax, [(8.50, 2.36), (9.02, 2.36), (9.02, 1.86), (10.18, 1.86), (10.18, 2.14)], color=DYNAMIC, lw=1.35, z=4)
    cuboid(ax, 10.90, 2.06, 0.34, 0.42, depth=0.10, face="#E3A44B", edge=DYNAMIC, label="dynamic 64D", label_size=6.0)
    arrow(ax, (10.60, 2.34), (10.90, 2.34), color=DYNAMIC, lw=1.4)
    route(ax, [(11.34, 2.31), (11.73, 2.31), (11.73, 3.80), (8.48, 3.80), (8.48, 4.35)], color=DYNAMIC, lw=1.35, dashed=True, z=3)
    ax.text(11.77, 3.18, "write into\nSRU memory", fontsize=6.3, color=DYNAMIC, ha="left", va="center")

    # Query for the late scene attention: h and state jointly select scene cells.
    route(ax, [(10.22, 5.35), (10.55, 5.35), (10.55, 5.58), (10.82, 5.58)], color=QUERY, lw=1.15, dashed=True, z=4)
    ax.text(10.56, 5.67, "Q = [hₜ, state]", fontsize=6.3, color=QUERY, ha="center")

    # Compact asymmetric critic annotation.
    rounded_panel(ax, 4.82, 0.25, 12.54, 0.66, "white", "#8A8A8A", radius=0.10, lw=1.0, dashed=True, z=1)
    ax.text(5.02, 0.58, "ASYMMETRIC CRITIC", fontsize=7.2, color=CRITIC, fontweight="bold", ha="left", va="center")
    ax.text(6.46, 0.58, "training only", fontsize=6.5, color=MUTED, fontstyle="italic", ha="left", va="center")
    ax.text(9.10, 0.58, "state 16 + depth 64 + privileged height 64 + dynamic 64", fontsize=6.8, color=MUTED, ha="center", va="center")
    ax.text(11.42, 0.58, "→", fontsize=10, color=CRITIC, ha="center", va="center")
    ax.text(13.45, 0.58, "Critic SRU 512 + scene query 64 + time 8", fontsize=6.8, color=MUTED, ha="center", va="center")
    ax.text(16.25, 0.58, "→  V(sₜ)", fontsize=7.5, color=CRITIC, fontweight="bold", ha="center", va="center")

    ax.text(0.38, 0.43, "Extended from the attention-based SRU navigation architecture (Yang et al., 2025)", fontsize=7.3, color=MUTED, fontstyle="italic", ha="left")
    ax.text(17.62, 0.10, "Mask is an observation; no mask-CNN or auxiliary segmentation loss", fontsize=6.7, color=MUTED, ha="right")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    outputs = {
        "svg": OUT_DIR / "instinct_nav_sru_paper_style_architecture.svg",
        "pdf": OUT_DIR / "instinct_nav_sru_paper_style_architecture.pdf",
        "png": OUT_DIR / "instinct_nav_sru_paper_style_architecture_4k.png",
    }
    fig.savefig(outputs["svg"], format="svg")
    fig.savefig(outputs["pdf"], format="pdf")
    fig.savefig(outputs["png"], format="png", dpi=240)
    plt.close(fig)
    for output in outputs.values():
        print(output)


if __name__ == "__main__":
    main()
