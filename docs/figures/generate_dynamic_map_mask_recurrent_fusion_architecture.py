"""Generate the presentation architecture figure for DynamicMapMask-RecurrentFusion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent

COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "line": "#AAB7B8",
    "sensor": "#EAF3F8",
    "sensor_edge": "#2874A6",
    "dynamic": "#FDEBD0",
    "dynamic_edge": "#D68910",
    "memory": "#E8F6F3",
    "memory_edge": "#148F77",
    "policy": "#F4ECF7",
    "policy_edge": "#7D3C98",
    "critic": "#F2F3F4",
    "critic_edge": "#626567",
    "control": "#E9F7EF",
    "control_edge": "#239B56",
    "white": "#FFFFFF",
}


def box(
    ax,
    xy,
    wh,
    title,
    detail="",
    *,
    face="white",
    edge="#17202A",
    lw=1.5,
    title_size=10.5,
    detail_size=8.2,
    dashed=False,
    zorder=3,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.61 if detail else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=zorder + 1,
        linespacing=1.05,
    )
    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.27,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=COLORS["muted"],
            zorder=zorder + 1,
            linespacing=1.13,
        )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.7,
    dashed=False,
    connectionstyle="arc3",
    label=None,
    label_offset=(0, 0.10),
    zorder=2,
):
    color = color or COLORS["ink"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.95},
            zorder=zorder + 1,
        )
    return patch


def routed_arrow(ax, points, *, color=None, lw=1.7, dashed=False, label=None, label_xy=None, zorder=2):
    """Draw a right-angle route with an arrowhead on the final segment."""
    color = color or COLORS["ink"]
    linestyle = "--" if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed, zorder=zorder)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.95},
            zorder=zorder + 1,
        )


def lane(ax, y, title, subtitle, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (8.72, y),
            6.82,
            2.18,
            boxstyle="round,pad=0.025,rounding_size=0.09",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.45,
            zorder=0,
        )
    )
    ax.text(8.88, y + 1.92, title, ha="left", va="center", fontsize=11, fontweight="bold", color=edge)
    ax.text(10.10, y + 1.92, subtitle, ha="left", va="center", fontsize=7.5, color=COLORS["muted"])


def draw_token_grid(ax, x, y, size=0.75):
    cell = size / 8
    for row in range(8):
        for col in range(8):
            intensity = 0.15 + 0.7 * ((row * 5 + col * 3) % 11) / 10
            color = plt.cm.Blues(intensity)
            ax.add_patch(
                Rectangle(
                    (x + col * cell, y + row * cell),
                    cell - 0.006,
                    cell - 0.006,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=5,
                )
            )
    for row, col in ((1, 5), (2, 5), (5, 2), (6, 2), (6, 3)):
        ax.add_patch(
            Rectangle(
                (x + col * cell, y + row * cell),
                cell - 0.006,
                cell - 0.006,
                facecolor=COLORS["dynamic_edge"],
                edgecolor="white",
                linewidth=0.25,
                zorder=6,
            )
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.45,
        8.58,
        "DynamicMapMask-RecurrentFusion Navigation Policy",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.47,
        8.22,
        "Dynamic BEV evidence enters both recurrent memory and post-SRU scene reasoning",
        fontsize=10.5,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    ax.text(
        15.55,
        8.52,
        "HIGH-LEVEL PPO  ·  5 Hz",
        fontsize=9,
        fontweight="bold",
        color=COLORS["policy_edge"],
        ha="right",
        va="center",
    )

    # Column headings.
    headings = [(0.45, "OBSERVATIONS"), (2.75, "SPATIAL PERCEPTION"), (5.55, "BEV SCENE MEMORY"), (8.90, "RECURRENT ACTOR–CRITIC"), (13.62, "HIERARCHICAL CONTROL")]
    for x, label in headings:
        ax.text(x, 7.88, label, fontsize=8.2, fontweight="bold", color=COLORS["muted"], ha="left")
    ax.plot([0.45, 15.55], [7.72, 7.72], color="#D5D8DC", linewidth=1.0)

    # Inputs.
    box(
        ax,
        (0.45, 6.42),
        (1.82, 0.92),
        "Delayed metric depth",
        "shared torso camera  ·  64×36",
        face=COLORS["sensor"],
        edge=COLORS["sensor_edge"],
    )
    box(
        ax,
        (0.45, 5.17),
        (1.82, 0.92),
        "Robot state + goal",
        "proprio · goal · last action  |  16D",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
    )
    box(
        ax,
        (0.45, 3.67),
        (1.82, 1.14),
        "Dynamic mask",
        "TRAIN: simulator GT + corruption\nDEPLOY: YOLO11-seg",
        face=COLORS["dynamic"],
        edge=COLORS["dynamic_edge"],
    )
    box(
        ax,
        (0.45, 1.42),
        (1.82, 0.92),
        "Height scan",
        "privileged critic input",
        face=COLORS["critic"],
        edge=COLORS["critic_edge"],
        dashed=True,
    )

    # Perception.
    box(
        ax,
        (2.76, 6.31),
        (2.13, 1.13),
        "Depth encoder + attention",
        "64×5×8 encoded map\nself-attn + state-conditioned cross-attn → 64D",
        face=COLORS["sensor"],
        edge=COLORS["sensor_edge"],
    )
    box(
        ax,
        (2.76, 4.18),
        (2.13, 1.56),
        "Aligned camera → BEV",
        "same delayed visual packet\ndepth back-projection · robot-local grid\n64×64, 6.4 m",
        face=COLORS["memory"],
        edge=COLORS["memory_edge"],
    )
    box(
        ax,
        (2.76, 2.51),
        (2.13, 1.15),
        "Persistent VisScore",
        "Mₜ = max(hitₜ, 0.995 Mₜ₋₁)\ndynamic mask is never accumulated",
        face=COLORS["memory"],
        edge=COLORS["memory_edge"],
    )
    box(
        ax,
        (2.76, 1.30),
        (2.13, 0.86),
        "Height encoder + attention",
        "64×7×7 → 64D",
        face=COLORS["critic"],
        edge=COLORS["critic_edge"],
        dashed=True,
    )

    # BEV token representation.
    box(
        ax,
        (5.38, 4.89),
        (2.88, 1.51),
        "",
        face=COLORS["memory"],
        edge=COLORS["memory_edge"],
    )
    draw_token_grid(ax, 5.58, 5.23, size=0.78)
    ax.text(5.97, 5.12, "dynamic presence", fontsize=6.3, ha="center", color=COLORS["dynamic_edge"])
    ax.text(7.22, 6.00, "8×8 scene tokens  |  64×6", fontsize=10.5, fontweight="bold", ha="center", color=COLORS["ink"])
    ax.text(7.22, 5.58, "history · hit · depth · edge", fontsize=7.6, ha="center", color=COLORS["muted"])
    ax.text(7.22, 5.31, "Δvisibility · dynamic", fontsize=7.6, ha="center", color=COLORS["muted"])
    ax.text(7.22, 5.05, "current-only dynamic channel", fontsize=6.7, ha="center", color=COLORS["dynamic_edge"])

    box(
        ax,
        (5.38, 3.12),
        (2.88, 1.26),
        "DynamicMapCross",
        "Q: 5 current channels  ·  K,V: history\n4-head attention + FFN → 64×64D",
        face=COLORS["sensor"],
        edge=COLORS["sensor_edge"],
    )
    box(
        ax,
        (5.38, 1.48),
        (2.88, 1.13),
        "Presence-weighted pooling",
        "learned null token + attention prior\n→ dynamic context 64D",
        face=COLORS["dynamic"],
        edge=COLORS["dynamic_edge"],
    )

    # Actor and critic lanes.
    lane(ax, 4.85, "ACTOR", "observations 2960D", COLORS["policy"], COLORS["policy_edge"])
    lane(ax, 1.66, "CRITIC", "training only · observations 6097D", COLORS["critic"], COLORS["critic_edge"])

    box(
        ax,
        (8.98, 5.22),
        (1.42, 1.05),
        "Concat 144D",
        "state 16 + depth 64\n+ dynamic 64",
        face=COLORS["white"],
        edge=COLORS["policy_edge"],
        title_size=9.5,
        detail_size=7.0,
    )
    box(
        ax,
        (10.68, 5.22),
        (1.38, 1.05),
        "Actor SRU",
        "1 layer · hₜ 512D",
        face=COLORS["memory"],
        edge=COLORS["memory_edge"],
    )
    box(
        ax,
        (12.33, 5.22),
        (1.45, 1.05),
        "Scene query",
        "Q: [hₜ,state]\nK,V: 64 tokens → 64D",
        face=COLORS["sensor"],
        edge=COLORS["sensor_edge"],
        title_size=9.5,
        detail_size=6.9,
    )
    box(
        ax,
        (14.04, 5.22),
        (1.25, 1.05),
        "Policy head",
        "[hₜ,scene] 576D\nTC-dropout · 576→512\nMLP · 512→256→128→3",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
        title_size=9.3,
        detail_size=5.9,
    )

    box(
        ax,
        (8.98, 2.04),
        (1.42, 1.05),
        "Concat 208D",
        "state 16 + depth 64\n+ height 64 + dynamic 64",
        face=COLORS["white"],
        edge=COLORS["critic_edge"],
        title_size=9.5,
        detail_size=6.8,
    )
    box(
        ax,
        (10.68, 2.04),
        (1.38, 1.05),
        "Critic SRU",
        "1 layer · hₜ 512D",
        face=COLORS["memory"],
        edge=COLORS["memory_edge"],
    )
    box(
        ax,
        (12.33, 2.04),
        (1.45, 1.05),
        "Scene query",
        "Q: [hₜ,state]\nK,V: 64 tokens → 64D",
        face=COLORS["sensor"],
        edge=COLORS["sensor_edge"],
        title_size=9.5,
        detail_size=6.9,
    )
    box(
        ax,
        (14.04, 2.04),
        (1.25, 1.05),
        "Value head",
        "+ time 8D\nTC-dropout · 584→512\nMLP · 512→256→128→1",
        face=COLORS["critic"],
        edge=COLORS["critic_edge"],
        title_size=9.3,
        detail_size=5.9,
    )

    # Hierarchical controller.
    box(
        ax,
        (13.52, 3.78),
        (2.04, 0.86),
        "SE(2) command",
        "vₓ, vᵧ, ωz",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
    )
    box(
        ax,
        (13.52, 0.55),
        (2.04, 0.93),
        "Frozen Instinct Parkour",
        "Official ONNX · depth history → 29 joints",
        face=COLORS["control"],
        edge=COLORS["control_edge"],
        title_size=9.4,
        detail_size=6.8,
    )

    # Core arrows: observations and scene construction.
    arrow(ax, (2.27, 6.88), (2.76, 6.88), color=COLORS["sensor_edge"])
    arrow(ax, (2.27, 5.62), (2.76, 6.46), color=COLORS["policy_edge"], connectionstyle="arc3,rad=-0.18")
    arrow(ax, (2.27, 6.69), (2.76, 5.35), color=COLORS["sensor_edge"], connectionstyle="arc3,rad=0.08")
    arrow(ax, (2.27, 4.24), (2.76, 4.68), color=COLORS["dynamic_edge"])
    arrow(ax, (3.82, 4.18), (3.82, 3.66), color=COLORS["memory_edge"])
    arrow(ax, (4.89, 4.90), (5.38, 5.48), color=COLORS["memory_edge"])
    arrow(ax, (4.89, 3.08), (5.38, 5.12), color=COLORS["memory_edge"], connectionstyle="arc3,rad=-0.12")
    arrow(ax, (6.82, 4.89), (6.82, 4.38), color=COLORS["sensor_edge"])
    arrow(ax, (6.20, 4.89), (6.20, 2.61), color=COLORS["dynamic_edge"], connectionstyle="arc3,rad=0.08")
    arrow(ax, (2.27, 1.88), (2.76, 1.73), color=COLORS["critic_edge"], dashed=True)

    # Actor flow.
    arrow(ax, (4.89, 6.88), (8.98, 5.98), color=COLORS["sensor_edge"], connectionstyle="arc3,rad=-0.08", label="depth 64D", label_offset=(0.0, 0.06))
    arrow(ax, (2.27, 5.62), (8.98, 5.58), color=COLORS["policy_edge"], connectionstyle="arc3,rad=0.03", label="state 16D", label_offset=(0.0, 0.10))
    arrow(ax, (8.26, 2.05), (8.98, 5.37), color=COLORS["dynamic_edge"], connectionstyle="arc3,rad=-0.20", label="dynamic 64D", label_offset=(-0.08, 0.02))
    arrow(ax, (10.40, 5.75), (10.68, 5.75), color=COLORS["policy_edge"])
    arrow(ax, (12.06, 5.75), (12.33, 5.75), color=COLORS["memory_edge"])
    arrow(ax, (13.78, 5.75), (14.04, 5.75), color=COLORS["policy_edge"])
    arrow(ax, (14.67, 5.22), (14.55, 4.64), color=COLORS["policy_edge"])
    arrow(
        ax,
        (15.56, 4.20),
        (15.56, 1.02),
        color=COLORS["control_edge"],
        connectionstyle="arc3,rad=-0.24",
    )
    arrow(ax, (11.37, 6.27), (11.37, 6.70), color=COLORS["memory_edge"], connectionstyle="arc3,rad=0.6", label="hₜ₋₁", label_offset=(0.34, 0.0))
    arrow(ax, (8.26, 3.74), (12.33, 5.38), color=COLORS["sensor_edge"], connectionstyle="arc3,rad=-0.19", label="scene tokens K,V", label_offset=(0.15, 0.0))

    # Critic flow.
    routed_arrow(
        ax,
        [(4.89, 1.73), (5.10, 1.23), (8.72, 1.23), (9.15, 2.04)],
        color=COLORS["critic_edge"],
        dashed=True,
        label="height 64D",
        label_xy=(6.88, 1.23),
    )
    arrow(ax, (4.89, 6.60), (8.98, 2.85), color=COLORS["sensor_edge"], connectionstyle="arc3,rad=0.17", label="depth 64D", label_offset=(-0.15, -0.04))
    arrow(ax, (2.27, 5.46), (8.98, 2.65), color=COLORS["policy_edge"], connectionstyle="arc3,rad=0.14", label="state 16D", label_offset=(0.05, -0.06))
    arrow(ax, (8.26, 1.91), (8.98, 2.18), color=COLORS["dynamic_edge"], label="dynamic 64D", label_offset=(0.05, 0.12))
    arrow(ax, (10.40, 2.56), (10.68, 2.56), color=COLORS["critic_edge"])
    arrow(ax, (12.06, 2.56), (12.33, 2.56), color=COLORS["memory_edge"])
    arrow(ax, (13.78, 2.56), (14.04, 2.56), color=COLORS["critic_edge"])
    arrow(ax, (8.26, 3.48), (12.33, 2.84), color=COLORS["sensor_edge"], dashed=True, connectionstyle="arc3,rad=0.16", label="scene tokens K,V", label_offset=(0.0, 0.08))

    # PPO training band and legend.
    box(
        ax,
        (5.38, 0.43),
        (7.53, 0.65),
        "PPO objective",
        "clipped surrogate + 0.1 value + 0.005 entropy   |   no mask auxiliary loss (GT mask is an observation)",
        face="#FAFAFA",
        edge=COLORS["critic_edge"],
        title_size=8.9,
        detail_size=6.9,
        dashed=True,
    )
    arrow(ax, (14.25, 2.04), (12.65, 1.08), color=COLORS["critic_edge"], dashed=True, connectionstyle="arc3,rad=-0.10")
    arrow(ax, (13.54, 4.20), (11.90, 1.08), color=COLORS["policy_edge"], dashed=True, connectionstyle="arc3,rad=0.10")

    legend_y = 0.18
    legend = [
        (COLORS["sensor_edge"], "depth / spatial scene"),
        (COLORS["dynamic_edge"], "dynamic evidence"),
        (COLORS["memory_edge"], "recurrent memory"),
        (COLORS["critic_edge"], "critic-only / training"),
        (COLORS["control_edge"], "frozen low-level control"),
    ]
    x = 0.48
    for color, label in legend:
        ax.plot([x, x + 0.24], [legend_y, legend_y], color=color, linewidth=3.0)
        ax.text(x + 0.30, legend_y, label, fontsize=6.8, color=COLORS["muted"], ha="left", va="center")
        x += 1.82
    ax.text(15.55, 0.18, "InstinctNav · current training architecture", fontsize=6.8, color=COLORS["muted"], ha="right", va="center")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    outputs = {
        "svg": ROOT / "dynamic_map_mask_recurrent_fusion_architecture.svg",
        "pdf": ROOT / "dynamic_map_mask_recurrent_fusion_architecture.pdf",
        "png": ROOT / "dynamic_map_mask_recurrent_fusion_architecture_4k.png",
    }
    fig.savefig(outputs["svg"], format="svg", bbox_inches=None)
    fig.savefig(outputs["pdf"], format="pdf", bbox_inches=None)
    fig.savefig(outputs["png"], format="png", dpi=240, bbox_inches=None)
    plt.close(fig)
    for output in outputs.values():
        print(output)


if __name__ == "__main__":
    main()
