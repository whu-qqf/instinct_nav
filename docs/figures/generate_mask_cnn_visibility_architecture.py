"""Render the current MaskCNN + visibility SRU architecture without GPU use."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
C = dict(ink="#263238", gray="#617078", depth="#E5EFF7", mask="#FBE9CD",
         map="#DDF0E7", memory="#EBE3F3", head="#F7DFDF", frozen="#EAEDEF")


def box(ax, x, y, w, h, title, detail="", kind="depth", size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.07",
                 facecolor=C[kind], edgecolor=C["gray"], linewidth=1, zorder=3))
    ax.text(x+w/2, y+h*(0.66 if detail else 0.5), title, ha="center",
            va="center", fontsize=size, weight="bold", color=C["ink"], zorder=4)
    if detail:
        ax.text(x+w/2, y+h*0.28, detail, ha="center", va="center",
                fontsize=9, color=C["ink"], linespacing=1.3, zorder=4)


def route(ax, points, color=None, dashed=False):
    color = color or C["gray"]
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4,
                ls="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=12, lw=1.4, color=color,
                 linestyle="--" if dashed else "-", zorder=2))


def label(ax, x, y, text, size=9, **kw):
    ax.text(x, y, text, fontsize=size, color=C["gray"], va="center", **kw)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(22, 13))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    ax.set(xlim=(0, 22), ylim=(0, 13))
    ax.axis("off")
    label(ax, .2, 12.65, "InstinctNav | MaskCNN + Visibility SRU", 23, weight="bold")
    label(ax, 21.8, 12.65, "Current MaskCNN task  /  2026-09-09", 10, ha="right")
    label(ax, .2, 12.18, "A   ACTOR AND FROZEN CONTROL", 11, weight="bold")

    # Persistent visibility stays separate from the image-space dynamic encoder.
    box(ax, .2, 10.45, 3, 1.15, "Aligned visual packet",
        "Shared depth camera: H36 x W64\nDepth delay + obstacle-state history")
    box(ax, 4, 10.45, 3.1, 1.15, "Visibility BEV", "6.4 m x 6.4 m; 64 x 64 cells\n64 patches x 5 features", "map")
    box(ax, 8, 10.45, 3.2, 1.15, "Map cross-attention",
        "Q: current (4D); K,V: history (1D)\nMLPs + position; output 64 x 64", "map")
    box(ax, 12.5, 10.45, 3.3, 1.15, "Scene query attention",
        "Q = projection of [SRU h, proprio]\nMap K,V -> scene context 64D", "map")
    for a, b in [((3.2, 11), (4, 11)), ((7.1, 11), (8, 11)), ((11.2, 11), (12.5, 11))]:
        route(ax, [a, b])
    label(ax, 16.5, 11.2, "Visibility memory retained", 12, weight="bold")
    label(ax, 16.5, 10.8, "No dynamic-mask channel in BEV patches", 10)

    box(ax, .2, 8.25, 3, 1.2, "Depth encoder", "Pretrained DepthNoiseEncoder\n64 x 5 x 8 = 2560D")
    box(ax, 4, 8.25, 3.1, 1.2, "Visual attention", "Spatial self-attention + cross-attn\nProprio query -> depth feature 64D")
    route(ax, [(1.7, 10.45), (1.7, 9.45)])
    route(ax, [(3.2, 8.85), (4, 8.85)])
    box(ax, 4, 9.65, 3.1, .5, "Proprio + goal + prev. action: 16D", size=9)
    route(ax, [(5.55, 9.65), (5.55, 9.45)])
    box(ax, 8, 8.25, 1.7, 1.2, "Concat", "16 + 64 + 128\n= 208D", "memory")
    route(ax, [(7.1, 8.85), (8, 8.85)])
    route(ax, [(7.1, 9.9), (8.85, 9.9), (8.85, 9.45)])
    box(ax, 10.35, 8.25, 2.5, 1.2, "Actor SRU", "LSTM_SRU: 1 layer\nHidden state: 512D", "memory")
    route(ax, [(9.7, 8.85), (10.35, 8.85)])
    route(ax, [(11.6, 9.45), (11.6, 10.03), (14.15, 10.03), (14.15, 10.45)])
    route(ax, [(10.8, 8.25), (10.8, 7.85), (12.4, 7.85), (12.4, 8.25)])
    label(ax, 11.6, 7.62, "recurrent h, c", 9, ha="center")
    box(ax, 13.5, 8.25, 2, 1.2, "Late fusion", "512 + 64\n= 576D", "head")
    route(ax, [(12.85, 8.85), (13.5, 8.85)])
    route(ax, [(14.55, 10.45), (14.55, 9.45)])
    box(ax, 16.15, 8.25, 2.5, 1.2, "Actor head", "576 -> 512 -> 256\n-> 128 -> 3", "head")
    route(ax, [(15.5, 8.85), (16.15, 8.85)])
    box(ax, 19.35, 8.25, 2.4, 1.2, "SE(2) command", "vx, vy, yaw rate\nHigh level: 5 Hz", "head")
    route(ax, [(18.65, 8.85), (19.35, 8.85)])

    box(ax, .2, 6.25, 3, 1.2, "Dynamic image input",
        "GT mask + masked depth / 8\n2 x 36 x 64 = 4608D", "mask")
    box(ax, 4, 6.25, 3.1, 1.2, "Mask CNN + attention",
        "Image-space spatial features\nOutput: 128D (detail in B)", "mask")
    route(ax, [(3.2, 10.7), (3.55, 10.7), (3.55, 7.75), (1.7, 7.75), (1.7, 7.45)], "#BA8129")
    route(ax, [(3.2, 6.85), (4, 6.85)], "#BA8129")
    route(ax, [(7.1, 6.85), (8.85, 6.85), (8.85, 8.25)], "#BA8129")
    label(ax, 9.7, 6.85, "Dynamic image features enter recurrent memory directly", 11, weight="bold")
    label(ax, .2, 5.96, "Training: environment GT + segmentation perturbations; no mask prediction loss.", 10)
    box(ax, 18.2, 6.15, 3.55, 1.2, "Frozen G1 parkour controller",
        "Official ONNX + low-level observations\nDepth history 8 x 18 x 32; 50 Hz", "frozen", 10)
    route(ax, [(20.55, 8.25), (20.55, 7.35)])
    label(ax, 19.97, 5.9, "Joint position actions -> G1", 10, ha="center")
    route(ax, [(19.97, 6.15), (19.97, 6.02)])

    ax.axhline(5.55, color="#D5DCE0", lw=1)
    label(ax, .2, 5.23, "B   SHARED DYNAMIC IMAGE ENCODER", 11, weight="bold")
    specs = [(.2, 2.4, "2-channel image", "Mask + aligned masked depth\n2 x 36 x 64"),
             (3.25, 3.2, "3 x Conv + ELU", "Channels: 2 -> 16 -> 32 -> 32\n3 x 3 kernels; strides 1, 2, 2"),
             (7.1, 2.8, "Adaptive max pool", "32 x 9 x 16 -> 32 x 4 x 8\n32 spatial positions"),
             (10.55, 3.65, "Spatial self-attention", "Learned position + LN + 4-head MHA\n32D embedding; residual connection"),
             (14.85, 3.1, "Flatten + projection", "1024 -> LN -> 128 -> ELU\n-> Linear 128"),
             (18.6, 3.15, "Dynamic feature: 128D", "Shared actor / critic encoder\nOptimized by PPO gradients")]
    for x, w, title, detail in specs:
        box(ax, x, 3.8, w, 1.08, title, detail, "mask", 10)
    for (x, w, _, _), (nx, _, _, _) in zip(specs, specs[1:]):
        route(ax, [(x+w, 4.34), (nx, 4.34)], "#BA8129")
    label(ax, .2, 3.42, "Spatial positions are preserved through CNN and attention before 128D compression; this is not dynamic BEV-token pooling.", 10)

    ax.axhline(3.12, color="#D5DCE0", lw=1)
    label(ax, .2, 2.8, "C   ASYMMETRIC CRITIC  |  TRAINING ONLY", 11, weight="bold")
    box(ax, .2, 1.13, 4.25, 1.3, "Critic recurrent inputs",
        "Proprio 16 + depth attention 64\n+ privileged height attention 64\n+ shared dynamic image feature 128", "depth", 11)
    box(ax, 5.1, 1.13, 2.65, 1.3, "Critic SRU", "Input: 272D\nSeparate hidden state: 512D", "memory")
    box(ax, 8.4, 1.13, 3.6, 1.3, "Critic scene query", "Q: critic h + proprio; same BEV K,V\nScene context: 64D\nCritic h also bypasses to fusion", "map", 10)
    box(ax, 12.65, 1.13, 3, 1.3, "Late fusion", "h 512 + scene 64 + time 8\n= 584D\nTime: 1D -> Linear -> 8D", "head")
    box(ax, 16.3, 1.13, 3, 1.3, "Critic head", "584 -> 512 -> 256\n-> 128 -> 1", "head")
    box(ax, 19.95, 1.13, 1.8, 1.3, "Value", "V(o, h)", "head")
    for a, b in [(4.45, 5.1), (7.75, 8.4), (12, 12.65), (15.65, 16.3), (19.3, 19.95)]:
        route(ax, [(a, 1.78), (b, 1.78)])
    label(ax, .2, .7, "Flat observations: actor 7504D = 16 + 2560 + 320 + 4608; critic 10641D = actor + height 3136 + time 1.", 10)
    label(ax, .2, .3, "BEV patch: [history score, current hit, current depth, current edge, visibility delta].   Heads use ELU and fusion dropout (p = 0.2).", 9)
    for ext in ("png", "svg", "pdf"):
        path = OUT / f"mask_cnn_visibility_architecture.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
