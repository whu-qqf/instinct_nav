"""Render the high-level DynamicMapCross v4 navigation policy architecture as a PNG.

Structure is transcribed from source/rsl_rl/rsl_rl/modules/actor_critic_sru.py
(ActorCriticSRU + CrossAttentionFuseModule + LSTM_SRU) and the v4 deploy manifest.

Output: artifacts/network/dynamic_map_cross_v4_architecture.png
No CJK font is available in the env, so all labels are English/numeric.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette ------------------------------------------------------------
INPUT_FC, INPUT_EC = "#DCE9F5", "#3E7CB1"      # inputs (blue)
ENC_FC, ENC_EC = "#DFF2E1", "#3F9B4F"          # encoders (green)
ATT_FC, ATT_EC = "#FCEBD2", "#E08E0B"          # attention (orange)
SRU_FC, SRU_EC = "#E7DDF3", "#7C4DAB"          # recurrent (purple)
HEAD_FC, HEAD_EC = "#FAD9D9", "#C0392B"        # actor head / output (red)
LOW_FC, LOW_EC = "#DCDCDC", "#555555"          # frozen low level (gray)
CRIT_FC, CRIT_EC = "#F3F3F3", "#888888"        # critic-only (dashed gray)
CONCAT_FC, CONCAT_EC = "#FFF6DA", "#B8912B"    # concat nodes

# arrow colors by signal
C_PROP, C_DEPTH, C_TOK, C_HID, C_MAIN = "#2E6FB0", "#3F9B4F", "#E08E0B", "#7C4DAB", "#333333"


class Box:
    def __init__(self, cx, cy, w, h, text, fc, ec, fs=9.5, tc="black", ls="-", lw=1.6, bold=False):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.text, self.fc, self.ec = text, fc, ec
        self.fs, self.tc, self.ls, self.lw, self.bold = fs, tc, ls, lw, bold

    def draw(self, ax):
        ax.add_patch(FancyBboxPatch(
            (self.cx - self.w / 2, self.cy - self.h / 2), self.w, self.h,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            linewidth=self.lw, edgecolor=self.ec, facecolor=self.fc,
            linestyle=self.ls, zorder=3))
        ax.text(self.cx, self.cy, self.text, ha="center", va="center",
                fontsize=self.fs, color=self.tc, zorder=4,
                fontweight="bold" if self.bold else "normal")

    def top(self):    return (self.cx, self.cy + self.h / 2)
    def bottom(self): return (self.cx, self.cy - self.h / 2)
    def left(self):   return (self.cx - self.w / 2, self.cy)
    def right(self):  return (self.cx + self.w / 2, self.cy)


def arrow(ax, p0, p1, color=C_MAIN, style="-|>", conn="arc3,rad=0.0", lw=1.8, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=16, color=color,
        linewidth=lw, connectionstyle=conn, zorder=2, linestyle=ls,
        shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, label, fontsize=8, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5)


fig, ax = plt.subplots(figsize=(13.5, 17))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ===================== INPUTS =====================
i_cam  = Box(16, 93, 24, 6.5, "Shared camera\nraw depth  64 x 36", INPUT_FC, INPUT_EC)
i_prop = Box(50, 93, 22, 6.5, "proprio + goal + action\n16", INPUT_FC, INPUT_EC)
i_vis  = Box(84, 93, 26, 6.5, "VisScore history map (decay 0.995)\n+ current frame", INPUT_FC, INPUT_EC)

# ===================== ENCODER / PATCH =====================
enc = Box(16, 80, 26, 8,
          "DepthNoiseEncoder\n(standalone ONNX CNN)\n64 x 36  ->  64 x 5 x 8 = 2560", ENC_FC, ENC_EC)
patch = Box(84, 79, 28, 8.5,
            "dynamic-map patches  [64 x 5] = 320\n[hist_score, hit, depth, edge, delta_vis]",
            INPUT_FC, INPUT_EC)

# ===================== MAIN SPINE (left/center) =====================
att_img = Box(30, 66, 30, 9,
              "Depth Cross-Attention\nCrossAttentionFuseModule (4 heads)\n"
              "2560 (=64x5x8) + proprio  ->  image feat 64", ATT_FC, ATT_EC)
concat1 = Box(30, 54, 30, 6, "concat:  proprio(16)  (+)  image feat(64)  = 80", CONCAT_FC, CONCAT_EC)
sru     = Box(30, 43, 28, 8,
              "SRU memory  (LSTM_SRU)\nhidden = 256, 1 layer   (recurrent h, c)", SRU_FC, SRU_EC, bold=True)

# ===================== RIGHT BRANCH: DynamicMapCross =====================
lvl1 = Box(80, 64, 30, 12,
           "(1) map -> map  cross-attn (4 heads)\n"
           "Q  = current (Linear 4->64)\n"
           "K/V = history (Linear 1->64)  + 2D pos-embed\n"
           "-> 64 learned dynamic tokens", ATT_FC, ATT_EC)
lvl2 = Box(80, 45, 30, 12,
           "(2) query -> token  cross-attn (4 heads)\n"
           "Q  = SRU hidden(256) (+) proprio(16) -> 64\n"
           "K/V = 64 tokens\n"
           "-> scene feature 64", ATT_FC, ATT_EC, bold=True)

# dashed box grouping the two-level module (starts BELOW the patch input box)
ax.add_patch(FancyBboxPatch((62.5, 36.5), 35, 35.0,
             boxstyle="round,pad=0.2,rounding_size=0.8",
             linewidth=1.4, edgecolor="#B36B00", facecolor="none",
             linestyle=(0, (5, 3)), zorder=1))
ax.text(80, 73.2, "DynamicMapCross  (two-level cross-attention)",
        ha="center", va="center", fontsize=10, color="#8a5200", fontweight="bold", zorder=5)

# ===================== MERGE / HEAD / OUTPUT =====================
merge = Box(54, 29, 40, 6,
            "post-recurrent  =  SRU hidden(256)  (+)  scene feature(64)  =  320",
            CONCAT_FC, CONCAT_EC)
head = Box(54, 21, 26, 5.5, "Actor MLP  ->  3 raw actions", HEAD_FC, HEAD_EC, bold=True)
outp = Box(54, 13, 44, 6.5,
           "tanh  ->  affine\nvx in [0,1],   vy = 0,   wz in [-1,1]      (zero action -> [0.5, 0, 0])",
           HEAD_FC, HEAD_EC)
low = Box(54, 4.5, 52, 5.5,
          "Frozen Parkour low-level  (official ONNX, 50 Hz)  ->  29-D joint targets",
          LOW_FC, LOW_EC)

# ===================== CRITIC-ONLY NOTE =====================
crit = Box(15.5, 27, 26, 13,
           "CRITIC  (training only)\nadds on top of actor inputs:\n"
           "+ time_normalized (1)\n"
           "+ height_scan 64x7x7 = 3136\n     -> attn_height_net (privileged)\n"
           "shares SRU + token attn\n->  value       critic obs = 6033",
           CRIT_FC, CRIT_EC, fs=8.8, ls=(0, (4, 3)))

for b in [i_cam, i_prop, i_vis, enc, patch, att_img, concat1, sru,
          lvl1, lvl2, merge, head, outp, low, crit]:
    b.draw(ax)

# ===================== ARROWS =====================
# depth path (green)
arrow(ax, i_cam.bottom(), enc.top(), C_DEPTH)
arrow(ax, enc.bottom(), (att_img.left()[0] - 0.5, att_img.left()[1] + 1),
      C_DEPTH, conn="arc3,rad=-0.15", label="2560")
# proprio path (blue)
arrow(ax, i_prop.bottom(), att_img.top(), C_PROP, conn="arc3,rad=0.05", label="proprio 16")
arrow(ax, (att_img.cx, att_img.bottom()[1]), concat1.top(), C_DEPTH, label="img feat 64")
arrow(ax, i_prop.bottom(), (concat1.right()[0], concat1.right()[1] + 0.3),
      C_PROP, conn="arc3,rad=-0.35")
# concat -> SRU (main)
arrow(ax, concat1.bottom(), sru.top(), C_MAIN, label="80")
# vis path (blue) -> patch -> level1
arrow(ax, i_vis.bottom(), patch.top(), C_PROP)
arrow(ax, patch.bottom(), lvl1.top(), C_TOK, label="320")
# level1 -> level2 (tokens)
arrow(ax, lvl1.bottom(), lvl2.top(), C_TOK, label="64 tokens")
# SRU hidden -> level2 query  (purple)  and SRU hidden -> merge
arrow(ax, sru.right(), (lvl2.left()[0], lvl2.left()[1] + 3.0),
      C_HID, conn="arc3,rad=-0.18", label="hidden 256 (query)")
arrow(ax, (sru.cx, sru.bottom()[1]), (merge.left()[0] + 4, merge.top()[1] + 0.2),
      C_HID, conn="arc3,rad=-0.1", label="hidden 256")
# proprio also into level2 query
arrow(ax, (concat1.right()[0], concat1.right()[1] - 1.2),
      (lvl2.left()[0], lvl2.left()[1] - 2.0), C_PROP, conn="arc3,rad=-0.28")
# level2 scene feature -> merge (orange)
arrow(ax, lvl2.bottom(), (merge.right()[0] - 4, merge.top()[1] + 0.2),
      C_TOK, conn="arc3,rad=0.12", label="scene feat 64")
# merge -> head -> output -> low level (main)
arrow(ax, merge.bottom(), head.top(), C_MAIN, label="320")
arrow(ax, head.bottom(), outp.top(), C_MAIN, label="3")
arrow(ax, outp.bottom(), low.top(), C_MAIN, label="[vx, vy, wz]")
# critic taps SRU (dashed)
arrow(ax, sru.left(), crit.top(), "#888888",
      conn="arc3,rad=0.1", ls=(0, (4, 3)), lw=1.4)

# ===================== TITLE + LEGEND =====================
ax.text(50, 99.2, "instinct_nav  -  High-level DynamicMapCross Navigation Policy (v4)",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(50, 97.0,
        "high-level 5 Hz      actor obs = 16 + 2560 + 320 = 2896      critic obs = 6033",
        ha="center", va="center", fontsize=10.5, color="#444")

legend_items = [
    Line2D([0], [0], color=C_PROP, lw=3, label="proprio / goal path"),
    Line2D([0], [0], color=C_DEPTH, lw=3, label="depth path"),
    Line2D([0], [0], color=C_TOK, lw=3, label="dynamic-map / token path"),
    Line2D([0], [0], color=C_HID, lw=3, label="SRU recurrent hidden"),
    Line2D([0], [0], color=C_MAIN, lw=3, label="main / merged signal"),
    Line2D([0], [0], color="#888888", lw=2, ls="--", label="critic-only (training)"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.005, 0.005),
          fontsize=9, frameon=True, framealpha=0.95, ncol=1)

out_dir = "/workspace/instinct_nav/artifacts/network"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "dynamic_map_cross_v4_architecture.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out_path)
