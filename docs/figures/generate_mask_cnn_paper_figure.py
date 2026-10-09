"""Paper-style actor figure; schematic inputs, not captured simulator frames."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from generate_sru_paper_style_architecture import (
    arrow, route, cuboid, vertical_module, rounded_panel,
)

OUT = Path(__file__).resolve().parent
GRAY, GREEN, GOLD, RED, PURPLE = "#737C85", "#428779", "#C18A38", "#CE716C", "#9B87AF"


def text(ax, x, y, s, fs=10, color="#333333", **kwargs):
    ax.text(x, y, s, fontsize=fs, ha="center", va="center", color=color, **kwargs)


def tensor(ax, x, y, w, h, color, caption, grid=False):
    cuboid(ax, x, y, w, h, depth=.18, face=color, edge="#64686B")
    if grid:
        for i in range(1, 4):
            ax.plot([x+w*i/4]*2, [y, y+h], color="#7F8D8D", lw=.5, zorder=6)
        for j in range(1, 5):
            ax.plot([x, x+w], [y+h*j/5]*2, color="#7F8D8D", lw=.5, zorder=6)
    text(ax, x+w/2+.07, y-.3, caption, 9)


def concat(ax, x, y):
    ax.add_patch(Circle((x, y), .18, facecolor="white", edgecolor="#555555", lw=1.2, zorder=6))
    text(ax, x, y, "C", 10, weight="bold", zorder=7)


def image(ax, x, y, kind, label):
    yy, xx = np.mgrid[0:36, 0:64]
    depth = np.clip(1-yy/42, .12, 1)
    mask = (((xx > 12) & (xx < 23) & (yy > 14) & (yy < 31)) |
            ((xx > 42) & (xx < 49) & (yy > 11) & (yy < 22)))
    depth[mask & (xx < 32)] = .27
    depth[mask & (xx > 32)] = .58
    data = depth if kind == "depth" else mask if kind == "mask" else depth*mask
    ax.imshow(data, extent=(x, x+1.4, y, y+.8), cmap="cividis" if kind != "mask" else "gray",
              vmin=0, vmax=1, zorder=6, aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((x, y), 1.4, .8, fill=False, edgecolor="#666666", lw=.8, zorder=7))
    text(ax, x+.7, y-.22, label, 9)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(20, 8.7))
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.985)
    ax.set(xlim=(0, 20), ylim=(0, 8.7)); ax.axis("off")
    text(ax, .2, 8.37, "Dynamic-Aware SRU Navigation", 20)
    # Override title alignment explicitly to keep the full title on canvas.
    ax.texts[-1].set_ha("left")
    text(ax, 19.7, 8.35, "MaskCNN + visibility memory", 10, color=GRAY)
    ax.texts[-1].set_ha("right")
    rounded_panel(ax, .1, 1, 19.75, 6.9, "#FAFAFA", "#D3D6D8", radius=.24)
    rounded_panel(ax, .3, 5.95, 12.2, 1.7, "#EDF5F1", "#C8DCD3", radius=.15)
    rounded_panel(ax, .3, 1.25, 10.8, 2.18, "#FFF5E8", "#E4CEAC", radius=.15)

    text(ax, .55, 7.4, "Visibility memory", 11, color=GREEN, fontstyle="italic")
    ax.texts[-1].set_ha("left")
    # BEV branch: current observation retrieves historical visibility.
    n=8
    for i in range(n):
        for j in range(n):
            shade = .18+.55*((i*5+j*3)%11)/10
            ax.add_patch(Rectangle((2.4+i*.105,6.23+j*.105),.103,.103,
                                   facecolor=plt.cm.Greens(shade), lw=0, zorder=5))
    text(ax, 2.82, 7.32, r"$\mathcal{V}_t$", 13)
    text(ax, 2.82, 6.04, "BEV", 8)
    arrow(ax, (3.45,6.67), (4.05,6.67), color=GREEN)
    tensor(ax,4.12,6.25,.65,.75,"#BDDBCC",r"64 $\times$ 5",True)
    arrow(ax,(4.98,6.67),(5.55,6.67),color=GREEN)
    vertical_module(ax,5.6,6.12,.55,1.18,"MLPs + PE",GREEN,fontsize=9)
    arrow(ax,(6.18,6.67),(6.65,6.67),color=GREEN)
    vertical_module(ax,6.7,6.12,.65,1.18,"Cross-Attn",GREEN,fontsize=9)
    text(ax,7.06,7.48,"Q: current   K,V: history",8,color=GREEN)
    arrow(ax,(7.4,6.67),(8.05,6.67),color=GREEN)
    tensor(ax,8.13,6.25,.8,.75,"#BDDBCC",r"64 $\times$ 64",True)
    route(ax,[(9.17,6.67),(10.23,6.67)],color=GREEN)
    vertical_module(ax,10.3,6.12,.65,1.18,"Cross-Attn",GREEN,fontsize=9)
    text(ax,11.66,6.7,r"$z_t^{\mathrm{vis}}$",16,color=GREEN)
    text(ax,11.66,6.33,"64D",9,color=GREEN)
    arrow(ax,(10.98,6.7),(11.3,6.7),color=GREEN)

    # Original SRU depth backbone, rendered as spatial tensors and attention layers.
    image(ax,.55,4.15,"depth",r"Delayed depth $D_t$")
    arrow(ax,(2.0,4.55),(2.4,4.55))
    for i,(h,w) in enumerate([(1.05,.12),(.8,.18),(.55,.24)]):
        cuboid(ax,2.48+i*.34,4.55-h/2,w,h,depth=.13,face="#E1E5E8",edge=GRAY)
    text(ax,2.95,5.43,"Pretrained encoder",9,fontstyle="italic")
    text(ax,2.95,3.78,"DepthNoiseEncoder",8,color=GRAY)
    arrow(ax,(3.5,4.55),(3.9,4.55))
    tensor(ax,3.98,4.05,.65,.94,"#D7DEE3",r"64 $\times$ 5 $\times$ 8",True)
    arrow(ax,(4.85,4.55),(5.4,4.55))
    vertical_module(ax,5.45,3.93,.57,1.5,"Self-Attention",GRAY,fontsize=9)
    arrow(ax,(6.07,4.55),(6.47,4.55))
    tensor(ax,6.53,4.05,.62,.94,"#D7DEE3",r"$\mathcal{F}'_t$",True)
    arrow(ax,(7.38,4.55),(7.77,4.55))
    vertical_module(ax,7.83,3.93,.57,1.5,"Cross-Attention",GRAY,fontsize=9)
    arrow(ax,(8.43,4.55),(9,4.55))
    tensor(ax,9.08,4.4,.65,.22,"#D7DEE3",r"$z_t^{\mathrm{depth}}$: 64D")
    arrow(ax,(9.96,4.55),(11.5,4.55))
    concat(ax,11.7,4.55)
    text(ax,11.7,4.08,"208D",9)
    arrow(ax,(11.9,4.55),(12.52,4.55))
    ax.add_patch(Polygon([(12.57,3.68),(12.57,5.29),(12.95,5.68),(13.33,5.29),
                          (13.33,3.68),(12.95,3.3)],facecolor=PURPLE,edgecolor="#79638E",zorder=5))
    text(ax,12.95,4.5,"SRU",13,color="white",rotation=90,weight="bold",zorder=6)
    route(ax,[(13.1,3.46),(13.1,2.95),(14.05,2.95),(14.05,5.8),(12.95,5.8),(12.95,5.63)],dashed=True)
    text(ax,14.3,3.4,r"$h_{t-1},c_{t-1}$",11,rotation=90)
    arrow(ax,(13.35,4.55),(14.48,4.55))
    tensor(ax,14.55,4.39,.65,.26,"#CFC4DB",r"$h_t$: 512D")
    arrow(ax,(15.42,4.55),(16.0,4.55))
    concat(ax,16.2,4.55)
    text(ax,16.2,4.1,"576D",9)
    route(ax,[(12.0,6.7),(16.2,6.7),(16.2,4.75)],color=GREEN)
    route(ax,[(14.88,4.77),(14.88,5.6),(10.62,5.6),(10.62,6.08)],color=RED,dashed=True)
    text(ax,11.8,5.8,r"Query: $[h_t,p_t]$",9,color=RED)
    arrow(ax,(16.4,4.55),(17.0,4.55))
    vertical_module(ax,17.05,3.65,.67,1.85,"MLP + TC-Dropout","#8394AA",fontsize=10)
    arrow(ax,(17.77,4.55),(18.35,4.55))
    text(ax,18.92,4.6,r"$a_t$",22)
    text(ax,18.92,4.05,r"$(v_x,v_y,\omega_z)$",11)
    text(ax,18.92,3.68,"5 Hz",9,color=GRAY)
    route(ax,[(18.92,3.44),(18.92,2.9)])
    rounded_panel(ax,17.0,1.62,2.35,1.23,"#ECEFF1","#ADB4B9",radius=.08,z=5)
    text(ax,18.17,2.49,"Frozen G1 controller",10,weight="bold",zorder=6)
    text(ax,18.17,2.08,"Official parkour ONNX",9,zorder=6)
    text(ax,18.17,1.78,"Joint targets | 50 Hz",9,zorder=6)

    # Separate mask stream: no BEV projection or semantic-token pooling.
    text(ax,.55,3.17,"Dynamic image encoding",11,color=GOLD,fontstyle="italic")
    ax.texts[-1].set_ha("left")
    image(ax,.6,1.88,"mask",r"$M_t$")
    image(ax,2.2,1.88,"masked",r"$M_t\odot\bar{D}_t$")
    text(ax,2.1,1.42,r"Aligned 2 $\times$ 36 $\times$ 64 input",9)
    arrow(ax,(3.66,2.3),(4.15,2.3),color=GOLD)
    for i,h in enumerate([1.05,.8,.6]):
        cuboid(ax,4.2+i*.29,2.3-h/2,.15+.04*i,h,depth=.1,face="#F0C98D",edge=GOLD)
    text(ax,4.65,3.0,"CNN + pool",9,color=GOLD)
    text(ax,4.65,1.5,"2 / 16 / 32 / 32",8)
    arrow(ax,(5.15,2.3),(5.55,2.3),color=GOLD)
    tensor(ax,5.62,1.89,.55,.78,"#F0C98D",r"32 $\times$ 4 $\times$ 8",True)
    arrow(ax,(6.38,2.3),(6.84,2.3),color=GOLD)
    vertical_module(ax,6.9,1.58,.58,1.38,"Self-Attn + PE",GOLD,fontsize=9)
    arrow(ax,(7.52,2.3),(7.97,2.3),color=GOLD)
    vertical_module(ax,8.02,1.58,.52,1.38,"Flatten + MLP",GOLD,fontsize=9)
    arrow(ax,(8.59,2.3),(9.12,2.3),color=GOLD)
    tensor(ax,9.2,2.15,.85,.24,"#F0C98D",r"$z_t^{\mathrm{dyn}}$: 128D")
    route(ax,[(10.28,2.3),(11.7,2.3),(11.7,4.34)],color=GOLD,lw=1.8)

    # Proprio conditions depth attention and is concatenated before recurrence.
    # Place the proprio source in the open right-hand region instead of above title.
    # The explicit route is below the visibility branch and above depth attention.
    text(ax,9.48,5.68,r"$p_t$ (16D)",10,color=RED)
    route(ax,[(9.2,5.6),(8.12,5.6),(8.12,5.46)],color=RED,dashed=True)
    route(ax,[(9.7,5.53),(11.7,5.53),(11.7,4.76)],color=RED)
    route(ax,[(1.25,4.98),(1.25,6.62),(2.28,6.62)],color=GREEN)

    text(ax,.35,.64,"C",9,weight="bold")
    text(ax,.65,.64,"Concatenation",9); ax.texts[-1].set_ha("left")
    text(ax,4.6,.64,"Dashed red: query conditioning",9,color=RED)
    text(ax,9.15,.64,"Gold: dynamic image pathway",9,color=GOLD)
    text(ax,14.05,.64,"Green: visibility pathway",9,color=GREEN)
    text(ax,19.6,.64,"Actor shown",9,color=GRAY); ax.texts[-1].set_ha("right")
    text(ax,10,.22,"Schematic images. p: proprioception, goal and previous action. Training: perturbed GT masks; mask encoder shared with critic.",9,color=GRAY)
    for ext in ("png","pdf","svg"):
        p=OUT/f"mask_cnn_paper_architecture.{ext}"
        fig.savefig(p,dpi=240,facecolor="white")
        print(p)
    plt.close(fig)


if __name__ == "__main__":
    main()
