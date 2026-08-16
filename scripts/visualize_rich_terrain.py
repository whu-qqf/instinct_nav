#!/usr/bin/env python3
"""Render deterministic top-down previews of the RichTerrain height fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/terrain/rich_terrain_topdown.png"))
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=72)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import copy

        import matplotlib.pyplot as plt
        import numpy as np

        from instinct_nav.terrains import RICH_MAZE_TERRAIN_CFG

        names = list(RICH_MAZE_TERRAIN_CFG.sub_terrains)
        terrain_size_x, terrain_size_y = RICH_MAZE_TERRAIN_CFG.size
        fields: list[np.ndarray] = []
        for index, name in enumerate(names):
            cfg = copy.deepcopy(RICH_MAZE_TERRAIN_CFG.sub_terrains[name])
            cfg.size = RICH_MAZE_TERRAIN_CFG.size
            cfg.horizontal_scale = RICH_MAZE_TERRAIN_CFG.horizontal_scale
            cfg.vertical_scale = RICH_MAZE_TERRAIN_CFG.vertical_scale
            cfg.slope_threshold = RICH_MAZE_TERRAIN_CFG.slope_threshold
            cfg.rng = np.random.default_rng(args.seed + index)
            cfg.function(args.difficulty, cfg)
            fields.append(cfg.height_field_visual.squeeze(0).numpy() * cfg.vertical_scale)

        global_min = min(float(field.min()) for field in fields)
        global_max = max(float(field.max()) for field in fields)
        fig, axes_grid = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        axes = list(axes_grid.flat[: len(names)])
        for unused_axis in axes_grid.flat[len(names) :]:
            fig.delaxes(unused_axis)
        image = None
        for axis, name, field in zip(axes, names, fields):
            image = axis.imshow(
                field.T,
                origin="lower",
                extent=(0.0, terrain_size_x, 0.0, terrain_size_y),
                cmap="terrain",
                vmin=global_min,
                vmax=global_max,
                interpolation="nearest",
            )
            axis.set_title(f"{name}  [{field.min():.2f}, {field.max():.2f}] m")
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_aspect("equal")

        assert image is not None
        fig.colorbar(image, ax=axes, label="height [m]", shrink=0.85)
        fig.suptitle(f"RichTerrain top-down preview  difficulty={args.difficulty:.2f}, seed={args.seed}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
        print(f"Saved terrain preview to {args.output.resolve()}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
