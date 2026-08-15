#!/usr/bin/env python3
"""Open the real RichTerrain meshes in an interactive Isaac Sim viewport."""

from __future__ import annotations

import argparse
import copy

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain",
        default="all",
        help="Sub-terrain to load, or 'all' to load every RichTerrain type.",
    )
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--color-scheme", choices=("height", "random", "none"), default="height")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.headless:
        parser.error("inspect-terrain requires a GUI; remove --headless")
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be between 0 and 1")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

        from instinct_nav.terrains import RICH_MAZE_TERRAIN_CFG

        terrain_cfg = copy.deepcopy(RICH_MAZE_TERRAIN_CFG)
        available = tuple(terrain_cfg.sub_terrains)
        if args.terrain != "all":
            if args.terrain not in terrain_cfg.sub_terrains:
                parser.error(f"unknown terrain '{args.terrain}'; choose from: all, {', '.join(available)}")
            terrain_cfg.sub_terrains = {args.terrain: terrain_cfg.sub_terrains[args.terrain]}
            terrain_cfg.num_rows = 1
            terrain_cfg.num_cols = 1
        else:
            # Curriculum placement assigns terrain types deterministically by column.
            terrain_cfg.num_rows = 1
            terrain_cfg.num_cols = len(terrain_cfg.sub_terrains)
            for sub_cfg in terrain_cfg.sub_terrains.values():
                sub_cfg.proportion = 1.0

        terrain_cfg.seed = args.seed
        terrain_cfg.curriculum = args.terrain == "all"
        terrain_cfg.difficulty_range = (args.difficulty, args.difficulty)
        terrain_cfg.border_width = 2.0
        terrain_cfg.color_scheme = args.color_scheme
        terrain_cfg.use_cache = False

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device=args.device))
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)

        importer_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_cfg,
            num_envs=terrain_cfg.num_rows * terrain_cfg.num_cols,
            max_init_terrain_level=None,
            debug_vis=True,
        )
        if args.color_scheme in ("height", "random"):
            importer_cfg.visual_material = None
        terrain = TerrainImporter(importer_cfg)

        if args.terrain == "all":
            sim.set_camera_view(eye=[75.0, -150.0, 100.0], target=[0.0, 0.0, 0.0])
        else:
            sim.set_camera_view(eye=[28.0, -28.0, 20.0], target=[0.0, 0.0, 0.0])

        sim.reset()
        print(f"[INFO] Loaded RichTerrain: {args.terrain}")
        print(f"[INFO] Types: {', '.join(available)}")
        print("[INFO] Use the Isaac Sim viewport camera controls to explore; close the window to exit.")
        while simulation_app.is_running():
            sim.step()

        # Keep a live reference until the simulation loop ends.
        del terrain
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
