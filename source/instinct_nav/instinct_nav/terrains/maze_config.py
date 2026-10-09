# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze terrains."""

import copy

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_maze_cfg import HfMazeTerrainCfg

MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,  # Border around the entire terrain grid (not per-tile)
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=True,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "stairs": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            random_wall_ratio=1.0,
            non_maze_terrain=False,
            stairs=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,  # Enables pit/trough generation
        ),
    },
)
"""Maze terrain configuration for navigation tasks."""


RICH_MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.16,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=True,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.16,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "variable_stairs": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            stairs=True,
            stair_num_steps_range=(3, 7),
            stair_step_height_range=(0.10, 0.20),
            stair_step_depth_range=(0.28, 0.40),
            stair_max_total_height=1.0,
        ),
        "blocks": HfMazeTerrainCfg(
            proportion=0.16,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            non_maze_terrain=True,
            terrain_feature="blocks",
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.12,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,
        ),
        "traversal_mix": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=1.0,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            non_maze_terrain=True,
            terrain_feature="traversal_mix",
            stair_num_steps_range=(3, 7),
            stair_step_height_range=(0.10, 0.20),
            stair_step_depth_range=(0.28, 0.40),
            stair_max_total_height=1.0,
            add_ramps=True,
            add_raised_bars=True,
            add_sunken_bars=True,
        ),
    },
)
"""Expanded navigation terrain with variable stairs and InstinctLab-inspired features."""


HUMAN_SCALE_RICH_MAZE_TERRAIN_CFG = copy.deepcopy(RICH_MAZE_TERRAIN_CFG)
for _sub_terrain in HUMAN_SCALE_RICH_MAZE_TERRAIN_CFG.sub_terrains.values():
    _sub_terrain.grid_size = (20, 20)
    _sub_terrain.cell_size = 1.5
HUMAN_SCALE_RICH_MAZE_TERRAIN_CFG.sub_terrains["pits"].bridge_width_cells = 1
for _terrain_name in ("variable_stairs", "traversal_mix"):
    _stairs_cfg = HUMAN_SCALE_RICH_MAZE_TERRAIN_CFG.sub_terrains[_terrain_name]
    _stairs_cfg.stair_num_steps_range = (3, 5)
    _stairs_cfg.stair_step_depth_range = (0.28, 0.30)
"""30 m rich terrain with denser 1.5 m topology and single-cell pit bridges for G1."""
