#!/usr/bin/env python3
"""Showcase scene presets for eval GIF rendering.

The lightweight realistic preset only changes visual material and lighting.
The rivermark preset loads a downloaded outdoor USD scene. The outdoor_campus
preset loads a generated local USD scene with no external asset dependencies.
Both attach deterministic point-navigation cases because USD terrain has no
procedural height-field goal masks.
"""

from __future__ import annotations

from pathlib import Path

SHOWCASE_SCENE_CHOICES = ("none", "realistic", "rivermark", "outdoor_campus", "official_warehouse")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RIVERMARK_ASSET_ROOT = PROJECT_ROOT / "assets" / "showcase" / "rivermark"
DEFAULT_OUTDOOR_CAMPUS_ASSET_ROOT = PROJECT_ROOT / "assets" / "showcase" / "outdoor_campus"
DEFAULT_OFFICIAL_WAREHOUSE_ASSET_ROOT = PROJECT_ROOT / "assets" / "showcase" / "official_warehouse"
RIVERMARK_USD_RELATIVE_PATH = "rivermark.usd"
OUTDOOR_CAMPUS_USD_RELATIVE_PATH = "campus.usd"
OFFICIAL_WAREHOUSE_USD_RELATIVE_PATH = "official_warehouse_showcase.usd"

# Hand-picked short road/plaza routes from Rivermark OpenDRIVE coordinates.
# The z value is terrain height; the robot default root height is added by IsaacLab reset.
RIVERMARK_FIXED_CASES = (
    {
        "spawn_position_world": (145.9, 73.7, 6.05),
        "goal_position_world": (150.3, 82.7, 6.15),
        "spawn_heading_world": 0.48,
    },
    {
        "spawn_position_world": (147.4, 90.0, 6.10),
        "goal_position_world": (132.9, 117.1, 6.35),
        "spawn_heading_world": 2.06,
    },
    {
        "spawn_position_world": (-51.5, 70.1, 5.75),
        "goal_position_world": (-46.7, 80.0, 5.85),
        "spawn_heading_world": 1.12,
    },
    {
        "spawn_position_world": (117.1, 176.1, 9.50),
        "goal_position_world": (121.3, 168.8, 9.50),
        "spawn_heading_world": -0.49,
    },
)

OUTDOOR_CAMPUS_FIXED_CASES = (
    {
        "spawn_position_world": (-12.0, -18.0, 0.05),
        "goal_position_world": (-12.0, 18.0, 0.05),
        "spawn_heading_world": 1.57,
    },
    {
        "spawn_position_world": (-4.0, -18.0, 0.05),
        "goal_position_world": (-4.0, 18.0, 0.05),
        "spawn_heading_world": 1.57,
    },
    {
        "spawn_position_world": (4.0, -18.0, 0.05),
        "goal_position_world": (4.0, 18.0, 0.05),
        "spawn_heading_world": 1.57,
    },
    {
        "spawn_position_world": (12.0, -18.0, 0.05),
        "goal_position_world": (12.0, 18.0, 0.05),
        "spawn_heading_world": 1.57,
    },
)


def apply_showcase_scene_preset(
    env_cfg,
    preset: str,
    *,
    sim_utils,
    isaac_nucleus_dir: str,
    isaaclab_nucleus_dir: str,
    asset_root: str | Path | None = None,
) -> bool:
    """Apply a showcase preset to an env config.

    Args:
        env_cfg: IsaacLab environment config with ``scene.terrain`` and ``scene.sky_light``.
        preset: Preset name. ``none`` leaves the config unchanged.
        sim_utils: IsaacLab ``isaaclab.sim`` module, injected to keep this helper testable.
        isaac_nucleus_dir: Base Isaac Nucleus assets directory.
        isaaclab_nucleus_dir: Base IsaacLab Nucleus assets directory, kept for preset API symmetry.

    Returns:
        True when a preset changed the config, False for ``none``.
    """
    if preset == "none":
        return False
    if preset == "realistic":
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.34, 0.30, 0.23),
            roughness=0.92,
            metallic=0.0,
        )
        env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(
            intensity=1850.0,
            exposure=0.1,
            texture_file=f"{isaac_nucleus_dir}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        return True
    if preset == "rivermark":
        rivermark_root = Path(asset_root) if asset_root is not None else DEFAULT_RIVERMARK_ASSET_ROOT
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = str(rivermark_root / RIVERMARK_USD_RELATIVE_PATH)
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.env_spacing = 6.0
        env_cfg.scene.terrain.visual_material = None
        env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(
            intensity=2800.0,
            exposure=0.0,
            texture_file=str(rivermark_root / "dsready_content/nv_core/common_assets/environments/hdri/domeskyhdri__50__0_5.HDR"),
        )
        env_cfg.commands.robot_goal.debug_vis = True
        env_cfg.commands.robot_goal.showcase_fixed_cases = [dict(case) for case in RIVERMARK_FIXED_CASES]
        return True

    if preset == "outdoor_campus":
        campus_root = Path(asset_root) if asset_root is not None else DEFAULT_OUTDOOR_CAMPUS_ASSET_ROOT
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = str(campus_root / OUTDOOR_CAMPUS_USD_RELATIVE_PATH)
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.env_spacing = 8.0
        env_cfg.scene.terrain.visual_material = None
        env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=3200.0, exposure=0.0)
        raycast_meshes = ["/World/ground/terrain/Geometry/Ground"]
        for sensor_name in ("raycast_camera", "height_scanner_critic", "height_scanner"):
            sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
            if sensor_cfg is not None and hasattr(sensor_cfg, "mesh_prim_paths"):
                sensor_cfg.mesh_prim_paths = list(raycast_meshes)
        env_cfg.commands.robot_goal.debug_vis = True
        env_cfg.commands.robot_goal.showcase_fixed_cases = [dict(case) for case in OUTDOOR_CAMPUS_FIXED_CASES]
        return True

    if preset == "official_warehouse":
        warehouse_root = Path(asset_root) if asset_root is not None else DEFAULT_OFFICIAL_WAREHOUSE_ASSET_ROOT
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = str(warehouse_root / OFFICIAL_WAREHOUSE_USD_RELATIVE_PATH)
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.env_spacing = 8.0
        env_cfg.scene.terrain.visual_material = None
        env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=2600.0, exposure=0.0)
        raycast_meshes = ["/World/ground/terrain/Geometry/RaycastProxy"]
        for sensor_name in ("raycast_camera", "height_scanner_critic", "height_scanner"):
            sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
            if sensor_cfg is not None and hasattr(sensor_cfg, "mesh_prim_paths"):
                sensor_cfg.mesh_prim_paths = list(raycast_meshes)
        env_cfg.commands.robot_goal.debug_vis = True
        env_cfg.commands.robot_goal.showcase_fixed_cases = [dict(case) for case in OUTDOOR_CAMPUS_FIXED_CASES]
        return True

    choices = ", ".join(SHOWCASE_SCENE_CHOICES)
    raise ValueError(f"Unsupported showcase scene {preset!r}. Expected one of: {choices}.")
