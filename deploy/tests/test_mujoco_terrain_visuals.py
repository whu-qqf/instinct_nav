import contextlib
import sys
from pathlib import Path

import mujoco
import numpy as np


MUJOCO_DIR = Path(__file__).resolve().parents[1] / "mujoco"
sys.path.insert(0, str(MUJOCO_DIR))

import g1_parkour_sim2sim as low  # noqa: E402
import g1_sru_navigation_sim2sim as navigation  # noqa: E402


def _terrain_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(
            "terrain_"
        )
    ]


def test_stairs_and_ramp_are_added_to_model():
    stairs = low.build_model(
        low.URDF,
        0.005,
        low.TerrainConfig(terrain="stairs", stair_steps=4),
    )
    ramp = low.build_model(low.URDF, 0.005, low.TerrainConfig(terrain="ramp"))

    assert _terrain_names(stairs) == [f"terrain_stair_{index}" for index in range(4)]
    assert _terrain_names(ramp) == ["terrain_ramp"]


def test_depth_raycast_group_contains_terrain_but_not_robot():
    model = low.build_model(
        low.URDF,
        0.005,
        low.TerrainConfig(terrain="stairs", stair_steps=3),
    )
    navigation._prepare_raycast_groups(model)

    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name == "floor" or name.startswith("terrain_"):
            assert model.geom_group[index] == 0
        else:
            assert model.geom_group[index] == 1


def test_debug_overlay_contains_frustum_frame_and_goal():
    model = low.build_model(low.URDF, 0.005)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    camera = navigation.RayDepthCamera(model, torso_id, navigation.OnboardConfig())

    class FakeViewer:
        def __init__(self):
            self.user_scn = mujoco.MjvScene(model, maxgeom=32)

        def lock(self):
            return contextlib.nullcontext()

    viewer = FakeViewer()
    navigation._update_debug_visuals(
        viewer,
        data,
        camera,
        torso_id,
        np.asarray([2.0, 0.0, 0.5]),
        3.0,
    )

    # 8 camera-frustum lines, 3 frame axes, 1 goal sphere and 1 goal-height line.
    assert viewer.user_scn.ngeom == 13
