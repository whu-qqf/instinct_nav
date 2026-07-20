from collections import deque

import torch

from scripts.eval_depth_viz import select_delayed_depth_frames, select_episode_history_for_env


def test_select_delayed_depth_frames_uses_per_env_lags():
    history = deque(
        [
            torch.tensor([[[[10.0]]], [[[20.0]]]]),
            torch.tensor([[[[11.0]]], [[[21.0]]]]),
            torch.tensor([[[[12.0]]], [[[22.0]]]]),
        ],
        maxlen=3,
    )
    lags = torch.tensor([0, 2])

    delayed = select_delayed_depth_frames(history, lags)

    assert delayed.tolist() == [[[[12.0]]], [[[20.0]]]]


def test_select_delayed_depth_frames_falls_back_to_latest_when_lag_exceeds_history():
    history = deque(
        [
            torch.tensor([[[[10.0]]], [[[20.0]]]]),
            torch.tensor([[[[11.0]]], [[[21.0]]]]),
        ],
        maxlen=2,
    )
    lags = torch.tensor([3, 1])

    delayed = select_delayed_depth_frames(history, lags)

    assert delayed.tolist() == [[[[11.0]]], [[[20.0]]]]


def test_rear_gravity_aligned_camera_pose_is_behind_right_above_and_targets_robot():
    from scripts.eval_depth_viz import rear_gravity_aligned_camera_pose

    eye, target = rear_gravity_aligned_camera_pose(
        root_pos=[10.0, 20.0, 0.5],
        yaw=0.0,
        height_m=5.0,
        distance_m=4.0,
        lateral_m=1.5,
        target_height_m=0.5,
    )

    assert eye.tolist() == [6.0, 18.5, 5.5]
    assert target.tolist() == [10.0, 20.0, 1.0]

def test_select_episode_history_for_env_filters_reset_boundary_and_keeps_observations_aligned():
    snapshots = [
        {"episode_ids": torch.tensor([0, 0]), "value": "old"},
        {"episode_ids": torch.tensor([1, 0]), "value": "current_0"},
        {"episode_ids": torch.tensor([1, 1]), "value": "current_1"},
    ]
    observations = [{"value": "old_obs"}, {"value": "current_0_obs"}, {"value": "current_1_obs"}]

    selected_snapshots, selected_observations = select_episode_history_for_env(
        snapshots=snapshots,
        observations=observations,
        env_id=0,
        reference_snapshot=snapshots[-1],
    )

    assert [snapshot["value"] for snapshot in selected_snapshots] == ["current_0", "current_1"]
    assert [observation["value"] for observation in selected_observations] == ["current_0_obs", "current_1_obs"]

