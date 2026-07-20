import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_backend_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "instinct_nav"
        / "navigation"
        / "mdp"
        / "navigation"
        / "actions"
        / "sonic_backend.py"
    )
    spec = importlib.util.spec_from_file_location("sonic_backend", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_backend_module = _load_backend_class()
DeterministicSonicBackend = _backend_module.DeterministicSonicBackend


def test_fake_backend_outputs_joint_targets_with_expected_shape():
    backend = DeterministicSonicBackend(num_envs=3, action_dim=6, device="cpu")
    commands = {
        "target_vel": torch.tensor([[0.0], [0.5], [1.0]]),
        "movement_direction": torch.zeros(3, 3),
        "facing_direction": torch.zeros(3, 3),
        "mode": torch.zeros(3, 1, dtype=torch.long),
        "height": torch.full((3, 1), -1.0),
    }

    actions = backend.compute_joint_targets(commands)

    assert actions.shape == (3, 6)
    assert torch.allclose(actions[0], torch.zeros(6))
    assert torch.count_nonzero(actions[1]).item() > 0
    assert torch.count_nonzero(actions[2]).item() > 0


def test_onnx_backend_requires_existing_model_files(tmp_path):
    missing_planner = tmp_path / "missing_planner.onnx"
    missing_encoder = tmp_path / "missing_encoder.onnx"
    missing_decoder = tmp_path / "missing_decoder.onnx"

    with pytest.raises(FileNotFoundError, match="missing_planner.onnx"):
        _backend_module.SonicOnnxBackend(
            num_envs=1,
            action_dim=29,
            device="cpu",
            planner_path=missing_planner,
            encoder_path=missing_encoder,
            decoder_path=missing_decoder,
        )


def test_onnx_backend_validates_release_model_contracts():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )

    assert backend.planner_input_shapes["context_mujoco_qpos"] == [1, 4, 36]
    assert backend.encoder_input_shape == [1, 1762]
    assert backend.encoder_output_shape == [1, 64]
    assert backend.decoder_input_shape == [1, 994]
    assert backend.decoder_output_shape == [1, 29]


def test_onnx_backend_rejects_action_dim_mismatch():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"

    with pytest.raises(ValueError, match="action_dim.*29"):
        _backend_module.SonicOnnxBackend(
            num_envs=1,
            action_dim=37,
            device="cpu",
            planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
            encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
            decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
        )


def test_onnx_backend_runs_release_decoder_and_returns_joint_targets():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    backend = _backend_module.SonicOnnxBackend(
        num_envs=2,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    commands = {
        "target_vel": torch.tensor([[0.0], [0.5]], dtype=torch.float32),
        "movement_direction": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[0], [2]], dtype=torch.long),
        "height": torch.full((2, 1), -1.0, dtype=torch.float32),
    }

    actions = backend.compute_joint_targets(commands)

    assert actions.shape == (2, 29)
    assert torch.isfinite(actions).all()
    assert not torch.allclose(actions, torch.zeros_like(actions))


def test_onnx_backend_conditions_release_tokens_on_navigation_commands():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    idle_backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    forward_backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    idle_commands = {
        "target_vel": torch.tensor([[0.0]], dtype=torch.float32),
        "movement_direction": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[0]], dtype=torch.long),
        "height": torch.full((1, 1), -1.0, dtype=torch.float32),
    }
    forward_commands = {
        "target_vel": torch.tensor([[0.6]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2]], dtype=torch.long),
        "height": torch.full((1, 1), -1.0, dtype=torch.float32),
    }

    idle_actions = idle_backend.compute_joint_targets(idle_commands)
    forward_actions = forward_backend.compute_joint_targets(forward_commands)

    assert not torch.allclose(idle_actions, forward_actions)
    assert np.count_nonzero(forward_backend._token_state) > 0


def test_onnx_backend_uses_real_low_level_observation_history():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    commands = {
        "target_vel": torch.tensor([[0.4]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2]], dtype=torch.long),
        "height": torch.full((1, 1), -1.0, dtype=torch.float32),
    }
    low_level_observations = torch.zeros(1, 286, dtype=torch.float32)
    low_level_observations[:, 3:6] = torch.tensor([[0.11, -0.22, 0.33]])
    low_level_observations[:, 6:9] = torch.tensor([[0.20, -0.30, -0.93]])
    low_level_observations[:, 12:41] = torch.linspace(-0.05, 0.05, 29).unsqueeze(0)
    low_level_observations[:, 41:70] = torch.linspace(0.7, -0.7, 29).unsqueeze(0)

    backend.compute_joint_targets(commands, low_level_observations=low_level_observations)

    np.testing.assert_allclose(backend._his_base_ang_vel[0, -1], [0.11, -0.22, 0.33], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(backend._his_gravity[0, -1], [0.20, -0.30, -0.93], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        backend._his_joint_pos[0, -1],
        backend.default_angles + low_level_observations[0, 12:41].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        backend._his_joint_vel[0, -1],
        low_level_observations[0, 41:70].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


class _CountingSession:
    def __init__(self, kind):
        self.kind = kind
        self.run_count = 0

    def run(self, output_names, inputs):
        self.run_count += 1
        if self.kind == "planner":
            return [np.zeros((1, 64, 36), dtype=np.float32)]
        if self.kind == "encoder":
            return [np.full((1, 64), self.run_count, dtype=np.float32)]
        if self.kind == "decoder":
            return [np.zeros((1, 29), dtype=np.float32)]
        raise AssertionError(self.kind)


def _make_counting_onnx_backend(num_envs=2):
    backend = object.__new__(_backend_module.SonicOnnxBackend)
    backend.num_envs = num_envs
    backend.action_dim = 29
    backend.device = torch.device("cpu")
    backend.planner_session = _CountingSession("planner")
    backend.encoder_session = _CountingSession("encoder")
    backend.decoder_session = _CountingSession("decoder")
    backend._token_state = np.zeros((num_envs, 64), dtype=np.float32)
    backend._his_base_ang_vel = np.zeros((num_envs, 10, 3), dtype=np.float32)
    backend._his_joint_pos = np.tile(backend.default_angles.reshape(1, 1, 29), (num_envs, 10, 1)).astype(np.float32)
    backend._his_joint_vel = np.zeros((num_envs, 10, 29), dtype=np.float32)
    backend._his_last_actions = np.zeros((num_envs, 10, 29), dtype=np.float32)
    backend._his_gravity = np.zeros((num_envs, 10, 3), dtype=np.float32)
    backend._his_gravity[:, :, 2] = -1.0
    return backend


def test_onnx_backend_reuses_planner_tokens_for_unchanged_commands():
    backend = _make_counting_onnx_backend(num_envs=2)
    commands = {
        "target_vel": torch.tensor([[0.4], [0.4]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2], [2]], dtype=torch.long),
        "height": torch.full((2, 1), -1.0, dtype=torch.float32),
    }

    backend.compute_joint_targets(commands)
    backend.compute_joint_targets(commands)

    assert backend.planner_session.run_count == 2
    assert backend.encoder_session.run_count == 2
    assert backend.decoder_session.run_count == 4



def test_torch_decoder_matches_release_onnx_decoder_for_batch():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    decoder_path = release_dir / "policy" / "release" / "model_decoder.onnx"
    torch_decoder = _backend_module.SonicTorchDecoder(decoder_path, device="cpu")
    onnx_backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=decoder_path,
    )
    obs = torch.linspace(-1.0, 1.0, 3 * 994, dtype=torch.float32).reshape(3, 994)

    torch_action = torch_decoder(obs).detach().numpy()
    onnx_action = np.concatenate([
        onnx_backend.decoder_session.run(["action"], {"obs_dict": obs[i : i + 1].numpy()})[0]
        for i in range(obs.shape[0])
    ], axis=0)

    np.testing.assert_allclose(torch_action, onnx_action, rtol=1e-5, atol=1e-5)


def test_torch_backend_returns_batched_joint_targets_without_onnx_sessions():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    backend = _backend_module.SonicTorchBackend(
        num_envs=128,
        action_dim=29,
        device="cpu",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    commands = {
        "target_vel": torch.full((128, 1), 0.4, dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(128, 1),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(128, 1),
        "mode": torch.full((128, 1), 2, dtype=torch.long),
        "height": torch.full((128, 1), -1.0, dtype=torch.float32),
    }
    low_level_observations = torch.zeros(128, 286, dtype=torch.float32)

    actions = backend.compute_joint_targets(commands, low_level_observations=low_level_observations)

    assert actions.shape == (128, 29)
    assert torch.isfinite(actions).all()


def test_torch_planner_encoder_matches_release_onnx_token_for_command():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    planner_path = release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"
    encoder_path = release_dir / "policy" / "release" / "model_encoder.onnx"
    torch_planner_encoder = _backend_module.SonicTorchPlannerEncoder(
        planner_path=planner_path,
        encoder_path=encoder_path,
        device="cpu",
    )
    onnx_backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=planner_path,
        encoder_path=encoder_path,
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    context = torch.zeros(1, 4, 36, dtype=torch.float32)
    context[:, :, 3] = 1.0
    context[:, :, 7:] = torch.as_tensor(onnx_backend.default_angles, dtype=torch.float32).reshape(1, 1, 29)
    commands = {
        "target_vel": torch.tensor([[0.6]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2]], dtype=torch.long),
        "height": torch.full((1, 1), -1.0, dtype=torch.float32),
    }

    torch_token = torch_planner_encoder(context, commands)
    commands_np = {name: tensor.numpy() for name, tensor in commands.items()}
    planner_qpos = onnx_backend._run_planner(0, commands_np)
    encoder_obs = onnx_backend._build_encoder_obs(planner_qpos)
    onnx_token = onnx_backend.encoder_session.run(["encoded_tokens"], {"obs_dict": encoder_obs})[0]

    assert torch_token.shape == (1, 64)
    np.testing.assert_allclose(torch_token.detach().numpy(), onnx_token, rtol=1e-4, atol=1e-4)


def test_torch_full_backend_matches_onnx_backend_first_step_for_single_env():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    planner_path = release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"
    encoder_path = release_dir / "policy" / "release" / "model_encoder.onnx"
    decoder_path = release_dir / "policy" / "release" / "model_decoder.onnx"
    torch_backend = _backend_module.SonicTorchFullBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=planner_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )
    onnx_backend = _backend_module.SonicOnnxBackend(
        num_envs=1,
        action_dim=29,
        device="cpu",
        planner_path=planner_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )
    commands = {
        "target_vel": torch.tensor([[0.6]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2]], dtype=torch.long),
        "height": torch.full((1, 1), -1.0, dtype=torch.float32),
    }
    low_level_observations = torch.zeros(1, 286, dtype=torch.float32)

    torch_targets = torch_backend.compute_joint_targets(commands, low_level_observations=low_level_observations)
    onnx_targets = onnx_backend.compute_joint_targets(commands, low_level_observations=low_level_observations)

    assert torch_targets.shape == (1, 29)
    np.testing.assert_allclose(torch_targets.detach().numpy(), onnx_targets.detach().numpy(), rtol=1e-4, atol=1e-4)


def test_torch_full_backend_supports_multiple_envs_with_fixed_batch_release_encoder():
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    backend = _backend_module.SonicTorchFullBackend(
        num_envs=3,
        action_dim=29,
        device="cpu",
        planner_path=release_dir / "planner" / "target_vel" / "V2" / "planner_sonic.onnx",
        encoder_path=release_dir / "policy" / "release" / "model_encoder.onnx",
        decoder_path=release_dir / "policy" / "release" / "model_decoder.onnx",
    )
    commands = {
        "target_vel": torch.tensor([[0.0], [0.4], [0.8]], dtype=torch.float32),
        "movement_direction": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[0], [1], [2]], dtype=torch.long),
        "height": torch.full((3, 1), -1.0, dtype=torch.float32),
    }
    low_level_observations = torch.zeros(3, 286, dtype=torch.float32)

    actions = backend.compute_joint_targets(commands, low_level_observations=low_level_observations)

    assert actions.shape == (3, 29)
    assert torch.isfinite(actions).all()


def test_motionbricks_backend_rejects_lfs_pointer_checkpoints(tmp_path):
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    encoder_path = release_dir / "policy" / "release" / "model_encoder.onnx"
    decoder_path = release_dir / "policy" / "release" / "model_decoder.onnx"
    motionbricks_root = tmp_path / "motionbricks"
    for rel in _backend_module.SonicMotionBricksBackend.required_checkpoint_relpaths:
        ckpt = motionbricks_root / rel
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"0" * 2048)
    pointer_ckpt = motionbricks_root / "out" / "motionbricks_pose" / "version_1" / "checkpoints" / "model-step=2000000.ckpt"
    pointer_ckpt.write_text("version https://git-lfs.github.com/spec/v1\n")

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        _backend_module.SonicMotionBricksBackend(
            num_envs=1,
            action_dim=29,
            device="cpu",
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            motionbricks_root=motionbricks_root,
        )


def test_motionbricks_backend_uses_original_planner_and_decoder(monkeypatch, tmp_path):
    release_dir = Path(__file__).resolve().parents[3] / "source" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    encoder_path = release_dir / "policy" / "release" / "model_encoder.onnx"
    decoder_path = release_dir / "policy" / "release" / "model_decoder.onnx"
    motionbricks_root = tmp_path / "motionbricks"
    for rel in _backend_module.SonicMotionBricksBackend.required_checkpoint_relpaths:
        ckpt = motionbricks_root / rel
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"0" * 2048)
    skeleton = motionbricks_root / "assets" / "skeletons" / "g1" / "g1.xml"
    skeleton.parent.mkdir(parents=True, exist_ok=True)
    skeleton.write_text("<mujoco/>\n")

    class _FakeFullAgent:
        def __init__(self):
            self.calls = []

        def reset(self):
            pass

        def generate_new_frames(self, inputs, controller_dt=0.25, force_generation=False):
            del controller_dt, force_generation
            self.calls.append(inputs)
            batch = inputs["context_mujoco_qpos"].shape[0]
            qpos = torch.zeros(batch, 64, 36, dtype=torch.float32)
            qpos[:, :, 3] = 1.0
            qpos[:, :, 7:] = inputs["context_mujoco_qpos"][:, -1:, 7:]
            qpos[:, :, 7] += inputs["target_vel"].reshape(batch, 1)
            return qpos, torch.full((batch,), 64)

    fake_agent = _FakeFullAgent()

    def _fake_create_agent(self):
        return fake_agent

    monkeypatch.setattr(_backend_module.SonicMotionBricksBackend, "_create_full_agent", _fake_create_agent)

    backend = _backend_module.SonicMotionBricksBackend(
        num_envs=2,
        action_dim=29,
        device="cpu",
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        motionbricks_root=motionbricks_root,
    )
    commands = {
        "target_vel": torch.tensor([[0.3], [0.7]], dtype=torch.float32),
        "movement_direction": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "facing_direction": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "mode": torch.tensor([[2], [2]], dtype=torch.long),
        "height": torch.full((2, 1), -1.0, dtype=torch.float32),
    }
    low_level_observations = torch.zeros(2, 286, dtype=torch.float32)

    actions = backend.compute_joint_targets(commands, low_level_observations=low_level_observations)

    assert actions.shape == (2, 29)
    assert torch.isfinite(actions).all()
    assert len(fake_agent.calls) == 1
    assert fake_agent.calls[0]["context_mujoco_qpos"].shape == (2, 4, 36)
    assert backend._token_state.shape == (2, 64)

