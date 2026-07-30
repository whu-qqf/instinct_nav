from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


class InstinctFrozenPolicyBackend:
    """Frozen InstinctLab G1 parkour actor that returns raw actions.

    The returned action is intentionally not converted to absolute joint targets.
    It must be passed through the same JointPositionActionCfg used during
    InstinctLab training: scale=beyondmimic_action_scale and use_default_offset=True.
    """

    depth_shape = (8, 18, 32)

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        checkpoint_path: str | Path,
        instinct_rl_root: str | Path | None = None,
    ):
        if action_dim != 29:
            raise ValueError(f"Instinct frozen policy action_dim must be 29, got {action_dim}")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))

        if instinct_rl_root is None:
            instinct_rl_root = os.environ.get(
                "SRU_INSTINCT_RL_ROOT",
                str(Path(__file__).resolve().parents[7] / "external" / "instinct_rl"),
            )
        instinct_rl_path = Path(instinct_rl_root) if instinct_rl_root else None
        if instinct_rl_path and str(instinct_rl_path) not in sys.path:
            sys.path.insert(0, str(instinct_rl_path))
        try:
            from instinct_rl import modules
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Instinct frozen backend requires instinct_rl. Set instinct_rl_root to the local instinct_rl checkout."
            ) from exc

        self.policy = modules.build_actor_critic(
            "EncoderMoEActorCritic",
            self._policy_cfg(),
            self.obs_format(),
            num_actions=29,
            num_rewards=1,
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

    @classmethod
    def obs_format(cls):
        policy = OrderedDict(
            [
                ("base_ang_vel", (24,)),
                ("projected_gravity", (24,)),
                ("velocity_commands", (24,)),
                ("joint_pos", (232,)),
                ("joint_vel", (232,)),
                ("actions", (232,)),
                ("depth_image", cls.depth_shape),
            ]
        )
        critic = OrderedDict([("base_lin_vel", (24,)), *policy.items()])
        return {"policy": policy, "critic": critic}

    @staticmethod
    def _policy_cfg():
        depth_encoder = {
            "class_name": "Conv2dHeadModel",
            "component_names": ["depth_image"],
            "output_size": 128,
            "takeout_input_components": True,
            "channels": [4],
            "kernel_sizes": [3],
            "strides": [1],
            "hidden_sizes": [256, 256],
            "paddings": [1],
            "nonlinearity": "ReLU",
            "use_maxpool": True,
        }
        return {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
            "num_moe_experts": 4,
            "moe_gate_hidden_dims": [],
            "encoder_configs": {"depth_encoder": depth_encoder.copy()},
            "critic_encoder_configs": {"depth_encoder": depth_encoder.copy()},
        }

    @classmethod
    def expected_obs_dim(cls) -> int:
        return 24 + 24 + 24 + 232 + 232 + 232 + 8 * 18 * 32

    def _validate_obs(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device, dtype=torch.float32)
        expected_shape = (self.num_envs, self.expected_obs_dim())
        if tuple(observations.shape) != expected_shape:
            raise ValueError(f"Instinct policy observation shape {tuple(observations.shape)} != {expected_shape}")
        return observations

    @torch.inference_mode()
    def compute_raw_actions(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy.act_inference(self._validate_obs(observations))


class OfficialOnnxFrozenPolicyBackend:
    """GPU PyTorch reconstruction of the official onboard ONNX policy pair."""

    depth_shape = InstinctFrozenPolicyBackend.depth_shape
    proprio_dim = 24 + 24 + 24 + 232 + 232 + 232

    def __init__(self, num_envs, action_dim, device, actor_onnx_path, depth_encoder_onnx_path):
        if action_dim != 29:
            raise ValueError(f"Official frozen policy action_dim must be 29, got {action_dim}")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.actor_onnx_path = Path(actor_onnx_path)
        self.depth_encoder_onnx_path = Path(depth_encoder_onnx_path)
        for path in (self.actor_onnx_path, self.depth_encoder_onnx_path):
            if not path.is_file():
                raise FileNotFoundError(str(path))

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 18 * 32, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        ).to(self.device)
        self.actor = nn.Sequential(
            nn.Linear(896, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29),
        ).to(self.device)
        self._load_onnx_weights()
        for module in (self.depth_encoder, self.actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def obs_format(cls):
        return InstinctFrozenPolicyBackend.obs_format()

    @classmethod
    def expected_obs_dim(cls):
        return InstinctFrozenPolicyBackend.expected_obs_dim()

    @staticmethod
    def _initializers(path):
        try:
            import onnx
            from onnx import numpy_helper
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Official ONNX frozen backend requires the 'onnx' package to import weights once at startup."
            ) from exc
        model = onnx.load(str(path))
        return {item.name: torch.from_numpy(numpy_helper.to_array(item).copy()) for item in model.graph.initializer}

    @staticmethod
    def _copy_linear(layer, weights, prefix):
        layer.weight.data.copy_(weights[f"{prefix}.weight"].to(layer.weight))
        layer.bias.data.copy_(weights[f"{prefix}.bias"].to(layer.bias))

    def _load_onnx_weights(self):
        depth = self._initializers(self.depth_encoder_onnx_path)
        conv = self.depth_encoder[0]
        conv.weight.data.copy_(depth["conv.conv.0.weight"].to(conv.weight))
        conv.bias.data.copy_(depth["conv.conv.0.bias"].to(conv.bias))
        self._copy_linear(self.depth_encoder[3], depth, "head.model.0")
        self._copy_linear(self.depth_encoder[5], depth, "head.model.2")
        self._copy_linear(self.depth_encoder[7], depth, "head.model.4")
        actor = self._initializers(self.actor_onnx_path)
        self._copy_linear(self.actor[0], actor, "0")
        self._copy_linear(self.actor[2], actor, "2")
        self._copy_linear(self.actor[4], actor, "4")
        self._copy_linear(self.actor[6], actor, "6")

    def _validate_obs(self, observations):
        observations = observations.to(self.device, dtype=torch.float32)
        expected_shape = (self.num_envs, self.expected_obs_dim())
        if tuple(observations.shape) != expected_shape:
            raise ValueError(f"Official policy observation shape {tuple(observations.shape)} != {expected_shape}")
        return observations

    @torch.inference_mode()
    def compute_raw_actions(self, observations):
        observations = self._validate_obs(observations)
        proprio = observations[:, : self.proprio_dim]
        depth = observations[:, self.proprio_dim :].reshape(self.num_envs, *self.depth_shape)
        depth_latent = self.depth_encoder(depth)
        return self.actor(torch.cat((proprio, depth_latent), dim=1))
