from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch


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
