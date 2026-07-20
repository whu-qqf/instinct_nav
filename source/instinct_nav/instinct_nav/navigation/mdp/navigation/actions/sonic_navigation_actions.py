from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm

from .sonic_backend import (
    DeterministicSonicBackend,
    InstinctFrozenG1Backend,
    SonicMotionBricksBackend,
    SonicOnnxBackend,
    SonicTorchBackend,
    SonicTorchFullBackend,
)
from .sonic_command_adapter import SonicCommandAdapter

if TYPE_CHECKING:
    from .sonic_navigation_actions_cfg import SonicNavigationSE2ActionCfg


class SonicNavigationSE2Action(ActionTerm):
    """High-level SE(2) navigation action backed by a frozen SONIC-style low-level controller."""

    cfg: SonicNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SonicNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.low_level_position_action_term: ActionTerm = self.cfg.low_level_position_action.class_type(
            cfg.low_level_position_action, env
        )
        self._action_dim = 3
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros_like(self._raw_navigation_velocity_actions)
        self._prev_filtered_velocity_commands = torch.zeros_like(self._raw_navigation_velocity_actions)
        self._low_level_position_actions = torch.zeros(
            self.num_envs, self.low_level_position_action_term.action_dim, device=self.device
        )
        self._prev_low_level_position_actions = torch.zeros_like(self._low_level_position_actions)
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._policy_scaling = torch.tensor(self.cfg.policy_scaling, device=self.device).repeat(self.num_envs, 1)
        self._policy_bias = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._counter = 0
        self._adapter = SonicCommandAdapter(self.num_envs, self.device, max_speed=self.cfg.max_speed)
        if self.cfg.backend == "fake":
            self._backend = DeterministicSonicBackend(
                self.num_envs, self.low_level_position_action_term.action_dim, self.device
            )
        elif self.cfg.backend == "onnx":
            missing_paths = [
                name
                for name in ("planner_path", "encoder_path", "decoder_path")
                if getattr(self.cfg, name) in (None, "")
            ]
            if missing_paths:
                raise ValueError(f"SONIC ONNX backend requires model path fields: {', '.join(missing_paths)}")
            self._backend = SonicOnnxBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                planner_path=self.cfg.planner_path,
                encoder_path=self.cfg.encoder_path,
                decoder_path=self.cfg.decoder_path,
            )
        elif self.cfg.backend == "torch":
            if self.cfg.decoder_path in (None, ""):
                raise ValueError("SONIC torch backend requires decoder_path")
            self._backend = SonicTorchBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                decoder_path=self.cfg.decoder_path,
            )
        elif self.cfg.backend == "torch_full":
            missing_paths = [
                name
                for name in ("planner_path", "encoder_path", "decoder_path")
                if getattr(self.cfg, name) in (None, "")
            ]
            if missing_paths:
                raise ValueError(f"SONIC torch_full backend requires model path fields: {', '.join(missing_paths)}")
            self._backend = SonicTorchFullBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                planner_path=self.cfg.planner_path,
                encoder_path=self.cfg.encoder_path,
                decoder_path=self.cfg.decoder_path,
            )
        elif self.cfg.backend == "motionbricks":
            missing_paths = [
                name
                for name in ("encoder_path", "decoder_path")
                if getattr(self.cfg, name) in (None, "")
            ]
            if missing_paths:
                raise ValueError(f"SONIC motionbricks backend requires model path fields: {', '.join(missing_paths)}")
            self._backend = SonicMotionBricksBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                encoder_path=self.cfg.encoder_path,
                decoder_path=self.cfg.decoder_path,
                motionbricks_root=self.cfg.motionbricks_root,
            )
        elif self.cfg.backend == "instinct":
            if self.cfg.instinct_checkpoint_path in (None, ""):
                raise ValueError("Instinct backend requires instinct_checkpoint_path")
            self._backend = InstinctFrozenG1Backend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                checkpoint_path=self.cfg.instinct_checkpoint_path,
                instinct_rl_root=self.cfg.instinct_rl_root,
                depth_mode=self.cfg.instinct_depth_mode,
            )
        else:
            raise ValueError(
                f"Unsupported SONIC backend '{self.cfg.backend}'. Expected 'fake', 'onnx', 'torch', "
                "'torch_full', 'motionbricks', or 'instinct'."
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    def process_actions(self, actions):
        self._raw_navigation_velocity_actions[:] = actions
        if self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions
        else:
            self._processed_navigation_velocity_actions[:] = (
                self._raw_navigation_velocity_actions * self._scale + self._offset
            )

        if self.cfg.policy_distr_type == "gaussian":
            self._processed_navigation_velocity_actions[:] = torch.tanh(self._processed_navigation_velocity_actions)
        elif self.cfg.policy_distr_type == "beta":
            self._processed_navigation_velocity_actions[:] = (self._processed_navigation_velocity_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
        base_lin_vel = observations[:, 0:3]
        vel_xyz = base_lin_vel.norm(dim=1, keepdim=True)
        self._processed_navigation_velocity_actions[:] = (
            self._processed_navigation_velocity_actions + vel_xyz * self._policy_bias
        ) * self._policy_scaling
        if self.cfg.enable_low_pass_filter:
            alpha = self.cfg.low_pass_filter_alpha
            self._processed_navigation_velocity_actions[:] = (
                alpha * self._prev_filtered_velocity_commands
                + (1.0 - alpha) * self._processed_navigation_velocity_actions
            )
            self._prev_filtered_velocity_commands[:] = self._processed_navigation_velocity_actions

    @torch.inference_mode()
    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions[:] = self._low_level_position_actions
            commands = self._adapter.convert(
                self._processed_navigation_velocity_actions,
                dt=self.cfg.low_level_decimation * self._env.physics_dt,
            )
            commands["velocity_command"] = self._processed_navigation_velocity_actions
            low_level_observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
            self._low_level_position_actions[:] = self._backend.compute_joint_targets(
                commands,
                low_level_observations=low_level_observations,
            )
            self.low_level_position_action_term.process_actions(self._low_level_position_actions)

        self.low_level_position_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: torch.Tensor | None = None):
        self._counter = 0
        if env_ids is None:
            self._prev_filtered_velocity_commands.zero_()
            self._adapter.reset()
        else:
            self._prev_filtered_velocity_commands[env_ids] = 0.0
            self._adapter.reset(env_ids)
