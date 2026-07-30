from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm

from .instinct_frozen_backend import InstinctFrozenPolicyBackend, OfficialOnnxFrozenPolicyBackend

if TYPE_CHECKING:
    from .instinct_navigation_actions_cfg import InstinctFrozenNavigationSE2ActionCfg


class InstinctFrozenNavigationSE2Action(ActionTerm):
    """SE(2) high-level action using the original frozen InstinctLab G1 low-level policy."""

    cfg: InstinctFrozenNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: InstinctFrozenNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
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
        if self.cfg.backend_type == "checkpoint":
            self._backend = InstinctFrozenPolicyBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                checkpoint_path=self.cfg.checkpoint_path,
                instinct_rl_root=self.cfg.instinct_rl_root,
            )
        elif self.cfg.backend_type == "official_onnx":
            if not self.cfg.actor_onnx_path or not self.cfg.depth_encoder_onnx_path:
                raise ValueError("official_onnx backend requires actor_onnx_path and depth_encoder_onnx_path")
            self._backend = OfficialOnnxFrozenPolicyBackend(
                self.num_envs,
                self.low_level_position_action_term.action_dim,
                self.device,
                actor_onnx_path=self.cfg.actor_onnx_path,
                depth_encoder_onnx_path=self.cfg.depth_encoder_onnx_path,
            )
        else:
            raise ValueError(f"Unknown Instinct frozen backend_type: {self.cfg.backend_type!r}")

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

    def _write_base_velocity_command(self):
        command_term = self._env.command_manager.get_term(self.cfg.command_name)
        if command_term is None or not hasattr(command_term, "vel_command_b"):
            raise RuntimeError(f"Command term {self.cfg.command_name!r} must expose vel_command_b like InstinctLab PoseVelocityCommand")
        command_term.vel_command_b[:] = self._processed_navigation_velocity_actions

    def process_actions(self, actions):
        self._raw_navigation_velocity_actions[:] = actions
        if self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions
        else:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions * self._scale + self._offset
        if self.cfg.policy_distr_type == "gaussian":
            self._processed_navigation_velocity_actions[:] = torch.tanh(self._processed_navigation_velocity_actions)
        elif self.cfg.policy_distr_type == "beta":
            self._processed_navigation_velocity_actions[:] = (self._processed_navigation_velocity_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")
        observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
        if isinstance(observations, dict):
            base_ang_vel = observations["base_ang_vel"][:, -3:] if observations["base_ang_vel"].shape[1] >= 3 else observations["base_ang_vel"]
            vel_xyz = base_ang_vel.norm(dim=1, keepdim=True) * 0.0
        else:
            vel_xyz = torch.zeros(self.num_envs, 1, device=self.device)
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
        self._write_base_velocity_command()

    @torch.inference_mode()
    def apply_actions(self):
        self._write_base_velocity_command()
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions[:] = self._low_level_position_actions
            observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group, update_history=True)
            if isinstance(observations, dict):
                observations = torch.cat([observations[name].reshape(self.num_envs, -1) for name in self._backend.obs_format()["policy"]], dim=1)
            self._low_level_position_actions[:] = self._backend.compute_raw_actions(observations)
            self.low_level_position_action_term.process_actions(self._low_level_position_actions)
        self.low_level_position_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: torch.Tensor | None = None):
        self._counter = 0
        if env_ids is None:
            self._prev_filtered_velocity_commands.zero_()
            self._raw_navigation_velocity_actions.zero_()
            self._processed_navigation_velocity_actions.zero_()
            self._low_level_position_actions.zero_()
            self._prev_low_level_position_actions.zero_()
        else:
            self._prev_filtered_velocity_commands[env_ids] = 0.0
            self._raw_navigation_velocity_actions[env_ids] = 0.0
            self._processed_navigation_velocity_actions[env_ids] = 0.0
            self._low_level_position_actions[env_ids] = 0.0
            self._prev_low_level_position_actions[env_ids] = 0.0
        self.low_level_position_action_term.reset(env_ids)
