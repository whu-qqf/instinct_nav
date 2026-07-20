from __future__ import annotations

import os
import sys
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class SonicLowLevelBackend(Protocol):
    """Interface for frozen SONIC-style low-level controllers."""

    def compute_joint_targets(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return joint target tensor with shape [num_envs, action_dim]."""


class InstinctFrozenG1Backend:
    """Frozen InstinctLab G1 parkour policy adapted to the SRU velocity-command interface."""

    sru_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    instinct_joint_names = [
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "waist_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "waist_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "waist_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]
    default_joint_pos_by_name = {
        "left_hip_pitch_joint": -0.312,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.312,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.6,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.6,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    history_len = 8
    depth_shape = (8, 18, 32)

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        checkpoint_path: str | Path,
        instinct_rl_root: str | Path | None = None,
        depth_mode: str = "strict",
    ):
        if action_dim != 29:
            raise ValueError(f"InstinctFrozenG1Backend action_dim must be 29, got {action_dim}")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.depth_mode = depth_mode
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(str(self.checkpoint_path))

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
                "Instinct backend requires instinct_rl. Set instinct_rl_root to the local instinct_rl checkout."
            ) from exc

        self.sru_to_instinct = torch.as_tensor(
            [self.sru_joint_names.index(name) for name in self.instinct_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.instinct_to_sru = torch.as_tensor(
            [self.instinct_joint_names.index(name) for name in self.sru_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.default_angles_instinct = torch.as_tensor(
            [self.default_joint_pos_by_name[name] for name in self.instinct_joint_names],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 29)
        self.default_angles_sru = torch.as_tensor(
            [self.default_joint_pos_by_name[name] for name in self.sru_joint_names],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 29)
        self.action_scale_instinct = torch.as_tensor(
            [self._beyondmimic_action_scale(name) for name in self.instinct_joint_names],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 29)

        self.policy = modules.build_actor_critic(
            "EncoderMoEActorCritic",
            self._policy_cfg(),
            self._obs_format(),
            num_actions=29,
            num_rewards=1,
        ).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        self._his_base_ang_vel = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self._his_projected_gravity = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self._his_projected_gravity[:, :, 2] = -1.0
        self._his_velocity_commands = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self._his_joint_pos = torch.zeros(self.num_envs, self.history_len, 29, device=self.device)
        self._his_joint_vel = torch.zeros(self.num_envs, self.history_len, 29, device=self.device)
        self._his_actions = torch.zeros(self.num_envs, self.history_len, 29, device=self.device)
        self._his_depth = torch.zeros(self.num_envs, *self.depth_shape, device=self.device)

    @staticmethod
    def _beyondmimic_action_scale(joint_name: str) -> float:
        armature_5020 = 0.003609725
        armature_7520_14 = 0.010177520
        armature_7520_22 = 0.025101925
        armature_4010 = 0.00425
        natural_freq = 10.0 * 2.0 * 3.1415926535
        stiffness_5020 = armature_5020 * natural_freq**2
        stiffness_7520_14 = armature_7520_14 * natural_freq**2
        stiffness_7520_22 = armature_7520_22 * natural_freq**2
        stiffness_4010 = armature_4010 * natural_freq**2
        if "_hip_pitch_joint" in joint_name or "_hip_yaw_joint" in joint_name:
            return 0.25 * 88.0 / stiffness_7520_14
        if "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
            return 0.25 * 139.0 / stiffness_7520_22
        if "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
            return 0.25 * 50.0 / (2.0 * stiffness_5020)
        if joint_name in ("waist_roll_joint", "waist_pitch_joint"):
            return 0.25 * 50.0 / (2.0 * stiffness_5020)
        if joint_name == "waist_yaw_joint":
            return 0.25 * 88.0 / stiffness_7520_14
        if "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            return 0.25 * 5.0 / stiffness_4010
        return 0.25 * 25.0 / stiffness_5020

    @classmethod
    def _obs_format(cls):
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

    def _roll_histories(self):
        self._his_base_ang_vel = torch.roll(self._his_base_ang_vel, shifts=-1, dims=1)
        self._his_projected_gravity = torch.roll(self._his_projected_gravity, shifts=-1, dims=1)
        self._his_velocity_commands = torch.roll(self._his_velocity_commands, shifts=-1, dims=1)
        self._his_joint_pos = torch.roll(self._his_joint_pos, shifts=-1, dims=1)
        self._his_joint_vel = torch.roll(self._his_joint_vel, shifts=-1, dims=1)
        self._his_actions = torch.roll(self._his_actions, shifts=-1, dims=1)

    def _extract_depth_image(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor,
    ) -> torch.Tensor:
        del low_level_observations
        depth_mode = getattr(self, "depth_mode", "strict")
        if depth_mode != "strict":
            raise ValueError("InstinctFrozenG1Backend only supports depth_mode='strict' for faithful execution")
        depth_image = commands.get("depth_image")
        if depth_image is None:
            raise ValueError(
                "InstinctFrozenG1Backend requires commands['depth_image'] with shape "
                f"[num_envs, {self.depth_shape[0]}, {self.depth_shape[1]}, {self.depth_shape[2]}]. "
                "Height-scan or zero-depth substitution is not allowed."
            )
        depth_image = depth_image.to(self.device, dtype=torch.float32)
        expected_shape = (self.num_envs, *self.depth_shape)
        if tuple(depth_image.shape) != expected_shape:
            raise ValueError(f"commands['depth_image'] shape {tuple(depth_image.shape)} != {expected_shape}")
        return depth_image

    def _update_history(self, commands: dict[str, torch.Tensor], low_level_observations: torch.Tensor):
        observations = low_level_observations.to(self.device, dtype=torch.float32)
        if observations.shape[0] != self.num_envs or observations.shape[1] < 99:
            raise ValueError(
                "low_level_observations must have shape [num_envs, >=99] with SRU G1 low-level terms"
            )
        self._roll_histories()
        self._his_base_ang_vel[:, -1] = observations[:, 3:6] * 0.25
        self._his_projected_gravity[:, -1] = observations[:, 6:9]
        self._his_velocity_commands[:, -1] = commands["velocity_command"].to(self.device, dtype=torch.float32)
        self._his_joint_pos[:, -1] = observations[:, 12:41][:, self.sru_to_instinct]
        self._his_joint_vel[:, -1] = observations[:, 41:70][:, self.sru_to_instinct] * 0.05
        self._his_depth = self._extract_depth_image(commands, observations)

    def _build_actor_obs(self) -> torch.Tensor:
        return torch.cat(
            [
                self._his_base_ang_vel.reshape(self.num_envs, -1),
                self._his_projected_gravity.reshape(self.num_envs, -1),
                self._his_velocity_commands.reshape(self.num_envs, -1),
                self._his_joint_pos.reshape(self.num_envs, -1),
                self._his_joint_vel.reshape(self.num_envs, -1),
                self._his_actions.reshape(self.num_envs, -1),
                self._his_depth.reshape(self.num_envs, -1),
            ],
            dim=1,
        )

    @torch.inference_mode()
    def compute_joint_targets(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if low_level_observations is None:
            raise ValueError("InstinctFrozenG1Backend requires low_level_observations")
        self._update_history(commands, low_level_observations)
        action_instinct = self.policy.act_inference(self._build_actor_obs())
        joint_target_instinct = self.default_angles_instinct + action_instinct * self.action_scale_instinct
        self._his_actions[:, -1] = action_instinct
        return joint_target_instinct[:, self.instinct_to_sru]


class DeterministicSonicBackend:
    """Deterministic test backend used to validate the IsaacLab action contract."""

    def __init__(self, num_envs: int, action_dim: int, device: str | torch.device):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.joint_pattern = torch.linspace(-1.0, 1.0, action_dim, device=self.device).unsqueeze(0)

    def compute_joint_targets(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del low_level_observations
        target_vel = commands["target_vel"].to(self.device).clamp(min=0.0)
        self.phase = self.phase + 0.1 + target_vel[:, 0] * 0.05
        amplitude = target_vel.clamp(max=1.0)
        return amplitude * torch.sin(self.phase).unsqueeze(1) * self.joint_pattern



class _OnnxScatterElements(torch.nn.Module):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        output = data.clone()
        axis = self.axis if self.axis >= 0 else data.dim() + self.axis
        return output.scatter(axis, indices.to(torch.long), updates.to(data.dtype))


class _OnnxOneHot(torch.nn.Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        depth_int = int(depth.reshape(-1)[0].item())
        off_value = values.reshape(-1)[0]
        on_value = values.reshape(-1)[1]
        result = torch.nn.functional.one_hot(indices.to(torch.long), num_classes=depth_int).to(values.dtype)
        result = result * (on_value - off_value) + off_value
        if self.axis != -1 and self.axis != result.dim() - 1:
            axis = self.axis if self.axis >= 0 else result.dim() + self.axis
            result = torch.movedim(result, -1, axis)
        return result


def _register_onnx2torch_one_hot_converter():
    try:
        from onnx2torch.node_converters.registry import add_converter, _CONVERTER_REGISTRY, OperationDescription
        from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SonicTorchPlannerEncoder requires onnx2torch in the active instinct_nav runtime environment."
        ) from exc

    if OperationDescription("", "OneHot", 11) not in _CONVERTER_REGISTRY:
        @add_converter(operation_type="OneHot", version=11)
        def _one_hot_converter(node, graph):
            del graph
            return OperationConverterResult(
                torch_module=_OnnxOneHot(axis=node.attributes.get("axis", -1)),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

    from onnx2torch.node_converters.comparisons import OnnxCompare

    for operation_type in ("LessOrEqual", "GreaterOrEqual"):
        if OperationDescription("", operation_type, 16) in _CONVERTER_REGISTRY:
            continue

        @add_converter(operation_type=operation_type, version=16)
        def _compare_converter(node, graph):
            del graph
            return OperationConverterResult(
                torch_module=OnnxCompare(operation_type=node.operation_type),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

    if OperationDescription("", "ScatterElements", 16) not in _CONVERTER_REGISTRY:
        @add_converter(operation_type="ScatterElements", version=16)
        def _scatter_elements_converter(node, graph):
            del graph
            reduction = node.attributes.get("reduction", "none")
            if reduction != "none":
                raise NotImplementedError(f"ScatterElements reduction={reduction!r} is not supported")
            return OperationConverterResult(
                torch_module=_OnnxScatterElements(axis=node.attributes.get("axis", 0)),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

    from onnx2torch.node_converters.flatten import OnnxFlatten

    if not getattr(OnnxFlatten, "_sonic_safe_forward", False):
        def _safe_flatten_forward(self, x):
            axis = self.axis if self.axis >= 0 else x.dim() + self.axis
            if axis > x.dim():
                x = x.reshape(*x.shape, *([1] * (axis - x.dim())))
            x = torch.flatten(x, end_dim=axis - 1)
            if x.dim() == 1:
                return x.reshape(x.shape[0], 1)
            return torch.flatten(x, start_dim=1)

        OnnxFlatten.forward = _safe_flatten_forward
        OnnxFlatten._sonic_safe_forward = True

    from onnx2torch.node_converters.reshape import OnnxReshape

    if not getattr(OnnxReshape, "_sonic_safe_do_reshape", False):
        @staticmethod
        def _safe_do_reshape(input_tensor, shape):
            if torch.any(shape == 0):
                shape = torch.as_tensor([
                    input_tensor.shape[i] if int(dim_size) == 0 else int(dim_size)
                    for i, dim_size in enumerate(shape)
                ], device=shape.device, dtype=shape.dtype)
            else:
                shape = shape.to(dtype=torch.long)

            try:
                return torch.reshape(input_tensor, torch.Size(shape.tolist()))
            except RuntimeError:
                if shape.numel() > 0 and input_tensor.ndim > 0:
                    batch = input_tensor.shape[0]
                    original_product = int(torch.prod(shape).item())
                    if int(shape[0].item()) == 1:
                        candidate = shape.clone()
                        candidate[0] = batch
                        if int(torch.prod(candidate).item()) == input_tensor.numel():
                            return torch.reshape(input_tensor, torch.Size(candidate.tolist()))
                        if original_product > 0 and input_tensor.numel() % original_product == 0:
                            inferred_batch = input_tensor.numel() // original_product
                            candidate = shape.clone()
                            candidate[0] = inferred_batch
                            if int(torch.prod(candidate).item()) == input_tensor.numel():
                                return torch.reshape(input_tensor, torch.Size(candidate.tolist()))
                    candidate = torch.cat([torch.as_tensor([batch], device=shape.device, dtype=shape.dtype), shape])
                    if int(torch.prod(candidate).item()) == input_tensor.numel():
                        return torch.reshape(input_tensor, torch.Size(candidate.tolist()))
                    if original_product > 0 and input_tensor.numel() % original_product == 0:
                        candidate = torch.cat([
                            torch.as_tensor([input_tensor.numel() // original_product], device=shape.device, dtype=shape.dtype),
                            shape,
                        ])
                        if int(torch.prod(candidate).item()) == input_tensor.numel():
                            return torch.reshape(input_tensor, torch.Size(candidate.tolist()))
                raise

        OnnxReshape._do_reshape = _safe_do_reshape
        OnnxReshape._sonic_safe_do_reshape = True

    from onnx2torch.node_converters.concat import OnnxConcat

    if not getattr(OnnxConcat, "_sonic_safe_forward", False):
        def _safe_concat_forward(self, *input_tensors):
            axis = self.axis if self.axis >= 0 else max(t.dim() for t in input_tensors) + self.axis
            target_batch = max((t.shape[0] for t in input_tensors if t.ndim > 0), default=1)
            expanded = []
            for tensor in input_tensors:
                if tensor.ndim > 0 and axis != 0 and target_batch > 1:
                    if tensor.shape[0] == 1:
                        shape = list(tensor.shape)
                        shape[0] = target_batch
                        tensor = tensor.expand(*shape)
                    elif target_batch % tensor.shape[0] == 0 and tensor.shape[0] != target_batch:
                        tensor = tensor.repeat_interleave(target_batch // tensor.shape[0], dim=0)
                expanded.append(tensor)
            return torch.cat(expanded, self.axis)

        OnnxConcat.forward = _safe_concat_forward
        OnnxConcat._sonic_safe_forward = True

    from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation

    if not getattr(OnnxBinaryMathOperation, "_sonic_safe_forward", False):
        def _safe_binary_forward(self, first, second):
            if self.broadcast == 1 and self.axis is not None:
                from onnx2torch.utils.common import old_style_broadcast
                second = old_style_broadcast(first, second, self.axis)
            try:
                return self.math_op_function(first, second)
            except RuntimeError:
                if first.ndim == 2 and first.shape[1] == 1 and second.ndim > 2 and first.shape[0] == second.shape[0]:
                    return self.math_op_function(first.reshape(first.shape[0], *([1] * (second.ndim - 1))), second)
                if second.ndim == 2 and second.shape[1] == 1 and first.ndim > 2 and second.shape[0] == first.shape[0]:
                    return self.math_op_function(first, second.reshape(second.shape[0], *([1] * (first.ndim - 1))))
                if (
                    first.ndim == second.ndim + 1
                    and first.ndim >= 3
                    and first.shape[0] == second.shape[0]
                    and tuple(first.shape[2:]) == tuple(second.shape[1:])
                ):
                    return self.math_op_function(first, second.unsqueeze(1))
                if (
                    second.ndim == first.ndim + 1
                    and second.shape[0] == first.shape[0]
                ):
                    if first.ndim == 1:
                        return self.math_op_function(first.unsqueeze(-1), second)
                    if second.ndim >= 3 and tuple(second.shape[2:]) == tuple(first.shape[1:]):
                        return self.math_op_function(first.unsqueeze(1), second)
                if (
                    first.ndim == second.ndim + 1
                    and first.shape[0] == second.shape[0]
                    and second.ndim == 1
                ):
                    return self.math_op_function(first, second.unsqueeze(-1))
                if first.ndim == 2 and second.ndim == 2 and first.shape[0] == second.shape[0]:
                    return self.math_op_function(first.unsqueeze(-1), second.unsqueeze(1))
                if (
                    first.ndim >= 2
                    and second.ndim >= 1
                    and first.shape[-1] > 1
                    and second.shape[-1] % first.shape[-1] == 0
                ):
                    second_candidate = second.reshape(
                        *second.shape[:-1],
                        second.shape[-1] // first.shape[-1],
                        first.shape[-1],
                    )
                    return self.math_op_function(first, second_candidate)
                if (
                    second.ndim >= 2
                    and first.ndim >= 1
                    and second.shape[-1] > 1
                    and first.shape[-1] % second.shape[-1] == 0
                ):
                    first_candidate = first.reshape(
                        *first.shape[:-1],
                        first.shape[-1] // second.shape[-1],
                        second.shape[-1],
                    )
                    return self.math_op_function(first_candidate, second)
                raise

        OnnxBinaryMathOperation.forward = _safe_binary_forward
        OnnxBinaryMathOperation._sonic_safe_forward = True

    from onnx2torch.node_converters.scatter_nd import OnnxScatterND

    if not getattr(OnnxScatterND, "_sonic_safe_forward", False):
        def _safe_scatter_nd_forward(self, data, indices, updates):
            output = data.clone()
            try:
                ind_dim = indices.dim()
                output_indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
                output_updates = updates.reshape((-1, *updates.shape[ind_dim - 1 :]))
                output[output_indices] = output_updates
                return output
            except RuntimeError:
                if data.ndim >= 2 and updates.ndim >= 1 and updates.shape[0] == data.shape[0]:
                    local_indices = indices
                    if indices.shape[-1] in (data.ndim, data.ndim - 1) and torch.all(indices[..., 0] == 0):
                        local_indices = indices[..., 1:]
                    flat_indices = local_indices.reshape((-1, local_indices.shape[-1]))
                    output_indices = flat_indices.T.tolist()
                    num_indexed = flat_indices.shape[0]
                    value_tail = tuple(data.shape[1 + local_indices.shape[-1] :])
                    if data.shape[0] == 1:
                        flat_updates = updates.reshape(-1, *value_tail)
                        if flat_updates.shape[0] % num_indexed == 0:
                            inferred_batch = flat_updates.shape[0] // num_indexed
                            output = output.expand(inferred_batch, *output.shape[1:]).clone()
                            flat_updates = flat_updates.reshape(inferred_batch, num_indexed, *value_tail)
                            for batch_id in range(inferred_batch):
                                output[batch_id][tuple(output_indices)] = flat_updates[batch_id]
                            return output
                    flat_updates = updates.reshape(data.shape[0], num_indexed, *value_tail)
                    for batch_id in range(data.shape[0]):
                        output[batch_id][tuple(output_indices)] = flat_updates[batch_id]
                    return output
                    for batch_id in range(data.shape[0]):
                        batch_updates = updates[batch_id].reshape((-1, *updates[batch_id].shape[ind_dim - 1 :]))
                        output[batch_id][tuple(output_indices)] = batch_updates
                    return output
                raise

        OnnxScatterND.forward = _safe_scatter_nd_forward
        OnnxScatterND._sonic_safe_forward = True


def _strip_trailing_empty_onnx_inputs(model):
    for node in model.graph.node:
        while node.input and node.input[-1] == "":
            node.input.pop()
    return model


class SonicTorchPlannerEncoder(torch.nn.Module):
    """Torch-converted release SONIC planner and encoder token generator."""

    allowed_pred_num_tokens = torch.as_tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)

    def __init__(self, planner_path: str | Path, encoder_path: str | Path, device: str | torch.device):
        super().__init__()
        try:
            import onnx
            from onnx2torch import convert
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SonicTorchPlannerEncoder requires onnx and onnx2torch in the SRU venv."
            ) from exc

        _register_onnx2torch_one_hot_converter()
        self.device = torch.device(device)
        self.planner = convert(_strip_trailing_empty_onnx_inputs(onnx.load(str(planner_path)))).eval().to(self.device)
        self.encoder = convert(_strip_trailing_empty_onnx_inputs(onnx.load(str(encoder_path)))).eval().to(self.device)
        self.register_buffer("_allowed_pred_num_tokens", self.allowed_pred_num_tokens.to(self.device), persistent=False)

    @staticmethod
    def _quat_wxyz_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
        quat = quat.to(torch.float32)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        w, x, y, z = quat.unbind(dim=-1)
        return torch.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ], dim=-1)

    def _build_encoder_obs(self, planner_qpos: torch.Tensor) -> torch.Tensor:
        batch = planner_qpos.shape[0]
        encoder_obs = torch.zeros(batch, 1762, dtype=torch.float32, device=planner_qpos.device)
        encoder_obs[:, 1] = 1.0
        future_qpos = planner_qpos[:, :10]
        joint_pos = future_qpos[:, :, 7:36]
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel[:, 1:] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / 0.1
        encoder_obs[:, 4:584] = torch.cat([joint_pos, joint_vel], dim=-1).reshape(batch, -1)
        encoder_obs[:, 584:644] = self._quat_wxyz_to_rot6d(future_qpos[:, :, 3:7]).reshape(batch, -1)
        return encoder_obs

    def _planner_single(self, context: torch.Tensor, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.planner(
            context,
            commands["target_vel"].reshape(1),
            commands["mode"].to(torch.long).reshape(1),
            commands["movement_direction"].reshape(1, 3),
            commands["facing_direction"].reshape(1, 3),
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, 1, dtype=torch.long, device=self.device),
            torch.zeros(1, 4, 3, dtype=torch.float32, device=self.device),
            torch.zeros(1, 4, dtype=torch.float32, device=self.device),
            self._allowed_pred_num_tokens,
            commands["height"].reshape(1),
        )
        if isinstance(output, dict):
            return output["mujoco_qpos"]
        return output[0]

    @torch.inference_mode()
    def forward(self, context_mujoco_qpos: torch.Tensor, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        context = context_mujoco_qpos.to(self.device, dtype=torch.float32)
        batch = context.shape[0]
        planner_qpos = []
        for env_id in range(batch):
            command_i = {name: value[env_id].to(self.device) for name, value in commands.items()}
            planner_qpos.append(self._planner_single(context[env_id : env_id + 1], command_i))
        qpos = torch.cat(planner_qpos, dim=0)
        encoder_obs = self._build_encoder_obs(qpos)
        return torch.cat([self.encoder(encoder_obs[env_id : env_id + 1]) for env_id in range(batch)], dim=0)


class SonicTorchDecoder(torch.nn.Module):
    """Batched Torch implementation of the release SONIC decoder ONNX graph."""

    weight_names = [
        "onnx::MatMul_136",
        "onnx::MatMul_137",
        "onnx::MatMul_138",
        "onnx::MatMul_139",
        "onnx::MatMul_140",
        "onnx::MatMul_141",
        "onnx::MatMul_142",
    ]
    bias_names = [
        "module.decoders.g1_dyn.module.0.bias",
        "module.decoders.g1_dyn.module.2.bias",
        "module.decoders.g1_dyn.module.4.bias",
        "module.decoders.g1_dyn.module.6.bias",
        "module.decoders.g1_dyn.module.8.bias",
        "module.decoders.g1_dyn.module.10.bias",
        "module.decoders.g1_dyn.module.12.bias",
    ]

    def __init__(self, decoder_path: str | Path, device: str | torch.device):
        super().__init__()
        try:
            import onnx
            from onnx import numpy_helper
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("SonicTorchDecoder requires the onnx Python package.") from exc

        model = onnx.load(str(decoder_path))
        initializers = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.as_tensor(initializers[name]), requires_grad=False) for name in self.weight_names]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.as_tensor(initializers[name]), requires_grad=False) for name in self.bias_names]
        )
        self.to(torch.device(device))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != 994:
            raise ValueError(f"decoder obs must have shape [batch, 994], got {tuple(obs.shape)}")
        x = obs.unsqueeze(1)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, weight) + bias
            if index < len(self.weights) - 1:
                x = torch.nn.functional.silu(x)
        return x[:, 0, :29]


class SonicTorchBackend:
    """Scalable frozen SONIC decoder prior with batched Torch execution."""

    default_angles = torch.as_tensor(np.array([
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ], dtype=np.float32))
    action_scale = torch.as_tensor(np.array([
        0.35056284, 0.35056284, 0.54516023, 0.35056284, 0.43856567, 0.43856567,
        0.35056284, 0.35056284, 0.54516023, 0.35056284, 0.43856567, 0.43856567,
        0.54516023, 0.43856567, 0.43856567,
        0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.07452953, 0.07452953,
        0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.07452953, 0.07452953,
    ], dtype=np.float32))

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        decoder_path: str | Path,
    ):
        if action_dim != 29:
            raise ValueError(f"SonicTorchBackend action_dim must be 29 for release decoder, got {action_dim}")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.decoder = SonicTorchDecoder(decoder_path, self.device).eval()
        self.default_angles_t = self.default_angles.to(self.device).reshape(1, 29)
        self.action_scale_t = self.action_scale.to(self.device).reshape(1, 29)
        self._token_state = torch.zeros(self.num_envs, 64, device=self.device)
        self._his_base_ang_vel = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self._his_joint_pos = self.default_angles_t.reshape(1, 1, 29).repeat(self.num_envs, 10, 1)
        self._his_joint_vel = torch.zeros(self.num_envs, 10, 29, device=self.device)
        self._his_last_actions = torch.zeros(self.num_envs, 10, 29, device=self.device)
        self._his_gravity = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self._his_gravity[:, :, 2] = -1.0

    def _update_token_state_from_commands(self, commands: dict[str, torch.Tensor]):
        token = torch.zeros(self.num_envs, 64, device=self.device)
        token[:, 0:1] = commands["target_vel"].to(self.device, dtype=torch.float32)
        token[:, 1:4] = commands["movement_direction"].to(self.device, dtype=torch.float32)
        token[:, 4:7] = commands["facing_direction"].to(self.device, dtype=torch.float32)
        token[:, 7:8] = commands["mode"].to(self.device, dtype=torch.float32) / 2.0
        token[:, 8:9] = commands["height"].to(self.device, dtype=torch.float32)
        self._token_state = token

    def _update_decoder_history_from_low_level_observations(self, low_level_observations: torch.Tensor):
        observations = low_level_observations.to(self.device, dtype=torch.float32)
        if observations.shape[0] != self.num_envs or observations.shape[1] < 70:
            raise ValueError(
                "low_level_observations must have shape [num_envs, >=70] with "
                "base_ang_vel, projected_gravity, joint_pos, and joint_vel terms"
            )
        self._his_base_ang_vel = torch.roll(self._his_base_ang_vel, shifts=-1, dims=1)
        self._his_base_ang_vel[:, -1] = observations[:, 3:6]
        self._his_gravity = torch.roll(self._his_gravity, shifts=-1, dims=1)
        self._his_gravity[:, -1] = observations[:, 6:9]
        self._his_joint_pos = torch.roll(self._his_joint_pos, shifts=-1, dims=1)
        self._his_joint_pos[:, -1] = self.default_angles_t + observations[:, 12:41]
        self._his_joint_vel = torch.roll(self._his_joint_vel, shifts=-1, dims=1)
        self._his_joint_vel[:, -1] = observations[:, 41:70]

    def _build_decoder_obs(self) -> torch.Tensor:
        return torch.cat([
            self._token_state,
            self._his_base_ang_vel.reshape(self.num_envs, -1),
            self._his_joint_pos.reshape(self.num_envs, -1),
            self._his_joint_vel.reshape(self.num_envs, -1),
            self._his_last_actions.reshape(self.num_envs, -1),
            self._his_gravity.reshape(self.num_envs, -1),
        ], dim=1)

    @torch.inference_mode()
    def compute_joint_targets(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._update_token_state_from_commands(commands)
        uses_real_history = low_level_observations is not None
        if uses_real_history:
            self._update_decoder_history_from_low_level_observations(low_level_observations)
        decoder_obs = self._build_decoder_obs()
        action = self.decoder(decoder_obs)
        joint_target = self.default_angles_t + action * self.action_scale_t
        if not uses_real_history:
            self._his_joint_pos = torch.roll(self._his_joint_pos, shifts=-1, dims=1)
            self._his_joint_pos[:, -1] = joint_target
        self._his_last_actions = torch.roll(self._his_last_actions, shifts=-1, dims=1)
        self._his_last_actions[:, -1] = action
        return joint_target


class SonicMotionBricksBackend(SonicTorchBackend):
    """SONIC backend using MotionBricks' original PyTorch planner with the release encoder/decoder."""

    default_motionbricks_root = Path(os.environ.get("MOTIONBRICKS_ROOT", ""))
    required_checkpoint_relpaths = (
        Path("out/G1-clip.ckpt"),
        Path("out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt"),
        Path("out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt"),
        Path("out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt"),
    )
    allowed_pred_num_tokens = torch.as_tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        encoder_path: str | Path,
        decoder_path: str | Path,
        motionbricks_root: str | Path | None = None,
    ):
        super().__init__(num_envs=num_envs, action_dim=action_dim, device=device, decoder_path=decoder_path)
        try:
            import onnx
            from onnx2torch import convert
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SonicMotionBricksBackend requires onnx and onnx2torch in the SRU venv."
            ) from exc

        self.motionbricks_root = Path(motionbricks_root or self.default_motionbricks_root)
        self._validate_motionbricks_assets()
        _register_onnx2torch_one_hot_converter()
        self.encoder = convert(_strip_trailing_empty_onnx_inputs(onnx.load(str(encoder_path)))).eval().to(self.device)
        self.full_agent = self._create_full_agent()
        self._last_token_command = None
        self._token_age = torch.full((self.num_envs,), 1_000_000, dtype=torch.long, device=self.device)
        self._token_refresh_interval = 10
        self.registered_allowed_pred_num_tokens = self.allowed_pred_num_tokens.to(self.device)

    def _validate_motionbricks_assets(self):
        for relpath in self.required_checkpoint_relpaths:
            path = self.motionbricks_root / relpath
            if not path.is_file():
                raise FileNotFoundError(str(path))
            with path.open("rb") as f:
                head = f.read(64)
            if head.startswith(b"version https://git-lfs.github.com/spec/v1") or path.stat().st_size < 1024:
                raise RuntimeError(
                    f"MotionBricks checkpoint {path} is a Git LFS pointer or incomplete file. "
                    "Fetch real weights with `git lfs pull --include=\"motionbricks/out/**\" --exclude=\"\"`."
                )
        skeleton = self.motionbricks_root / "assets" / "skeletons" / "g1" / "g1.xml"
        if not skeleton.is_file():
            raise FileNotFoundError(str(skeleton))

    def _create_full_agent(self):
        motionbricks_path = str(self.motionbricks_root)
        if motionbricks_path not in sys.path:
            sys.path.insert(0, motionbricks_path)
        try:
            import torch as t
            from motionbricks.exp_setup.experiment import test as build_motionbricks_models
            from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
            from motionbricks.motion_backbone.inference.motion_inference import motion_inference
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SonicMotionBricksBackend requires MotionBricks dependencies. Install them with "
                "Install MotionBricks in the active instinct_nav runtime environment: `python -m pip install -e "
                f"{self.motionbricks_root}`."
            ) from exc

        args = Namespace(
            result_dir=str(self.motionbricks_root / "out"),
            data_root=str(self.motionbricks_root / "datasets"),
            explicit_dataset_folder=str(self.motionbricks_root / "datasets" / "motionbricks-G1"),
            clips_ckpt=str(self.motionbricks_root / "out" / "G1-clip.ckpt"),
            return_model_configs=True,
            return_dataloader=False,
            EXP="default",
        )
        models, confs = build_motionbricks_models(args)
        for model_name in ("pose", "root"):
            state_dict = t.load(confs[model_name].ckpt_path, map_location=self.device)["state_dict"]
            models[model_name].load_state_dict(state_dict)
        inferencer = motion_inference(models, models["pose"].args, device=str(self.device))
        agent = full_navigation_agent(
            inferencer,
            train_dataloader=None,
            device=str(self.device),
            skeleton_xml=str(self.motionbricks_root / "assets" / "skeletons" / "g1" / "g1.xml"),
            ckpt_path=args.clips_ckpt,
            reprocess_clips=False,
            force_canonicalization=True,
            filter_qpos=True,
            clips="G1",
        ).to(self.device)
        agent.eval()
        return agent

    def _build_planner_context(self, env_ids: torch.Tensor) -> torch.Tensor:
        context = torch.zeros(env_ids.numel(), 4, 36, dtype=torch.float32, device=self.device)
        context[:, :, 3] = 1.0
        context[:, :, 7:] = self._his_joint_pos[env_ids, -4:]
        return context

    def _build_encoder_obs(self, planner_qpos: torch.Tensor) -> torch.Tensor:
        batch = planner_qpos.shape[0]
        encoder_obs = torch.zeros(batch, 1762, dtype=torch.float32, device=planner_qpos.device)
        encoder_obs[:, 1] = 1.0
        future_qpos = planner_qpos[:, :10]
        joint_pos = future_qpos[:, :, 7:36]
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel[:, 1:] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / 0.1
        encoder_obs[:, 4:584] = torch.cat([joint_pos, joint_vel], dim=-1).reshape(batch, -1)
        encoder_obs[:, 584:644] = SonicTorchPlannerEncoder._quat_wxyz_to_rot6d(future_qpos[:, :, 3:7]).reshape(batch, -1)
        return encoder_obs

    def _build_token_command_signature(self, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            commands["target_vel"].to(self.device, dtype=torch.float32),
            commands["movement_direction"].to(self.device, dtype=torch.float32),
            commands["facing_direction"].to(self.device, dtype=torch.float32),
            commands["mode"].to(self.device, dtype=torch.float32),
            commands["height"].to(self.device, dtype=torch.float32),
        ], dim=1)

    def _token_update_mask(self, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        signature = self._build_token_command_signature(commands)
        if self._last_token_command is None:
            changed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            changed = torch.any(torch.abs(signature - self._last_token_command) > 1.0e-4, dim=1)
        mask = changed | (self._token_age >= self._token_refresh_interval)
        self._last_token_command = signature.clone()
        self._token_age += 1
        return mask

    @torch.inference_mode()
    def _run_motionbricks_planner(self, context: torch.Tensor, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = context.shape[0]
        planner_inputs = {
            "context_mujoco_qpos": context.to(self.device, dtype=torch.float32),
            "target_vel": commands["target_vel"].to(self.device, dtype=torch.float32).reshape(batch),
            "mode": commands["mode"].to(self.device, dtype=torch.long).reshape(batch),
            "movement_direction": commands["movement_direction"].to(self.device, dtype=torch.float32),
            "facing_direction": commands["facing_direction"].to(self.device, dtype=torch.float32),
            "random_seed": torch.zeros(batch, dtype=torch.long, device=self.device),
            "has_specific_target": torch.zeros(batch, 1, dtype=torch.long, device=self.device),
            "specific_target_positions": torch.zeros(batch, 4, 3, dtype=torch.float32, device=self.device),
            "specific_target_headings": torch.zeros(batch, 4, dtype=torch.float32, device=self.device),
            "allowed_pred_num_tokens": self.registered_allowed_pred_num_tokens.repeat(batch, 1),
            "height": commands["height"].to(self.device, dtype=torch.float32).reshape(batch),
        }
        qpos, _num_pred_frames = self.full_agent.generate_new_frames(
            planner_inputs,
            controller_dt=0.25,
            force_generation=True,
        )
        return qpos.to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def _update_token_state_from_commands(self, commands: dict[str, torch.Tensor]):
        mask = self._token_update_mask(commands)
        if not torch.any(mask):
            return
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()
        context = self._build_planner_context(env_ids)
        masked_commands = {name: value.to(self.device)[env_ids] for name, value in commands.items()}
        planner_qpos = self._run_motionbricks_planner(context, masked_commands)
        encoder_obs = self._build_encoder_obs(planner_qpos)
        self._token_state[env_ids] = self.encoder(encoder_obs)
        self._token_age[env_ids] = 0




class SonicTorchFullBackend(SonicTorchBackend):
    """Torch SONIC backend using release planner, encoder, and decoder modules."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        planner_path: str | Path,
        encoder_path: str | Path,
        decoder_path: str | Path,
    ):
        super().__init__(num_envs=num_envs, action_dim=action_dim, device=device, decoder_path=decoder_path)
        self.planner_encoder = SonicTorchPlannerEncoder(planner_path, encoder_path, self.device).eval()
        self._last_token_command = None
        self._token_age = torch.full((self.num_envs,), 1_000_000, dtype=torch.long, device=self.device)
        self._token_refresh_interval = 10

    def _build_planner_context(self, env_ids: torch.Tensor) -> torch.Tensor:
        context = torch.zeros(env_ids.numel(), 4, 36, dtype=torch.float32, device=self.device)
        context[:, :, 3] = 1.0
        context[:, :, 7:] = self._his_joint_pos[env_ids, -4:]
        return context

    def _build_token_command_signature(self, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            commands["target_vel"].to(self.device, dtype=torch.float32),
            commands["movement_direction"].to(self.device, dtype=torch.float32),
            commands["facing_direction"].to(self.device, dtype=torch.float32),
            commands["mode"].to(self.device, dtype=torch.float32),
            commands["height"].to(self.device, dtype=torch.float32),
        ], dim=1)

    def _token_update_mask(self, commands: dict[str, torch.Tensor]) -> torch.Tensor:
        signature = self._build_token_command_signature(commands)
        if self._last_token_command is None:
            changed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            changed = torch.any(torch.abs(signature - self._last_token_command) > 1.0e-4, dim=1)
        mask = changed | (self._token_age >= self._token_refresh_interval)
        self._last_token_command = signature.clone()
        self._token_age += 1
        return mask

    @torch.inference_mode()
    def _update_token_state_from_commands(self, commands: dict[str, torch.Tensor]):
        mask = self._token_update_mask(commands)
        if not torch.any(mask):
            return
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()
        context = self._build_planner_context(env_ids)
        masked_commands = {name: value.to(self.device)[env_ids] for name, value in commands.items()}
        self._token_state[env_ids] = self.planner_encoder(context, masked_commands)
        self._token_age[env_ids] = 0




class SonicOnnxBackend:
    """Frozen SONIC backend boundary for release ONNX artifacts."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        planner_path: str | Path,
        encoder_path: str | Path,
        decoder_path: str | Path,
    ):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.planner_path = self._require_file(planner_path)
        self.encoder_path = self._require_file(encoder_path)
        self.decoder_path = self._require_file(decoder_path)
        self._ort = self._load_onnxruntime()
        self._providers = ["CPUExecutionProvider"]
        self.planner_session = self._ort.InferenceSession(str(self.planner_path), providers=self._providers)
        self.encoder_session = self._ort.InferenceSession(str(self.encoder_path), providers=self._providers)
        self.decoder_session = self._ort.InferenceSession(str(self.decoder_path), providers=self._providers)
        self._validate_contracts()
        self._token_state = np.zeros((self.num_envs, 64), dtype=np.float32)
        self._his_base_ang_vel = np.zeros((self.num_envs, 10, 3), dtype=np.float32)
        self._his_joint_pos = np.tile(self.default_angles.reshape(1, 1, 29), (self.num_envs, 10, 1)).astype(np.float32)
        self._his_joint_vel = np.zeros((self.num_envs, 10, 29), dtype=np.float32)
        self._his_last_actions = np.zeros((self.num_envs, 10, 29), dtype=np.float32)
        self._his_gravity = np.zeros((self.num_envs, 10, 3), dtype=np.float32)
        self._his_gravity[:, :, 2] = -1.0
        self._last_token_command = None
        self._token_age = np.full((self.num_envs,), 1_000_000, dtype=np.int64)
        self._token_refresh_interval = 10

    @staticmethod
    def _require_file(path: str | Path) -> Path:
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(str(model_path))
        return model_path

    @staticmethod
    def _load_onnxruntime():
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SonicOnnxBackend requires onnxruntime. Install it into the SRU venv with "
                "Install onnxruntime in the active instinct_nav runtime environment."
            ) from exc
        return ort

    @staticmethod
    def _shape_by_name(session) -> dict[str, list[int | str]]:
        return {node.name: list(node.shape) for node in session.get_inputs()}

    @staticmethod
    def _output_shape_by_name(session) -> dict[str, list[int | str]]:
        return {node.name: list(node.shape) for node in session.get_outputs()}

    def _validate_contracts(self):
        self.planner_input_shapes = self._shape_by_name(self.planner_session)
        planner_output_shapes = self._output_shape_by_name(self.planner_session)
        required_planner_inputs = {
            "context_mujoco_qpos": [1, 4, 36],
            "target_vel": [1],
            "mode": [1],
            "movement_direction": [1, 3],
            "facing_direction": [1, 3],
            "random_seed": [1],
            "has_specific_target": [1, 1],
            "specific_target_positions": [1, 4, 3],
            "specific_target_headings": [1, 4],
            "allowed_pred_num_tokens": [1, 11],
            "height": [1],
        }
        for name, expected_shape in required_planner_inputs.items():
            actual_shape = self.planner_input_shapes.get(name)
            if actual_shape != expected_shape:
                raise ValueError(f"planner input {name!r} shape {actual_shape} != {expected_shape}")
        if planner_output_shapes.get("mujoco_qpos") != [1, 64, 36]:
            raise ValueError(f"planner output 'mujoco_qpos' shape {planner_output_shapes.get('mujoco_qpos')} != [1, 64, 36]")
        if planner_output_shapes.get("num_pred_frames") != [1]:
            raise ValueError(f"planner output 'num_pred_frames' shape {planner_output_shapes.get('num_pred_frames')} != [1]")

        encoder_inputs = self.encoder_session.get_inputs()
        encoder_outputs = self.encoder_session.get_outputs()
        if len(encoder_inputs) != 1 or encoder_inputs[0].name != "obs_dict":
            raise ValueError("encoder must have exactly one input named 'obs_dict'")
        if len(encoder_outputs) != 1 or encoder_outputs[0].name != "encoded_tokens":
            raise ValueError("encoder must have exactly one output named 'encoded_tokens'")
        self.encoder_input_shape = list(encoder_inputs[0].shape)
        self.encoder_output_shape = list(encoder_outputs[0].shape)
        if self.encoder_input_shape != [1, 1762]:
            raise ValueError(f"encoder input shape {self.encoder_input_shape} != [1, 1762]")
        if self.encoder_output_shape != [1, 64]:
            raise ValueError(f"encoder output shape {self.encoder_output_shape} != [1, 64]")

        decoder_inputs = self.decoder_session.get_inputs()
        decoder_outputs = self.decoder_session.get_outputs()
        if len(decoder_inputs) != 1 or decoder_inputs[0].name != "obs_dict":
            raise ValueError("decoder must have exactly one input named 'obs_dict'")
        if len(decoder_outputs) != 1 or decoder_outputs[0].name != "action":
            raise ValueError("decoder must have exactly one output named 'action'")
        self.decoder_input_shape = list(decoder_inputs[0].shape)
        self.decoder_output_shape = list(decoder_outputs[0].shape)
        if self.decoder_input_shape != [1, 994]:
            raise ValueError(f"decoder input shape {self.decoder_input_shape} != [1, 994]")
        if self.decoder_output_shape != [1, 29]:
            raise ValueError(f"decoder output shape {self.decoder_output_shape} != [1, 29]")
        if self.action_dim != 29:
            raise ValueError(f"SonicOnnxBackend action_dim must be 29 for release decoder, got {self.action_dim}")

    default_angles = np.array([
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    # Release policy_parameters.hpp action_scale values evaluated from the published motor constants.
    action_scale = np.array([
        0.35056284, 0.35056284, 0.54516023, 0.35056284, 0.43856567, 0.43856567,
        0.35056284, 0.35056284, 0.54516023, 0.35056284, 0.43856567, 0.43856567,
        0.54516023, 0.43856567, 0.43856567,
        0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.07452953, 0.07452953,
        0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.43856567, 0.07452953, 0.07452953,
    ], dtype=np.float32)

    allowed_pred_num_tokens = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)

    @staticmethod
    def _to_numpy(commands: dict[str, torch.Tensor], name: str) -> np.ndarray:
        return commands[name].detach().cpu().numpy()

    @staticmethod
    def _quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1.0e-6)
        w, x, y, z = np.moveaxis(quat / norm, -1, 0)
        return np.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ], axis=-1).astype(np.float32)

    def _build_planner_context(self, env_id: int) -> np.ndarray:
        context = np.zeros((1, 4, 36), dtype=np.float32)
        context[:, :, 3] = 1.0
        context[0, :, 7:] = self._his_joint_pos[env_id, -4:]
        return context

    def _run_planner(self, env_id: int, commands_np: dict[str, np.ndarray]) -> np.ndarray:
        planner_inputs = {
            "context_mujoco_qpos": self._build_planner_context(env_id),
            "target_vel": commands_np["target_vel"][env_id].astype(np.float32).reshape(1),
            "mode": commands_np["mode"][env_id].astype(np.int64).reshape(1),
            "movement_direction": commands_np["movement_direction"][env_id].astype(np.float32).reshape(1, 3),
            "facing_direction": commands_np["facing_direction"][env_id].astype(np.float32).reshape(1, 3),
            "random_seed": np.zeros((1,), dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": self.allowed_pred_num_tokens,
            "height": commands_np["height"][env_id].astype(np.float32).reshape(1),
        }
        return self.planner_session.run(["mujoco_qpos"], planner_inputs)[0].astype(np.float32).reshape(64, 36)

    def _build_encoder_obs(self, planner_qpos: np.ndarray) -> np.ndarray:
        encoder_obs = np.zeros((1, 1762), dtype=np.float32)
        encoder_obs[0, 0] = 0.0
        encoder_obs[0, 1] = 1.0

        future_qpos = planner_qpos[:10]
        joint_pos = future_qpos[:, 7:36]
        joint_vel = np.zeros_like(joint_pos)
        joint_vel[1:] = np.diff(joint_pos, axis=0) / 0.1
        encoder_obs[0, 4:584] = np.concatenate([joint_pos, joint_vel], axis=-1).reshape(-1)
        encoder_obs[0, 584:644] = self._quat_wxyz_to_rot6d(future_qpos[:, 3:7]).reshape(-1)
        return encoder_obs

    def _update_token_state(self, env_id: int, commands_np: dict[str, np.ndarray]):
        planner_qpos = self._run_planner(env_id, commands_np)
        encoder_obs = self._build_encoder_obs(planner_qpos)
        self._token_state[env_id] = self.encoder_session.run(
            ["encoded_tokens"], {"obs_dict": encoder_obs}
        )[0].astype(np.float32).reshape(64)

    def _build_token_command_signature(self, commands_np: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            commands_np["target_vel"].astype(np.float32),
            commands_np["movement_direction"].astype(np.float32),
            commands_np["facing_direction"].astype(np.float32),
            commands_np["mode"].astype(np.float32),
            commands_np["height"].astype(np.float32),
        ], axis=1)

    def _token_update_mask(self, commands_np: dict[str, np.ndarray]) -> np.ndarray:
        signature = self._build_token_command_signature(commands_np)
        last_signature = getattr(self, "_last_token_command", None)
        token_age = getattr(self, "_token_age", np.full((self.num_envs,), 1_000_000, dtype=np.int64))
        if last_signature is None:
            changed = np.ones((self.num_envs,), dtype=bool)
        else:
            changed = np.any(np.abs(signature - last_signature) > 1.0e-4, axis=1)
        refresh_interval = getattr(self, "_token_refresh_interval", 10)
        mask = changed | (token_age >= refresh_interval)
        self._last_token_command = signature.copy()
        self._token_age = token_age + 1
        return mask

    def _update_decoder_history_from_low_level_observations(self, low_level_observations: torch.Tensor):
        observations = low_level_observations.detach().cpu().numpy().astype(np.float32)
        if observations.shape[0] != self.num_envs or observations.shape[1] < 70:
            raise ValueError(
                "low_level_observations must have shape [num_envs, >=70] with "
                "base_ang_vel, projected_gravity, joint_pos, and joint_vel terms"
            )

        self._his_base_ang_vel = np.roll(self._his_base_ang_vel, shift=-1, axis=1)
        self._his_base_ang_vel[:, -1] = observations[:, 3:6]
        self._his_gravity = np.roll(self._his_gravity, shift=-1, axis=1)
        self._his_gravity[:, -1] = observations[:, 6:9]
        self._his_joint_pos = np.roll(self._his_joint_pos, shift=-1, axis=1)
        self._his_joint_pos[:, -1] = self.default_angles.reshape(1, 29) + observations[:, 12:41]
        self._his_joint_vel = np.roll(self._his_joint_vel, shift=-1, axis=1)
        self._his_joint_vel[:, -1] = observations[:, 41:70]

    def _build_decoder_obs(self, env_id: int) -> np.ndarray:
        return np.concatenate([
            self._token_state[env_id].reshape(-1),
            self._his_base_ang_vel[env_id].reshape(-1),
            self._his_joint_pos[env_id].reshape(-1),
            self._his_joint_vel[env_id].reshape(-1),
            self._his_last_actions[env_id].reshape(-1),
            self._his_gravity[env_id].reshape(-1),
        ]).astype(np.float32).reshape(1, 994)

    def compute_joint_targets(
        self,
        commands: dict[str, torch.Tensor],
        low_level_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        commands_np = {name: self._to_numpy(commands, name) for name in (
            "target_vel",
            "movement_direction",
            "facing_direction",
            "mode",
            "height",
        )}
        uses_real_history = low_level_observations is not None
        if uses_real_history:
            self._update_decoder_history_from_low_level_observations(low_level_observations)
        targets = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        token_update_mask = self._token_update_mask(commands_np)
        for env_id in range(self.num_envs):
            if token_update_mask[env_id]:
                self._update_token_state(env_id, commands_np)
                self._token_age[env_id] = 0
            decoder_obs = self._build_decoder_obs(env_id)
            action = self.decoder_session.run(["action"], {"obs_dict": decoder_obs})[0].astype(np.float32).reshape(29)
            joint_target = self.default_angles + action * self.action_scale
            targets[env_id] = joint_target
            if not uses_real_history:
                self._his_joint_pos[env_id] = np.roll(self._his_joint_pos[env_id], shift=-1, axis=0)
                self._his_joint_pos[env_id, -1] = joint_target
            self._his_last_actions[env_id] = np.roll(self._his_last_actions[env_id], shift=-1, axis=0)
            self._his_last_actions[env_id, -1] = action
        return torch.as_tensor(targets, device=self.device)
