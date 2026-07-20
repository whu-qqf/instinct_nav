#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy (PPO/MDPO) with automatic checkpoint loading.

Usage:
    python scripts/play.py --task <task_name> [options]

Arguments:
    --task                   Task name (required, typically *-Play-v0 variant)
    --checkpoint             Path to model checkpoint (.pt file)
    --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
    --num_envs              Number of parallel environments
    --video                 Enable video recording
    --video_length          Video length in steps (default: 200)

Examples:
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --checkpoint path/to/model.pt
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --video --num_envs 16

Note: Automatically finds latest checkpoint if --checkpoint not specified.
"""

from __future__ import annotations

import argparse
import sys
import json

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--bev_follow", action="store_true", default=False, help="Track env 0 robot from a top-down BEV viewer camera.")
parser.add_argument("--bev_height_m", type=float, default=22.0, help="Height of the BEV follow camera above the robot.")
parser.add_argument("--min_initial_distance", type=float, default=0.0, help="Require env 0 initial goal distance to be at least this many meters.")
parser.add_argument("--require_terrain_name", type=str, default=None, help="Require env 0 to start on this terrain type name, e.g. stairs.")
parser.add_argument("--max_resample_attempts", type=int, default=200, help="Maximum resets used to find the requested initial distance/terrain case.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum play steps before exiting. 0 means run until the app closes.")
parser.add_argument("--stop_after_first_done", action="store_true", default=False, help="Exit when env 0 reaches any terminal condition.")
parser.add_argument("--play_summary_out", type=str, default=None, help="Optional JSON path for env 0 play summary.")
parser.add_argument("--gif_dir", type=str, default=None, help="Optional directory to save play GIF clips.")
parser.add_argument("--gif_count", type=int, default=0, help="Number of GIF clips to save. 0 disables GIF export.")
parser.add_argument("--gif_length", type=int, default=300, help="Maximum steps per GIF clip.")
parser.add_argument("--gif_frame_ms", type=int, default=80, help="Frame duration in milliseconds for GIF export.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras
args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import numpy as np
import os
import re
import torch

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import instinct_nav  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx



class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def _ensure_batched_obs(self, value):
        if torch.is_tensor(value) and value.dim() == 1:
            return value.unsqueeze(0)
        return value

    def _to_legacy_obs(self, obs, extras=None):
        obs_keys = obs.keys() if hasattr(obs, "keys") else ()
        if "policy" in obs_keys:
            policy_obs = self._ensure_batched_obs(obs["policy"])
            critic_obs = obs["critic"] if "critic" in obs_keys else policy_obs
            critic_obs = self._ensure_batched_obs(critic_obs)
            return policy_obs, {"observations": {"critic": critic_obs}}
        observation_extras = (extras or {}).get("observations", {})
        critic_obs = observation_extras.get("critic", obs) if isinstance(observation_extras, dict) else obs
        obs = self._ensure_batched_obs(obs)
        critic_obs = self._ensure_batched_obs(critic_obs)
        return obs, {"observations": {"critic": critic_obs}}

    def reset(self):
        obs, extras = super().reset()
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, extras

    def get_observations(self):
        result = super().get_observations()
        if isinstance(result, tuple) and len(result) == 2:
            obs, extras = result
        else:
            obs, extras = result, {}
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, extras

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, rew, dones, extras

def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory.

    Args:
        log_path: Base log directory path
        checkpoint_pattern: Regex pattern for checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    # Find all run directories
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    # Sort to get latest run
    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    # Find checkpoint files
    checkpoint_files = []
    for f in os.listdir(run_path):
        if re.match(checkpoint_pattern, f):
            checkpoint_files.append(f)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    # Sort to get latest checkpoint
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    latest_checkpoint = checkpoint_files[-1]

    return os.path.join(run_path, latest_checkpoint)


def compute_bucket_sizes(num_cols: int, proportions: list[float]) -> list[int]:
    raw = [num_cols * proportion for proportion in proportions]
    base = [int(value) for value in raw]
    remaining = num_cols - sum(base)
    remainders = sorted(
        enumerate([value - int(value) for value in raw]),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, _ in remainders[:remaining]:
        base[index] += 1
    return base


def build_terrain_type_lookup(env) -> dict[int, str]:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_cfg = getattr(terrain.cfg, "terrain_generator", None) if terrain is not None else None
    if terrain is None or terrain_cfg is None or not hasattr(terrain, "terrain_types"):
        return {}

    sub_terrains = getattr(terrain_cfg, "sub_terrains", None)
    if not sub_terrains:
        return {}

    terrain_names = list(sub_terrains.keys())
    proportions = [sub_terrains[name].proportion for name in terrain_names]
    bucket_sizes = compute_bucket_sizes(int(terrain_cfg.num_cols), proportions)
    lookup = {}
    current = 0
    for name, bucket_size in zip(terrain_names, bucket_sizes):
        for terrain_type in range(current, current + bucket_size):
            lookup[terrain_type] = name
        current += bucket_size
    return lookup


def env0_terrain_name(env, lookup: dict[int, str]) -> tuple[str | None, int | None, int | None]:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return None, None, None
    terrain_type = int(terrain.terrain_types[0].detach().cpu().item())
    terrain_level = int(terrain.terrain_levels[0].detach().cpu().item()) if hasattr(terrain, "terrain_levels") else None
    return lookup.get(terrain_type, f"terrain_type_{terrain_type}"), terrain_type, terrain_level


def _render_rgb_frame(env):
    frame = None
    render_fn = getattr(env.unwrapped, "render", None)
    if callable(render_fn):
        frame = render_fn()
    if frame is None and hasattr(env, "render"):
        frame = env.render()
    if frame is None:
        return None
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    return frame


def _save_gif_frames(frames, output_path: str, frame_duration_ms: int):
    if not frames:
        print(f"[WARN] No frames captured for GIF: {output_path}", flush=True)
        return False
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pil_frames = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            if np.nanmax(frame) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(frame_duration_ms),
        loop=0,
        optimize=False,
    )
    print(f"[INFO] Saved GIF: {output_path}", flush=True)
    return True


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues.

    Args:
        runner: RSL-RL runner instance
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint to CPU first for compatibility
    loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model state - handle both standard algorithms (PPO) and MDPO
    if runner.is_mdpo:
        # MDPO uses two actor-critics, load same state into both
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        # Standard algorithms use one actor-critic
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    # Load normalizers if using empirical normalization
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    # Load optimizer if requested
    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Export using the module's export_jit method
    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print(f"[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Check if the module has export_onnx method
    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    # Export using the module's export_onnx method
    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print(f"[INFO] ONNX export complete!")


def main():
    """Play navigation policy with RSL-RL."""
    # Parse command-line arguments
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    # Instantiate the configs
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.bev_follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.0, 0.0, float(args_cli.bev_height_m))
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.commands.robot_goal.debug_vis = False

    if args_cli.require_terrain_name:
        required_terrain_name = args_cli.require_terrain_name.lower()
        terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        sub_terrains = getattr(terrain_generator, "sub_terrains", None) if terrain_generator is not None else None
        if not sub_terrains:
            raise RuntimeError("--require_terrain_name was set, but this task has no terrain_generator.sub_terrains.")
        matched_name = None
        for name in sub_terrains.keys():
            if name.lower() == required_terrain_name:
                matched_name = name
                break
        if matched_name is None:
            available = ", ".join(sub_terrains.keys())
            raise RuntimeError(f"Unknown terrain name {args_cli.require_terrain_name!r}. Available terrain names: {available}")
        for name, terrain_cfg in sub_terrains.items():
            terrain_cfg.proportion = 1.0 if name == matched_name else 0.0
        print(f"[INFO] Restricting play terrain generation to: {matched_name}", flush=True)

    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        # Get last checkpoint from log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video or args_cli.gif_count > 0) else None)
    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(os.path.abspath(resume_path)), "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording play video to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # Wrap the environment
    env = LegacyRslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint with compatibility handling
    load_checkpoint_with_fallback(runner, resume_path)

    # Export JIT if requested
    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    # Export ONNX if requested
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    # Obtain policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = env.get_observations()
    goal_term = env.unwrapped.command_manager._terms.get("robot_goal")
    terrain_lookup = build_terrain_type_lookup(env)
    required_terrain_name = args_cli.require_terrain_name.lower() if args_cli.require_terrain_name else None
    selected_initial_distance = None
    done_summary = None

    def case_matches() -> tuple[bool, float | None, str | None, int | None, int | None]:
        initial_distance = None
        distance_ok = True
        if goal_term is not None:
            initial_distance = float(goal_term.initial_distance_to_goal[0].detach().cpu())
            distance_ok = initial_distance >= float(args_cli.min_initial_distance)
        terrain_name, terrain_type, terrain_level = env0_terrain_name(env, terrain_lookup)
        terrain_ok = True
        if required_terrain_name is not None:
            terrain_ok = terrain_name is not None and terrain_name.lower() == required_terrain_name
        return distance_ok and terrain_ok, initial_distance, terrain_name, terrain_type, terrain_level

    needs_resample = args_cli.min_initial_distance > 0.0 or required_terrain_name is not None

    def select_navigation_case():
        nonlocal obs, selected_initial_distance
        if needs_resample:
            for attempt in range(max(args_cli.max_resample_attempts, 1)):
                matched, initial_distance, terrain_name, terrain_type, terrain_level = case_matches()
                if matched:
                    selected_initial_distance = initial_distance
                    print(
                        "[INFO] Selected env 0 navigation case: "
                        f"terrain={terrain_name} type={terrain_type} level={terrain_level} "
                        f"initial_distance={initial_distance:.2f} m after {attempt} reset(s)",
                        flush=True,
                    )
                    return terrain_name, terrain_type, terrain_level, initial_distance
                obs, _ = env.reset()
            _, initial_distance, terrain_name, terrain_type, terrain_level = case_matches()
            raise RuntimeError(
                "Could not sample env 0 navigation case "
                f"terrain={args_cli.require_terrain_name!r}, min_initial_distance={args_cli.min_initial_distance:.2f} m "
                f"after {args_cli.max_resample_attempts} resets. "
                f"Last case: terrain={terrain_name} type={terrain_type} level={terrain_level} "
                f"initial_distance={initial_distance:.2f} m"
            )
        if goal_term is not None:
            initial_distance = float(goal_term.initial_distance_to_goal[0].detach().cpu())
            terrain_name, terrain_type, terrain_level = env0_terrain_name(env, terrain_lookup)
            selected_initial_distance = initial_distance
            print(
                f"[INFO] Env 0 terrain={terrain_name} type={terrain_type} level={terrain_level} "
                f"initial_distance={initial_distance:.2f} m",
                flush=True,
            )
            return terrain_name, terrain_type, terrain_level, initial_distance
        return None, None, None, None

    current_terrain_name, current_terrain_type, current_terrain_level, _ = select_navigation_case()
    timestep = 0
    gif_saved_count = 0
    gif_frames = []
    gif_dir = os.path.abspath(os.path.expanduser(args_cli.gif_dir)) if args_cli.gif_dir else None

    # Simulate environment
    while simulation_app.is_running():
        # Run policy
        with torch.inference_mode():
            actions = policy(obs)
        pre_step_stats = None
        if goal_term is not None:
            pre_step_stats = {
                "initial_distance_m": float(goal_term.initial_distance_to_goal[0].detach().cpu()),
                "distance_to_goal_m": float(goal_term.distance_to_goal[0].detach().cpu()),
                "traveled_m": float(goal_term.total_distance_traveled[0].detach().cpu()),
            }
        # Step environment
        obs, _, dones, extras = env.step(actions)
        timestep += 1
        if args_cli.gif_count > 0:
            frame = _render_rgb_frame(env)
            if frame is not None:
                gif_frames.append(frame)
        if goal_term is not None and timestep % 25 == 0:
            distance = float(goal_term.distance_to_goal[0].detach().cpu())
            traveled = float(goal_term.total_distance_traveled[0].detach().cpu())
            print(f"[INFO] step={timestep} env0 distance_to_goal={distance:.2f} m traveled={traveled:.2f} m", flush=True)
        gif_clip_done = args_cli.gif_count > 0 and (bool(dones[0].item()) or timestep >= int(args_cli.gif_length))
        if gif_clip_done:
            clip_stem = args_cli.require_terrain_name or current_terrain_name or "play"
            output_dir = gif_dir or os.path.join(os.path.dirname(os.path.abspath(resume_path)), "videos", "play_gifs")
            output_path = os.path.join(output_dir, f"{clip_stem}_{gif_saved_count:03d}.gif")
            if _save_gif_frames(gif_frames, output_path, args_cli.gif_frame_ms):
                gif_saved_count += 1
            gif_frames = []
            if gif_saved_count >= int(args_cli.gif_count):
                break
            obs, _ = env.reset()
            current_terrain_name, current_terrain_type, current_terrain_level, _ = select_navigation_case()
            timestep = 0
            continue

        if args_cli.stop_after_first_done and bool(dones[0].item()):
            termination_terms = {}
            if "log" in extras and isinstance(extras["log"], dict):
                termination_terms = {
                    key: value
                    for key, value in extras["log"].items()
                    if key.startswith("Episode_Termination/")
                }
            if goal_term is not None:
                initial_distance = (
                    pre_step_stats["initial_distance_m"]
                    if pre_step_stats is not None
                    else float(goal_term.initial_distance_to_goal[0].detach().cpu())
                )
                final_distance = (
                    pre_step_stats["distance_to_goal_m"]
                    if pre_step_stats is not None
                    else float(goal_term.distance_to_goal[0].detach().cpu())
                )
                traveled = (
                    pre_step_stats["traveled_m"]
                    if pre_step_stats is not None
                    else float(goal_term.total_distance_traveled[0].detach().cpu())
                )
                done_summary = {
                    "steps": timestep,
                    "initial_distance_m": initial_distance,
                    "selected_initial_distance_m": selected_initial_distance,
                    "final_distance_m": final_distance,
                    "traveled_m": traveled,
                    "termination_terms": {key: str(value) for key, value in termination_terms.items()},
                    "done": True,
                }
                print(
                    "[INFO] Env 0 done: "
                    f"initial_distance={initial_distance:.2f} m final_distance={final_distance:.2f} m "
                    f"traveled={traveled:.2f} m termination_terms={termination_terms}",
                    flush=True,
                )
            break
        if args_cli.video and args_cli.gif_count <= 0:
            if timestep >= args_cli.video_length:
                break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached --max_steps={args_cli.max_steps}; exiting play loop.", flush=True)
            if goal_term is not None:
                done_summary = {
                    "steps": timestep,
                    "initial_distance_m": float(goal_term.initial_distance_to_goal[0].detach().cpu()),
                    "selected_initial_distance_m": selected_initial_distance,
                    "final_distance_m": float(goal_term.distance_to_goal[0].detach().cpu()),
                    "traveled_m": float(goal_term.total_distance_traveled[0].detach().cpu()),
                    "done": False,
                    "reason": "max_steps",
                }
            break

    if args_cli.play_summary_out is not None:
        summary = done_summary or {"steps": timestep, "selected_initial_distance_m": selected_initial_distance}
        summary.update(
            {
                "task": args_cli.task,
                "checkpoint": os.path.abspath(os.path.expanduser(resume_path)),
                "bev_follow": bool(args_cli.bev_follow),
                "bev_height_m": float(args_cli.bev_height_m),
                "min_initial_distance_m": float(args_cli.min_initial_distance),
            }
        )
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.play_summary_out)), exist_ok=True)
        with open(args_cli.play_summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Wrote play summary to: {args_cli.play_summary_out}", flush=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
