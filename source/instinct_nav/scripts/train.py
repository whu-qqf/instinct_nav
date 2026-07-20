#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Train a navigation policy using RSL-RL (PPO/MDPO algorithms).

Usage:
    python scripts/train.py --task <task_name> --num_envs <num> [options]

Arguments:
    --task               Task name (required)
    --num_envs           Number of parallel environments
    --seed               Random seed
    --max_iterations     Training iterations
    --run_name           Custom run name for logging
    --video              Enable video recording
    --video_length       Video length in steps (default: 200)
    --video_interval     Recording interval in steps (default: 2000)

Examples:
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --num_envs 2048
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --video --seed 42

Logs saved to: logs/rsl_rl/<experiment_name>/<timestamp>/
"""

from __future__ import annotations

import argparse
import sys

# Add the parent directory to the path so we can import from the extension
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train a navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the wandb run (appended to log directory).")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import os
from pathlib import Path
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import instinct_nav  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    import pickle

    def dump_pickle(filename: str, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def _to_legacy_obs(self, obs):
        policy_obs = obs["policy"]
        critic_obs = obs["critic"] if "critic" in obs.keys() else policy_obs
        return policy_obs, {"observations": {"critic": critic_obs}}

    def reset(self):
        obs, extras = super().reset()
        policy_obs, legacy_extras = self._to_legacy_obs(obs)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, extras

    def get_observations(self):
        return self._to_legacy_obs(super().get_observations())

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        policy_obs, legacy_extras = self._to_legacy_obs(obs)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, rew, dones, extras


def _register_sru_rsl_rl_extensions():
    """Expose SRU custom policy modules to rsl-rl-lib 3.x without shadowing the whole package."""
    import rsl_rl.modules as rsl_modules
    import rsl_rl.networks as rsl_networks
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    legacy_root = os.environ.get("SRU_LEGACY_RSL_RL_ROOT", str(Path(__file__).resolve().parents[2] / "rsl_rl" / "rsl_rl"))
    legacy_modules = os.path.join(legacy_root, "modules")
    legacy_networks = os.path.join(legacy_root, "networks")
    if os.path.isdir(legacy_modules) and legacy_modules not in rsl_modules.__path__:
        rsl_modules.__path__.append(legacy_modules)
    if os.path.isdir(legacy_networks) and legacy_networks not in rsl_networks.__path__:
        rsl_networks.__path__.append(legacy_networks)

    from rsl_rl.modules.actor_critic_sru import ActorCriticSRU as LegacyActorCriticSRU

    class ActorCriticSRU(LegacyActorCriticSRU):
        def __init__(self, obs, obs_groups, num_actions, **kwargs):
            self.obs_groups = obs_groups
            num_actor_obs = sum(obs[group].reshape(obs[group].shape[0], -1).shape[-1] for group in obs_groups["policy"])
            num_critic_obs = sum(obs[group].reshape(obs[group].shape[0], -1).shape[-1] for group in obs_groups["critic"])
            kwargs.pop("actor_obs_normalization", None)
            kwargs.pop("critic_obs_normalization", None)
            from torch.distributions import Distribution, Normal

            if not callable(Normal.set_default_validate_args):
                Normal.set_default_validate_args = Distribution.set_default_validate_args
            super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        def _flatten_obs_groups(self, obs, group_name):
            if not torch.is_tensor(obs):
                return torch.cat(
                    [obs[group].reshape(obs[group].shape[0], -1) for group in self.obs_groups[group_name]], dim=-1
                )
            return obs

        def _prepare_recurrent_batch_obs(self, obs, masks):
            if masks is not None and torch.is_tensor(obs) and obs.dim() == 2:
                return obs.unsqueeze(0)
            return obs

        def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
            obs = self._flatten_obs_groups(observations, "policy")
            return super().act(self._prepare_recurrent_batch_obs(obs, masks), masks, hidden_states, dropout_masks)

        def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
            obs = self._flatten_obs_groups(observations, "policy")
            return super().act_inference(self._prepare_recurrent_batch_obs(obs, masks), masks, hidden_states, dropout_masks)

        def evaluate(self, observations, masks=None, hidden_states=None, dropout_masks=None):
            obs = self._flatten_obs_groups(observations, "critic")
            return super().evaluate(self._prepare_recurrent_batch_obs(obs, masks), masks, hidden_states, dropout_masks)

        def update_normalization(self, obs):
            pass

    on_policy_runner.ActorCriticSRU = ActorCriticSRU


import rsl_rl as _rsl_rl_pkg
if "site-packages" in (_rsl_rl_pkg.__file__ or ""):
    _register_sru_rsl_rl_extensions()

# Set torch backends for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train navigation policy with RSL-RL."""
    # Load the configurations from the registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap the environment
    env = LegacyRslRlVecEnvWrapper(env)

    # Specify log directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Specify run directory based on timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Write git state to log
    runner.add_git_repo_to_log(__file__)
    # Save configuration
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
