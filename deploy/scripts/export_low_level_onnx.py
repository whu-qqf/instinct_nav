#!/usr/bin/env python3
"""Export the frozen InstinctLab G1 parkour low-level policy to ONNX.

This script keeps only the export path from InstinctLab's parkour play script.
It avoids importing GUI-only keyboard modules so it can run in headless Kit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _append_if_exists(path: Path) -> None:
    if path.exists():
        sys.path.append(str(path))


WORKSPACE = Path("/workspace")
INSTINCTLAB_ROOT = WORKSPACE / "InstinctLab"
_append_if_exists(INSTINCTLAB_ROOT / "scripts" / "instinct_rl")

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description="Export InstinctLab parkour low-level policy ONNX files.")
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.load_run = os.path.abspath(args_cli.checkpoint_path)
args_cli.checkpoint = os.path.basename(args_cli.checkpoint_path)
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from instinct_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    resume_path = os.path.abspath(args_cli.checkpoint_path)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    export_model_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)

    obs, _ = env.get_observations()
    print(f"[INFO] Exporting ONNX files to: {export_model_dir}")
    runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    expected = ["actor.onnx", "0-depth_encoder.onnx"]
    for name in expected:
        path = os.path.join(export_model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected exported file missing: {path}")
        print(f"[INFO] Exported: {path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
