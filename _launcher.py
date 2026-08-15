#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_SOURCE_ROOT = WORKSPACE_ROOT / "source" / "instinct_nav"
_RUNTIME_PATH = PROJECT_SOURCE_ROOT / "instinct_nav" / "runtime.py"
_RUNTIME_SPEC = importlib.util.spec_from_file_location("instinct_nav_runtime", _RUNTIME_PATH)
if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
    raise RuntimeError(f"Could not load runtime module from {_RUNTIME_PATH}")
_RUNTIME_MODULE = importlib.util.module_from_spec(_RUNTIME_SPEC)
_RUNTIME_SPEC.loader.exec_module(_RUNTIME_MODULE)
build_runtime_env = _RUNTIME_MODULE.build_runtime_env
python_executable = _RUNTIME_MODULE.python_executable


def run_workspace_script(script_rel_path: str, forwarded_args: list[str]) -> int:
    env = build_runtime_env(WORKSPACE_ROOT)
    script_path = WORKSPACE_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Workspace script not found: {script_path}")

    cmd = [python_executable(env), str(script_path), *forwarded_args]
    completed = subprocess.run(cmd, cwd=WORKSPACE_ROOT, env=env)
    return completed.returncode


SOURCE_ROOT = WORKSPACE_ROOT / "source"
EXTERNAL_ROOT = WORKSPACE_ROOT / "external"
def _project_path(env_name: str, project_default: Path) -> Path:
    if env_name in os.environ:
        return Path(os.environ[env_name]).expanduser()
    return project_default


SENIOR_NAV_TASK_ROOT = _project_path(
    "INSTINCT_NAV_SENIOR_NAV_TASK_ROOT",
    SOURCE_ROOT / "instinct_nav",
)
SENIOR_RSL_RL_ROOT = _project_path(
    "INSTINCT_NAV_SENIOR_RSL_RL_ROOT",
    SOURCE_ROOT / "rsl_rl",
)
SENIOR_INSTINCTLAB_SOURCE = _project_path(
    "INSTINCT_NAV_SENIOR_INSTINCTLAB_SOURCE",
    EXTERNAL_ROOT / "instinctlab",
)
SENIOR_INSTINCT_RL_ROOT = _project_path(
    "INSTINCT_NAV_SENIOR_INSTINCT_RL_ROOT",
    EXTERNAL_ROOT / "instinct_rl",
)
SENIOR_G1_CHECKPOINT = _project_path(
    "SRU_G1_INSTINCT_CHECKPOINT",
    WORKSPACE_ROOT / "artifacts" / "low_level" / "g1_parkour" / "model_30000.pt",
)


def run_senior_nav_script(script_name: str, forwarded_args: list[str]) -> int:
    env = build_runtime_env(WORKSPACE_ROOT)
    script_path = SENIOR_NAV_TASK_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"instinct_nav script not found: {script_path}")

    env.setdefault("SRU_LEGACY_RSL_RL_ROOT", str(SENIOR_RSL_RL_ROOT / "rsl_rl"))
    env.setdefault("SRU_INSTINCT_RL_ROOT", str(SENIOR_INSTINCT_RL_ROOT))
    env.setdefault("SRU_G1_INSTINCT_CHECKPOINT", str(SENIOR_G1_CHECKPOINT))
    senior_pythonpath = [
        str(SENIOR_RSL_RL_ROOT),
        str(SENIOR_NAV_TASK_ROOT),
        str(SENIOR_INSTINCTLAB_SOURCE),
        str(SENIOR_INSTINCT_RL_ROOT),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        senior_pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(senior_pythonpath)

    cmd = [python_executable(env), str(script_path), *forwarded_args]
    completed = subprocess.run(cmd, cwd=SENIOR_NAV_TASK_ROOT, env=env)
    return completed.returncode

def doctor_main(_: list[str]) -> int:
    env = build_runtime_env(WORKSPACE_ROOT)
    print("[INFO] instinct_nav runtime is ready")
    print(f"[INFO] runtime venv: {env['INSTINCT_NAV_RUNTIME_VENV']}")
    print(f"[INFO] isaaclab root: {env['INSTINCT_NAV_ISAACLAB_ROOT']}")
    print(f"[INFO] instinctlab root: {env['INSTINCT_NAV_INSTINCTLAB_ROOT']}")
    print(f"[INFO] isaaclab_rl root: {env['INSTINCT_NAV_ISAACLAB_RL_ROOT']}")
    return 0


def visualize_terrain_main(argv: list[str]) -> int:
    """Launch the RichTerrain top-down preview inside the Isaac Sim runtime."""
    return run_workspace_script("scripts/visualize_rich_terrain.py", argv)


def inspect_terrain_main(argv: list[str]) -> int:
    """Open RichTerrain in an interactive Isaac Sim viewport."""
    return run_workspace_script("scripts/inspect_rich_terrain.py", argv)


def train_senior_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Launch instinct_nav SRU G1 training path.")
    parser.add_argument("--task", type=str, default="Instinct-Nav-G1-SRU-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--no_load_optimizer", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args, passthrough = parser.parse_known_args(argv)

    forwarded = [
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
    ]
    if args.run_name:
        forwarded.extend(["--run_name", args.run_name])
    if args.wandb:
        forwarded.append("--wandb")
    if args.wandb_entity:
        forwarded.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_project:
        forwarded.extend(["--wandb_project", args.wandb_project])
    if args.resume_checkpoint:
        forwarded.extend(["--resume_checkpoint", args.resume_checkpoint])
    if args.no_load_optimizer:
        forwarded.append("--no_load_optimizer")
    if args.headless:
        forwarded.append("--headless")
    forwarded.extend(passthrough)
    return run_workspace_script("scripts/train_senior.py", forwarded)


def play_senior_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Launch instinct_nav SRU G1 checkpoint play.")
    parser.add_argument("--task", type=str, default="Instinct-Nav-G1-SRU-Play-v0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=600)
    parser.add_argument("--bev_follow", action="store_true", default=False)
    parser.add_argument("--bev_height_m", type=float, default=22.0)
    args, passthrough = parser.parse_known_args(argv)

    forwarded = ["--task", args.task, "--num_envs", str(args.num_envs)]
    if args.checkpoint:
        forwarded.extend(["--checkpoint", args.checkpoint])
    if args.steps:
        forwarded.extend(["--max_steps", str(args.steps)])
    if args.headless:
        forwarded.append("--headless")
    if args.video:
        forwarded.extend(["--video", "--video_length", str(args.video_length)])
    if args.bev_follow:
        forwarded.extend(["--bev_follow", "--bev_height_m", str(args.bev_height_m)])
    forwarded.extend(passthrough)
    return run_senior_nav_script("play.py", forwarded)


def eval_senior_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Launch instinct_nav SRU G1 checkpoint evaluation.")
    parser.add_argument("--task", type=str, default="Instinct-Nav-G1-SRU-Play-v0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--json_out", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--save_failure_gifs", action="store_true", default=False)
    parser.add_argument("--failure_gif_folder", type=str, default=None)
    parser.add_argument("--max_success_gifs", type=int, default=10)
    parser.add_argument("--max_failure_gifs", type=int, default=10)
    parser.add_argument("--success_gif_min_initial_distance", type=float, default=0.0)
    parser.add_argument("--success_gif_min_goal_height_delta", type=float, default=0.0)
    parser.add_argument("--gif_terrain_name", type=str, default=None)
    parser.add_argument("--failure_rgb_base_contact_only", action="store_true", default=False)
    parser.add_argument("--failure_rgb_include_successes", action="store_true", default=True)
    parser.add_argument("--no_failure_rgb_include_successes", dest="failure_rgb_include_successes", action="store_false")
    parser.add_argument("--failure_rgb_include_observations", action="store_true", default=True)
    parser.add_argument("--no_failure_rgb_include_observations", dest="failure_rgb_include_observations", action="store_false")
    parser.add_argument("--failure_rgb_history_frames", type=int, default=10)
    parser.add_argument("--failure_rgb_gif_frame_ms", type=int, default=250)
    parser.add_argument("--observation_gif_folder", type=str, default=None)
    parser.add_argument("--max_observation_gifs", type=int, default=0)
    args, passthrough = parser.parse_known_args(argv)

    forwarded = [
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--num_episodes", str(args.num_episodes),
        "--print_every", str(args.print_every),
    ]
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if checkpoint_path is not None:
        forwarded.extend(["--checkpoint", str(checkpoint_path)])
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.headless:
        forwarded.append("--headless")
    if args.json_out:
        forwarded.extend(["--json_out", args.json_out])
    if args.save_failure_gifs:
        if args.failure_gif_folder:
            failure_dir = Path(args.failure_gif_folder).expanduser()
        elif checkpoint_path is not None:
            failure_dir = checkpoint_path.parent / "videos" / "failure_gifs"
        else:
            failure_dir = WORKSPACE_ROOT / "logs" / "eval_failure_gifs"
        forwarded.extend([
            "--failure_rgb_dir", str(failure_dir),
            "--failure_rgb_limit", str(args.max_failure_gifs),
            "--failure_rgb_success_limit", str(args.max_success_gifs),
            "--failure_rgb_failure_limit", str(args.max_failure_gifs),
            "--failure_rgb_success_min_initial_distance", str(args.success_gif_min_initial_distance),
            "--failure_rgb_success_min_goal_height_delta", str(args.success_gif_min_goal_height_delta),
            "--failure_rgb_export_mode", "gif",
            "--failure_rgb_history_frames", str(args.failure_rgb_history_frames),
            "--failure_rgb_gif_frame_ms", str(args.failure_rgb_gif_frame_ms),
        ])
    if args.gif_terrain_name:
        forwarded.extend(["--failure_rgb_terrain_name", args.gif_terrain_name])
    if args.failure_rgb_base_contact_only:
        forwarded.append("--failure_rgb_base_contact_only")
    if args.failure_rgb_include_successes:
        forwarded.append("--failure_rgb_include_successes")
    if args.failure_rgb_include_observations:
        forwarded.append("--failure_rgb_include_observations")
    if args.observation_gif_folder:
        forwarded.extend(["--observation_gif_dir", args.observation_gif_folder])
    if args.max_observation_gifs:
        forwarded.extend(["--observation_gif_limit", str(args.max_observation_gifs)])
    forwarded.extend(passthrough)
    return run_senior_nav_script("eval.py", forwarded)


COMMANDS = {
    "doctor": doctor_main,
    "eval": eval_senior_main,
    "eval-senior": eval_senior_main,
    "play": play_senior_main,
    "play-senior": play_senior_main,
    "train": train_senior_main,
    "train-senior": train_senior_main,
    "inspect-terrain": inspect_terrain_main,
    "visualize-terrain": visualize_terrain_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="instinct_nav workspace launcher")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, passthrough = parser.parse_known_args()
    return COMMANDS[args.command](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
