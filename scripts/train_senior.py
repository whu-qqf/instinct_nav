#!/usr/bin/env python3
"""Train with the instinct_nav SRU G1 navigation stack.

This entry point intentionally bypasses the local instinct_rl runner and task
registry.  It imports the project instinct_nav package and launches the
project-local rsl_rl-based training path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "source"
EXTERNAL_ROOT = WORKSPACE_ROOT / "external"
def _project_path(env_name: str, project_default: Path) -> Path:
    if env_name in os.environ:
        return Path(os.environ[env_name]).expanduser()
    return project_default


CURRENT_ISAACLAB_ROOT = Path(os.environ.get("INSTINCT_NAV_ISAACLAB_ROOT", "/workspace/IsaacLab-2.3.2"))
CURRENT_ISAACLAB_SOURCE = CURRENT_ISAACLAB_ROOT / "source"
CURRENT_ISAACLAB_EXT_ROOT = CURRENT_ISAACLAB_SOURCE / "isaaclab"
CURRENT_ISAACLAB_TASKS_ROOT = CURRENT_ISAACLAB_SOURCE / "isaaclab_tasks"
CURRENT_ISAACLAB_RL_ROOT = Path(os.environ.get("INSTINCT_NAV_ISAACLAB_RL_ROOT", str(CURRENT_ISAACLAB_SOURCE / "isaaclab_rl")))
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

# IsaacLab core/AppLauncher must stay on the current IsaacSim 5.1 compatible tree.
# Senior IsaacLab core targets an older cp310 Kit app and cannot launch in this env.
senior_pythonpath_entries = [
    SENIOR_RSL_RL_ROOT,
    SENIOR_NAV_TASK_ROOT,
    SENIOR_INSTINCTLAB_SOURCE,
    SENIOR_INSTINCT_RL_ROOT,
]
for path in reversed(senior_pythonpath_entries):
    if path.exists():
        sys.path.insert(0, str(path))

os.environ.setdefault("SRU_LEGACY_RSL_RL_ROOT", str(SENIOR_RSL_RL_ROOT / "rsl_rl"))
os.environ.setdefault("SRU_INSTINCT_RL_ROOT", str(SENIOR_INSTINCT_RL_ROOT))
os.environ.setdefault("SRU_G1_INSTINCT_CHECKPOINT", str(SENIOR_G1_CHECKPOINT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Train instinct_nav SRU G1 navigation with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Instinct-Nav-G1-SRU-v0", help="instinct_nav task name.")
parser.add_argument("--seed", type=int, default=None, help="Override the runner seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Append a custom suffix to the log directory.")
parser.add_argument("--wandb", action="store_true", default=False, help="Use wandb logger instead of tensorboard.")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity override.")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project override.")
parser.add_argument("--resume_checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--no_load_optimizer", action="store_true", default=False, help="Resume model weights without optimizer state.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import instinct_nav  # noqa: F401,E402

from isaaclab.utils.io import dump_yaml  # noqa: E402

try:
    from isaaclab.utils.io import dump_pickle  # noqa: E402
except ImportError:
    import pickle

    def dump_pickle(filename: str, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402


class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Match the legacy SRU rsl_rl observation/extras contract."""

    def _to_legacy_obs(self, obs):
        policy_obs = obs["policy"]
        critic_obs = obs["critic"] if "critic" in obs.keys() else policy_obs
        legacy = {"critic": critic_obs}
        # Pass through auxiliary observation groups (e.g. the training-only mask label).
        if "mask_label" in obs.keys():
            legacy["mask_label"] = obs["mask_label"]
        return policy_obs, {"observations": legacy}

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
    import rsl_rl.modules as rsl_modules
    import rsl_rl.networks as rsl_networks
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    legacy_root = os.environ["SRU_LEGACY_RSL_RL_ROOT"]
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

_rsl_rl_file = Path(_rsl_rl_pkg.__file__).resolve() if _rsl_rl_pkg.__file__ else None
_senior_rsl_root = SENIOR_RSL_RL_ROOT.resolve()
if _rsl_rl_file is not None and str(_rsl_rl_file).startswith(str(_senior_rsl_root)):
    print(f"[INFO] Using senior rsl_rl package without adapter: {_rsl_rl_file}")
else:
    _register_sru_rsl_rl_extensions()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _module_file(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"<import failed: {exc}>"
    file = getattr(module, "__file__", None)
    if file is None:
        search_locations = getattr(getattr(module, "__spec__", None), "submodule_search_locations", None)
        if search_locations:
            return list(search_locations)[0]
        return None
    return str(Path(file).resolve())


def _sha256(path: str | os.PathLike | None) -> str | None:
    if path in (None, ""):
        return None
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return f"<missing: {file_path}>"
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_attr(obj, name: str, default=None):
    return getattr(obj, name, default)


def _cfg_summary(env_cfg, agent_cfg) -> dict:
    action_cfg = _safe_attr(_safe_attr(env_cfg, "actions"), "velocity_command")
    scene_cfg = _safe_attr(env_cfg, "scene")
    raycast_camera = _safe_attr(scene_cfg, "raycast_camera")
    camera = _safe_attr(scene_cfg, "camera")
    terrain = _safe_attr(scene_cfg, "terrain")
    terrain_generator = _safe_attr(terrain, "terrain_generator")
    return {
        "task": args_cli.task,
        "num_envs": _safe_attr(scene_cfg, "num_envs"),
        "seed": _safe_attr(agent_cfg, "seed"),
        "max_iterations": _safe_attr(agent_cfg, "max_iterations"),
        "logger": _safe_attr(agent_cfg, "logger"),
        "experiment_name": _safe_attr(agent_cfg, "experiment_name"),
        "run_name": _safe_attr(agent_cfg, "run_name"),
        "resume_checkpoint": args_cli.resume_checkpoint,
        "num_steps_per_env": _safe_attr(agent_cfg, "num_steps_per_env"),
        "empirical_normalization": _safe_attr(agent_cfg, "empirical_normalization"),
        "obs_groups": _safe_attr(agent_cfg, "obs_groups"),
        "policy": {
            "class_name": _safe_attr(_safe_attr(agent_cfg, "policy"), "class_name"),
            "actor_hidden_dims": _safe_attr(_safe_attr(agent_cfg, "policy"), "actor_hidden_dims"),
            "critic_hidden_dims": _safe_attr(_safe_attr(agent_cfg, "policy"), "critic_hidden_dims"),
            "rnn_hidden_size": _safe_attr(_safe_attr(agent_cfg, "policy"), "rnn_hidden_size"),
            "dropout": _safe_attr(_safe_attr(agent_cfg, "policy"), "dropout"),
            "init_noise_std": _safe_attr(_safe_attr(agent_cfg, "policy"), "init_noise_std"),
            "num_cameras": _safe_attr(_safe_attr(agent_cfg, "policy"), "num_cameras"),
            "use_actor_visibility_map": _safe_attr(_safe_attr(agent_cfg, "policy"), "use_actor_visibility_map"),
            "use_actor_visibility_query_attention": _safe_attr(_safe_attr(agent_cfg, "policy"), "use_actor_visibility_query_attention"),
        },
        "algorithm": {
            "class_name": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "class_name"),
            "num_learning_epochs": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "num_learning_epochs"),
            "num_mini_batches": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "num_mini_batches"),
            "learning_rate": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "learning_rate"),
            "schedule": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "schedule"),
            "desired_kl": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "desired_kl"),
            "gamma": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "gamma"),
            "lam": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "lam"),
            "entropy_coef": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "entropy_coef"),
            "value_loss_coef": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "value_loss_coef"),
            "max_grad_norm": _safe_attr(_safe_attr(agent_cfg, "algorithm"), "max_grad_norm"),
        },
        "action": {
            "class_type": str(_safe_attr(action_cfg, "class_type")),
            "backend_type": _safe_attr(action_cfg, "backend_type"),
            "checkpoint_path": _safe_attr(action_cfg, "checkpoint_path"),
            "checkpoint_sha256": _sha256(_safe_attr(action_cfg, "checkpoint_path")),
            "actor_onnx_path": _safe_attr(action_cfg, "actor_onnx_path"),
            "actor_onnx_sha256": _sha256(_safe_attr(action_cfg, "actor_onnx_path")),
            "depth_encoder_onnx_path": _safe_attr(action_cfg, "depth_encoder_onnx_path"),
            "depth_encoder_onnx_sha256": _sha256(_safe_attr(action_cfg, "depth_encoder_onnx_path")),
            "instinct_rl_root": _safe_attr(action_cfg, "instinct_rl_root"),
            "low_level_decimation": _safe_attr(action_cfg, "low_level_decimation"),
            "policy_scaling": _safe_attr(action_cfg, "policy_scaling"),
            "use_raw_actions": _safe_attr(action_cfg, "use_raw_actions"),
            "enable_low_pass_filter": _safe_attr(action_cfg, "enable_low_pass_filter"),
            "low_pass_filter_alpha": _safe_attr(action_cfg, "low_pass_filter_alpha"),
            "observation_group": _safe_attr(action_cfg, "observation_group"),
            "command_name": _safe_attr(action_cfg, "command_name"),
        },
        "camera": {
            "raycast_prim_path": _safe_attr(raycast_camera, "prim_path"),
            "raycast_update_period": _safe_attr(raycast_camera, "update_period"),
            "raycast_offset": str(_safe_attr(raycast_camera, "offset")),
            "raycast_pattern": str(_safe_attr(raycast_camera, "pattern_cfg")),
            "low_level_camera_prim_path": _safe_attr(camera, "prim_path"),
            "low_level_camera_update_period": _safe_attr(camera, "update_period"),
            "low_level_camera_offset": str(_safe_attr(camera, "offset")),
        },
        "terrain": {
            "max_init_terrain_level": _safe_attr(terrain, "max_init_terrain_level"),
            "difficulty_range": _safe_attr(terrain_generator, "difficulty_range"),
            "curriculum": _safe_attr(terrain_generator, "curriculum"),
        },
    }


def _runtime_diagnostics(env, env_cfg, agent_cfg) -> dict:
    obs, extras = env.get_observations()
    critic_obs = extras.get("observations", {}).get("critic") if isinstance(extras, dict) else None
    diagnostics = _cfg_summary(env_cfg, agent_cfg)
    diagnostics["imports"] = {
        "isaaclab.app": _module_file("isaaclab.app"),
        "isaaclab.envs": _module_file("isaaclab.envs"),
        "isaaclab_tasks.utils.parse_cfg": _module_file("isaaclab_tasks.utils.parse_cfg"),
        "isaaclab_rl.rsl_rl": _module_file("isaaclab_rl.rsl_rl"),
        "instinct_nav": _module_file("instinct_nav"),
        "instinct_nav.navigation": _module_file("instinct_nav.navigation"),
        "instinct_nav.navigation.config.g1_instinct.navigation_env_cfg": _module_file("instinct_nav.navigation.config.g1_instinct.navigation_env_cfg"),
        "rsl_rl": _module_file("rsl_rl"),
        "rsl_rl.runners.on_policy_runner": _module_file("rsl_rl.runners.on_policy_runner"),
        "rsl_rl.modules.actor_critic_sru": _module_file("rsl_rl.modules.actor_critic_sru"),
        "rsl_rl.algorithms.ppo": _module_file("rsl_rl.algorithms.ppo"),
        "instinctlab": _module_file("instinctlab"),
    }
    diagnostics["env_runtime"] = {
        "num_envs": getattr(env, "num_envs", None),
        "num_actions": getattr(env, "num_actions", None),
        "policy_obs_shape": list(obs.shape) if hasattr(obs, "shape") else str(type(obs)),
        "critic_obs_shape": list(critic_obs.shape) if hasattr(critic_obs, "shape") else None,
        "device": str(getattr(env, "device", None)),
    }
    diagnostics["sensors"] = {}
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    if scene is not None:
        sensors = getattr(scene, "sensors", {})
        for name in ["raycast_camera", "camera", "height_scanner", "height_scanner_critic"]:
            sensor = sensors.get(name) if hasattr(sensors, "get") else None
            if sensor is not None:
                diagnostics["sensors"][name] = {
                    "class": sensor.__class__.__module__ + "." + sensor.__class__.__name__,
                    "image_shape": list(getattr(sensor, "image_shape", [])) if getattr(sensor, "image_shape", None) is not None else None,
                    "device": str(getattr(sensor, "device", None)),
                }
    depth_encoder_path = SENIOR_NAV_TASK_ROOT / "instinct_nav" / "navigation" / "assets" / "data" / "Policies" / "depth_encoder" / "vae_pretrain_new.pth"
    diagnostics["depth_encoder"] = {
        "path": str(depth_encoder_path),
        "sha256": _sha256(depth_encoder_path),
    }
    return diagnostics


def _assert_senior_imports(diagnostics: dict) -> None:
    expected_prefixes = {
        "isaaclab.app": str(CURRENT_ISAACLAB_EXT_ROOT.resolve()),
        "isaaclab.envs": str(CURRENT_ISAACLAB_EXT_ROOT.resolve()),
        "isaaclab_tasks.utils.parse_cfg": str(CURRENT_ISAACLAB_TASKS_ROOT.resolve()),
        "isaaclab_rl.rsl_rl": str(CURRENT_ISAACLAB_RL_ROOT.resolve()),
        "instinct_nav": str(SENIOR_NAV_TASK_ROOT.resolve()),
        "instinct_nav.navigation.config.g1_instinct.navigation_env_cfg": str(SENIOR_NAV_TASK_ROOT.resolve()),
        "rsl_rl": str(SENIOR_RSL_RL_ROOT.resolve()),
        "rsl_rl.runners.on_policy_runner": str(SENIOR_RSL_RL_ROOT.resolve()),
        "rsl_rl.modules.actor_critic_sru": str(SENIOR_RSL_RL_ROOT.resolve()),
        "rsl_rl.algorithms.ppo": str(SENIOR_RSL_RL_ROOT.resolve()),
    }
    failures = []
    imports = diagnostics.get("imports", {})
    for module_name, expected_prefix in expected_prefixes.items():
        actual = imports.get(module_name)
        if actual is None or actual.startswith("<") or not str(actual).startswith(expected_prefix):
            failures.append(f"{module_name}: {actual} (expected prefix {expected_prefix})")
    if failures:
        raise RuntimeError("train-senior import path check failed:\n" + "\n".join(failures))


def _write_diagnostics(log_dir: str, diagnostics: dict) -> None:
    params_dir = Path(log_dir) / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    output_path = params_dir / "senior_diagnostics.json"
    output_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))
    print(f"[INFO] Wrote senior diagnostics: {output_path}")
    print("[INFO] Senior diagnostics summary:")
    print(json.dumps({
        "imports": diagnostics.get("imports", {}),
        "env_runtime": diagnostics.get("env_runtime", {}),
        "action": diagnostics.get("action", {}),
        "depth_encoder": diagnostics.get("depth_encoder", {}),
    }, indent=2, sort_keys=True, default=str))


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.wandb:
        agent_cfg.logger = "wandb"
    if args_cli.wandb_entity is not None:
        os.environ["WANDB_USERNAME"] = args_cli.wandb_entity
    if args_cli.wandb_project is not None:
        agent_cfg.wandb_project = args_cli.wandb_project

    resume_checkpoint = Path(args_cli.resume_checkpoint).expanduser() if args_cli.resume_checkpoint else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = LegacyRslRlVecEnvWrapper(env)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Using current IsaacLab extension root: {CURRENT_ISAACLAB_EXT_ROOT}")
    print(f"[INFO] Using instinct_nav task package: {SENIOR_NAV_TASK_ROOT}")
    print(f"[INFO] Using project rsl_rl extensions: {os.environ['SRU_LEGACY_RSL_RL_ROOT']}")
    print(f"[INFO] Using G1 low-level checkpoint: {os.environ['SRU_G1_INSTINCT_CHECKPOINT']}")
    print(f"[INFO] Logger: {agent_cfg.logger}")
    if agent_cfg.logger == "wandb":
        print(f"[INFO] wandb project: {agent_cfg.wandb_project}")
        print(f"[INFO] wandb entity: {os.environ.get('WANDB_USERNAME', None)}")
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    diagnostics = _runtime_diagnostics(env, env_cfg, agent_cfg)
    _assert_senior_imports(diagnostics)
    _write_diagnostics(log_dir, diagnostics)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Senior rsl_rl tries to snapshot git state from its own repository at learn() start.
    # The reference checkout is often mounted with dubious ownership, so keep code provenance
    # in senior_diagnostics.json and skip GitPython state capture for training robustness.
    runner.git_status_repos = []
    if resume_checkpoint is not None:
        print(f"[INFO] Resuming from checkpoint: {resume_checkpoint}")
        runner.load(str(resume_checkpoint), load_optimizer=not args_cli.no_load_optimizer)
        print(f"[INFO] Loaded checkpoint iteration: {runner.current_learning_iteration}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
