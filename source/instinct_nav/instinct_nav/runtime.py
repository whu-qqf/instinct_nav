from __future__ import annotations

import os
from pathlib import Path


def _resolve_existing_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_runtime_env(workspace_root: Path) -> dict[str, str]:
    env = os.environ.copy()

    cache_root = workspace_root / ".cache"
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env.setdefault("WANDB_DIR", str(cache_root / "wandb"))
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("PYTHONNOUSERSITE", "1")

    runtime_venv = _resolve_existing_dir(
        Path(env.get("INSTINCT_NAV_RUNTIME_VENV", "/workspace/instinctlab51-env")),
        "Runtime venv",
    )
    isaaclab_root = _resolve_existing_dir(
        Path(env.get("INSTINCT_NAV_ISAACLAB_ROOT", "/workspace/IsaacLab-2.3.2")),
        "IsaacLab root",
    )
    default_instinctlab_root = workspace_root / "external" / "instinctlab"
    if not default_instinctlab_root.exists():
        default_instinctlab_root = Path("/workspace/InstinctLab")
    instinctlab_root = _resolve_existing_dir(
        Path(env.get("INSTINCT_NAV_INSTINCTLAB_ROOT", str(default_instinctlab_root))),
        "InstinctLab root",
    )
    isaaclab_rl_root = _resolve_existing_dir(
        Path(
            env.get(
                "INSTINCT_NAV_ISAACLAB_RL_ROOT",
                str(isaaclab_root / "source" / "isaaclab_rl"),
            )
        ),
        "isaaclab_rl root",
    )

    env["INSTINCT_NAV_RUNTIME_VENV"] = str(runtime_venv)
    env["INSTINCT_NAV_ISAACLAB_ROOT"] = str(isaaclab_root)
    env["INSTINCT_NAV_INSTINCTLAB_ROOT"] = str(instinctlab_root)
    env["INSTINCT_NAV_ISAACLAB_RL_ROOT"] = str(isaaclab_rl_root)

    python_bin = runtime_venv / "bin" / "python"
    if not python_bin.exists():
        raise FileNotFoundError(f"Python runtime not found: {python_bin}")

    source_root = isaaclab_root / "source"
    if (instinctlab_root / "instinctlab").exists():
        instinctlab_source = instinctlab_root
    else:
        instinctlab_source = instinctlab_root / "source" / "instinctlab"
    project_source = workspace_root / "source" / "instinct_nav"
    project_rsl_rl = workspace_root / "source" / "rsl_rl"
    project_instinct_rl = workspace_root / "external" / "instinct_rl"
    for path in [cache_root, cache_root / "logs", cache_root / "wandb"]:
        path.mkdir(parents=True, exist_ok=True)

    env["PATH"] = f"{runtime_venv / 'bin'}:{env.get('PATH', '')}"
    pythonpath_entries = [
        str(project_source),
        str(project_rsl_rl),
        str(project_instinct_rl),
        str(workspace_root),
        str(source_root),
        str(isaaclab_rl_root),
        str(instinctlab_source),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def python_executable(env: dict[str, str]) -> str:
    return str(Path(env["INSTINCT_NAV_RUNTIME_VENV"]) / "bin" / "python")
