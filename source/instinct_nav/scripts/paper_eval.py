#!/usr/bin/env python3
"""Paper-grade evaluation orchestrator for navigation policies.

This script intentionally stays outside Isaac Lab runtime imports. It wraps
``eval.py`` so paper evaluations use one fixed case manifest across policies,
then aggregates the resulting JSON summaries into table-ready metrics.

Example config:

{
  "python": "/workspace/instinctlab51-env/bin/python",
  "eval_script": "/workspace/instinct_nav/source/instinct_nav/scripts/eval.py",
  "task": "Isaac-Nav-PPO-AoW-D-v0",
  "output_dir": "/workspace/instinct_nav/.cache/paper_eval/aow_main",
  "num_envs": 512,
  "num_episodes": 512,
  "seed": 42,
  "episode_length_s": 60.0,
  "headless": true,
  "extra_args": [],
  "policies": [
    {
      "name": "sru_baseline",
      "checkpoint": "/workspace/instinct_nav/logs/rsl_rl/.../model_8000.pt"
    },
    {
      "name": "query_map",
      "checkpoint": "/workspace/instinct_nav/logs/rsl_rl/.../model_14999.pt"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class PolicySpec(NamedTuple):
    name: str
    checkpoint: Path
    summary: Path | None = None
    robot: str = ""
    case_protocol: str = "fixed initial case manifest"
    usable_level: str = "paper-usable"
    notes: str = ""


RESULT_SHEET_COLUMNS = [
    "variant",
    "robot",
    "checkpoint",
    "summary_json",
    "episodes",
    "num_envs",
    "case_protocol",
    "successes",
    "success_rate",
    "base_contact_rate",
    "timeout_rate",
    "large_pitch_rate",
    "terrain_fall_rate",
    "ci_low",
    "ci_high",
    "usable_level",
    "notes",
]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score confidence interval for a binomial rate."""
    if total <= 0:
        return 0.0, 0.0
    phat = float(successes) / float(total)
    denom = 1.0 + (z * z) / float(total)
    center = (phat + (z * z) / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
        / denom
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    required = ["task", "output_dir", "num_envs", "num_episodes", "policies"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if not config["policies"]:
        raise ValueError("Config must include at least one policy.")
    task = str(config.get("task", ""))
    if "G1" in task or "g1" in task or "Instinct" in task:
        raise ValueError(
            "G1/Instinct evaluations belong to a separate project and must not be mixed into the explicit memory paper."
        )
    return config


def parse_policies(config: dict, output_dir: Path) -> list[PolicySpec]:
    policies = []
    default_robot = str(config.get("robot", ""))
    default_case_protocol = str(config.get("case_protocol", "fixed initial case manifest"))
    default_usable_level = str(config.get("usable_level", "paper-usable"))
    default_notes = str(config.get("notes", ""))
    for item in config["policies"]:
        if "name" not in item or "checkpoint" not in item:
            raise ValueError("Each policy must define 'name' and 'checkpoint'.")
        name = str(item["name"])
        if not name:
            raise ValueError("Policy name cannot be empty.")
        policies.append(
            PolicySpec(
                name=name,
                checkpoint=Path(str(item["checkpoint"])).expanduser(),
                summary=output_dir / f"{name}.summary.json",
                robot=str(item.get("robot", default_robot)),
                case_protocol=str(item.get("case_protocol", default_case_protocol)),
                usable_level=str(item.get("usable_level", default_usable_level)),
                notes=str(item.get("notes", default_notes)),
            )
        )
    return policies


def build_eval_command(
    *,
    python_executable: str,
    eval_script: Path,
    task: str,
    checkpoint: Path,
    num_envs: int,
    num_episodes: int,
    seed: int | None,
    episode_length_s: float | None,
    manifest_in: Path | None,
    manifest_out: Path | None,
    json_out: Path,
    headless: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        python_executable,
        str(eval_script),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(int(num_envs)),
        "--num_episodes",
        str(int(num_episodes)),
        "--initial_env_cases_only",
        "--json_out",
        str(json_out),
    ]
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if episode_length_s is not None:
        command.extend(["--episode_length_s", str(float(episode_length_s))])
    if manifest_in is not None:
        command.extend(["--case_manifest_in", str(manifest_in)])
    if manifest_out is not None:
        command.extend(["--case_manifest_out", str(manifest_out)])
    if headless:
        command.append("--headless")
    command.extend(extra_args)
    return command


def _rate_block(count: int, total: int) -> dict[str, object]:
    low, high = wilson_interval(count, total)
    return {
        "count": int(count),
        "total": int(total),
        "rate": float(count) / float(total) if total > 0 else 0.0,
        "ci95": [low, high],
    }


def _termination_rate(summary: dict, name: str, episodes: int) -> float:
    values = summary.get("termination_rates", {}).get(name, {})
    if "rate" in values:
        return float(values["rate"])
    count = int(values.get("count", 0))
    return float(count) / float(episodes) if episodes > 0 else 0.0


def make_result_row(
    summary: dict,
    *,
    variant: str,
    robot: str,
    checkpoint: Path | str,
    summary_json: Path | str,
    case_protocol: str,
    usable_level: str,
    notes: str = "",
) -> dict[str, object]:
    episodes = int(summary.get("episodes", 0))
    successes = int(summary.get("successes", round(float(summary.get("success_rate", 0.0)) * episodes)))
    ci_low, ci_high = wilson_interval(successes, episodes)
    return {
        "variant": str(variant),
        "robot": str(robot),
        "checkpoint": str(checkpoint),
        "summary_json": str(summary_json),
        "episodes": episodes,
        "num_envs": int(summary.get("num_envs", 0)),
        "case_protocol": str(case_protocol),
        "successes": successes,
        "success_rate": float(successes) / float(episodes) if episodes > 0 else 0.0,
        "base_contact_rate": _termination_rate(summary, "base_contact", episodes),
        "timeout_rate": _termination_rate(summary, "time_out", episodes),
        "large_pitch_rate": _termination_rate(summary, "large_pitch_angle", episodes),
        "terrain_fall_rate": _termination_rate(summary, "terrain_fall", episodes),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "usable_level": str(usable_level),
        "notes": str(notes),
    }


def format_result_sheet_csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=RESULT_SHEET_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in RESULT_SHEET_COLUMNS})
    return output.getvalue()


def aggregate_summaries(policies: list[PolicySpec]) -> dict[str, object]:
    rows = []
    result_sheet_rows = []
    for policy in policies:
        if policy.summary is None:
            raise ValueError(f"Policy {policy.name} has no summary path.")
        with Path(policy.summary).open("r", encoding="utf-8") as f:
            summary = json.load(f)

        episodes = int(summary.get("episodes", 0))
        successes = int(summary.get("successes", round(float(summary.get("success_rate", 0.0)) * episodes)))
        terminations = {}
        for name, values in summary.get("termination_rates", {}).items():
            count = int(values.get("count", round(float(values.get("rate", 0.0)) * episodes)))
            terminations[name] = _rate_block(count, episodes)

        rows.append(
            {
                "name": policy.name,
                "checkpoint": str(policy.checkpoint),
                "episodes": episodes,
                "success": _rate_block(successes, episodes),
                "terminations": terminations,
            }
        )
        result_sheet_rows.append(
            make_result_row(
                summary,
                variant=policy.name,
                robot=policy.robot,
                checkpoint=policy.checkpoint,
                summary_json=policy.summary,
                case_protocol=policy.case_protocol,
                usable_level=policy.usable_level,
                notes=policy.notes,
            )
        )
    return {"policies": rows, "result_sheet": result_sheet_rows}


def format_markdown_table(aggregate: dict[str, object]) -> str:
    lines = [
        "| Policy | Episodes | Success % [95% CI] | Base Contact % | Timeout % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate["policies"]:
        success = row["success"]
        base = row["terminations"].get("base_contact", {"rate": 0.0})
        timeout = row["terminations"].get("time_out", {"rate": 0.0})
        ci_low, ci_high = success["ci95"]
        lines.append(
            "| {name} | {episodes} | {succ:.1f} [{lo:.1f}, {hi:.1f}] | {base:.1f} | {timeout:.1f} |".format(
                name=row["name"],
                episodes=row["episodes"],
                succ=100.0 * success["rate"],
                lo=100.0 * ci_low,
                hi=100.0 * ci_high,
                base=100.0 * base["rate"],
                timeout=100.0 * timeout["rate"],
            )
        )
    return "\n".join(lines) + "\n"


def run_policy_evals(config: dict, dry_run: bool) -> list[PolicySpec]:
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = parse_policies(config, output_dir)
    manifest_path = output_dir / "case_manifest.json"

    python_executable = str(config.get("python", sys.executable))
    eval_script = Path(str(config.get("eval_script", Path(__file__).with_name("eval.py")))).expanduser()
    extra_args = [str(value) for value in config.get("extra_args", [])]

    for index, policy in enumerate(policies):
        command = build_eval_command(
            python_executable=python_executable,
            eval_script=eval_script,
            task=str(config["task"]),
            checkpoint=policy.checkpoint,
            num_envs=int(config["num_envs"]),
            num_episodes=int(config["num_episodes"]),
            seed=None if config.get("seed") is None else int(config["seed"]),
            episode_length_s=None if config.get("episode_length_s") is None else float(config["episode_length_s"]),
            manifest_in=None if index == 0 else manifest_path,
            manifest_out=manifest_path if index == 0 else None,
            json_out=policy.summary,
            headless=bool(config.get("headless", True)),
            extra_args=extra_args,
        )
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, check=True, cwd=str(Path.cwd()))
    return policies


def write_outputs(output_dir: Path, aggregate: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "aggregate.json"
    table_path = output_dir / "paper_table.md"
    result_sheet_path = output_dir / "paper_mvp_result_sheet.csv"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")
    table_path.write_text(format_markdown_table(aggregate), encoding="utf-8")
    result_sheet_path.write_text(format_result_sheet_csv(list(aggregate.get("result_sheet", []))), encoding="utf-8")
    print(f"[INFO] Wrote aggregate JSON: {aggregate_path}")
    print(f"[INFO] Wrote markdown table: {table_path}")
    print(f"[INFO] Wrote result sheet CSV: {result_sheet_path}")


def command_run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    policies = run_policy_evals(config, dry_run=bool(args.dry_run))
    if not args.dry_run:
        aggregate = aggregate_summaries(policies)
        write_outputs(Path(str(config["output_dir"])).expanduser().resolve(), aggregate)


def command_aggregate(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    policies = parse_policies(config, output_dir)
    aggregate = aggregate_summaries(policies)
    write_outputs(output_dir, aggregate)


def command_example_config(_: argparse.Namespace) -> None:
    example = {
        "python": "/workspace/instinctlab51-env/bin/python",
        "eval_script": "/workspace/instinct_nav/source/instinct_nav/scripts/eval.py",
        "task": "Isaac-Nav-PPO-AoW-D-v0",
        "output_dir": "/workspace/instinct_nav/.cache/paper_eval/aow_main",
        "num_envs": 512,
        "num_episodes": 512,
        "seed": 42,
        "episode_length_s": 60.0,
        "headless": True,
        "extra_args": [],
        "policies": [
            {"name": "sru_baseline", "checkpoint": "/workspace/instinct_nav/logs/rsl_rl/.../model_8000.pt"},
            {"name": "query_map", "checkpoint": "/workspace/instinct_nav/logs/rsl_rl/.../model_14999.pt"},
        ],
    }
    print(json.dumps(example, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run all policies and aggregate summaries.")
    run_parser.add_argument("--config", required=True, help="Path to paper_eval JSON config.")
    run_parser.add_argument("--dry_run", action="store_true", help="Print eval.py commands without running them.")
    run_parser.set_defaults(func=command_run)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate existing summary JSON files.")
    aggregate_parser.add_argument("--config", required=True, help="Path to paper_eval JSON config.")
    aggregate_parser.set_defaults(func=command_aggregate)

    example_parser = subparsers.add_parser("example_config", help="Print an example config JSON.")
    example_parser.set_defaults(func=command_example_config)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
