#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_KEYS = (
    "base_lin_vel_b",
    "base_ang_vel_b",
    "projected_gravity_b",
    "depth_m",
    "filtered_depth",
    "encoded_depth",
    "scene_tokens",
    "actor_obs",
    "raw_action",
    "actions",
    "command",
)


@dataclass
class ArraySummary:
    key: str
    count: int
    numel: int
    finite_ratio: float
    mean: float
    std: float
    min: float
    max: float


def main() -> None:
    args = parse_args()
    left = load_dump(Path(args.left), args.max_files, args.keys)
    right = load_dump(Path(args.right), args.max_files, args.keys)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise SystemExit("No common array keys found between debug dumps.")

    rows = []
    for key in keys:
        lsum = summarize(key, left[key])
        rsum = summarize(key, right[key])
        rows.append(
            {
                "key": key,
                "left": lsum.__dict__,
                "right": rsum.__dict__,
                "delta_mean": rsum.mean - lsum.mean,
                "delta_std": rsum.std - lsum.std,
                "mean_abs_delta": mean_abs_delta(left[key], right[key]),
            }
        )

    print_table(rows, args.left_label, args.right_label)
    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote comparison JSON: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two InstinctNav debug dump directories, e.g. IsaacLab eval vs robot dry-run."
    )
    parser.add_argument("--left", required=True, help="First debug dump directory.")
    parser.add_argument("--right", required=True, help="Second debug dump directory.")
    parser.add_argument("--left_label", default="left")
    parser.add_argument("--right_label", default="right")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def load_dump(dump_dir: Path, max_files: int, keys: Iterable[str]) -> dict[str, list[np.ndarray]]:
    dump_dir = dump_dir.expanduser()
    if not dump_dir.exists():
        raise FileNotFoundError(dump_dir)
    wanted = set(keys)
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in wanted}
    files = sorted(dump_dir.glob("*.npz"))[: max(max_files, 1)]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dump_dir}")
    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            for key in wanted:
                if key in data:
                    arrays[key].append(np.asarray(data[key], dtype=np.float32).reshape(-1))
    return {key: value for key, value in arrays.items() if value}


def summarize(key: str, chunks: list[np.ndarray]) -> ArraySummary:
    arr = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return ArraySummary(key, len(chunks), int(arr.size), 0.0, float("nan"), float("nan"), float("nan"), float("nan"))
    finite_arr = arr[finite]
    return ArraySummary(
        key=key,
        count=len(chunks),
        numel=int(arr.size),
        finite_ratio=float(finite.mean()),
        mean=float(finite_arr.mean()),
        std=float(finite_arr.std()),
        min=float(finite_arr.min()),
        max=float(finite_arr.max()),
    )


def mean_abs_delta(left_chunks: list[np.ndarray], right_chunks: list[np.ndarray]) -> float | None:
    n = min(len(left_chunks), len(right_chunks))
    if n <= 0:
        return None
    deltas = []
    for left, right in zip(left_chunks[:n], right_chunks[:n]):
        if left.shape != right.shape:
            continue
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            deltas.append(float(np.mean(np.abs(right[finite] - left[finite]))))
    if not deltas:
        return None
    return float(np.mean(deltas))


def print_table(rows: list[dict], left_label: str, right_label: str) -> None:
    print(f"Compared debug dumps: {left_label} -> {right_label}")
    print(
        f"{'key':<22} {'nL':>5} {'nR':>5} "
        f"{left_label + '_mean':>14} {right_label + '_mean':>14} {'d_mean':>12} "
        f"{left_label + '_std':>12} {right_label + '_std':>12} {'mean_abs_delta':>15}"
    )
    for row in rows:
        left = row["left"]
        right = row["right"]
        mad = row["mean_abs_delta"]
        mad_str = "n/a" if mad is None else f"{mad:.6g}"
        print(
            f"{row['key']:<22} {left['count']:>5} {right['count']:>5} "
            f"{left['mean']:>14.6g} {right['mean']:>14.6g} {row['delta_mean']:>12.6g} "
            f"{left['std']:>12.6g} {right['std']:>12.6g} {mad_str:>15}"
        )


if __name__ == "__main__":
    main()
