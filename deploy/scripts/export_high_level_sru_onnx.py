#!/usr/bin/env python3
"""Export the scene-token SRU high-level navigation policy to ONNX.

The deployed robot environment should not need PyTorch for high-level inference.
This exporter exposes SRU hidden/cell states as explicit ONNX inputs and outputs:

    obs [B, 2960], h_in [1, B, 512], c_in [1, B, 512]
      -> actions [B, 3], h_out [1, B, 512], c_out [1, B, 512]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_SOURCE = REPO_ROOT / "source" / "rsl_rl"
DEPLOY_SOURCE = REPO_ROOT / "deploy"
for source_path in (DEPLOY_SOURCE, RSL_RL_SOURCE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from instinct_nav_onboard.sru_checkpoint_policy import SceneTokenSruCheckpointPolicy  # noqa: E402


class SceneTokenSruOnnxActor(nn.Module):
    """ONNX-exportable actor path for current scene-token ActorCriticSRU."""

    def __init__(self, actor_critic: nn.Module):
        super().__init__()
        ac = actor_critic
        self.attn_image_net = copy.deepcopy(ac.attn_image_net)
        self.rnn = copy.deepcopy(ac.memory_a.rnn)
        self.linear = copy.deepcopy(ac.linear_dropout_actor.linear)
        self.linear_activation = copy.deepcopy(ac.linear_dropout_actor.activation)
        self.actor = copy.deepcopy(ac.actor)
        self.scene_token_projector = copy.deepcopy(ac.scene_token_projector)
        self.scene_token_position_embedding = nn.Parameter(ac.scene_token_position_embedding.detach().clone())
        self.scene_token_norm = copy.deepcopy(ac.scene_token_norm)
        self.scene_query_proj = copy.deepcopy(ac.scene_query_proj)
        self.scene_query_norm = copy.deepcopy(ac.scene_query_norm)
        self.scene_cross_attention = copy.deepcopy(ac.scene_cross_attention)
        self.scene_attention_ffn = copy.deepcopy(ac.scene_attention_ffn)

        self.num_image_features = int(ac.num_image_features)
        self.num_actor_scene_token_features = int(ac.num_actor_scene_token_features)
        self.actor_proprioceptive_input_dim = int(ac.actor_proprioceptive_input_dim)
        self.scene_token_num_tokens = int(ac.scene_token_num_tokens)
        self.scene_token_dim = int(ac.scene_token_dim)
        self.scene_token_feature_dim = int(ac.scene_token_feature_dim)
        self.image_c, self.image_h, self.image_w = ac.image_input_dims

    def forward(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_tail = self.num_image_features + self.num_actor_scene_token_features
        proprio = obs[..., :-visual_tail].reshape(-1, self.actor_proprioceptive_input_dim)
        image_start = -(self.num_image_features + self.num_actor_scene_token_features)
        image_end = -self.num_actor_scene_token_features
        image = obs[..., image_start:image_end].reshape(-1, self.image_c, self.image_h, self.image_w)
        scene_tokens = obs[..., -self.num_actor_scene_token_features :].reshape(
            -1, self.scene_token_num_tokens, self.scene_token_dim
        )

        image_features = self.attn_image_net(image, proprio)
        recurrent_input = torch.cat([image_features, proprio], dim=-1)
        recurrent_features, (h_out, c_out) = self.rnn(recurrent_input.unsqueeze(0), (h_in, c_in))
        recurrent_features = recurrent_features.squeeze(0)

        tokens = self.scene_token_projector(scene_tokens)
        tokens = tokens + self.scene_token_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
        query_context = torch.cat([recurrent_features, proprio], dim=-1)
        query = self.scene_query_proj(query_context).unsqueeze(1)
        query = self.scene_query_norm(query)
        tokens = self.scene_token_norm(tokens)
        attended, _ = self.scene_cross_attention(query, tokens, tokens, need_weights=False)
        scene_features = attended.squeeze(1)
        scene_features = scene_features + self.scene_attention_ffn(scene_features)

        actor_features = torch.cat([recurrent_features, scene_features], dim=-1)
        actor_features = self.linear_activation(self.linear(actor_features))
        actions = self.actor(actor_features)
        return actions, h_out, c_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current scene-token SRU high-level policy to ONNX.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="sru_high_level_scene_token.onnx")
    parser.add_argument("--manifest_name", type=str, default="deploy_manifest.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--parity_atol", type=float, default=1.0e-5)
    parser.add_argument("--parity_rtol", type=float, default=1.0e-4)
    parser.add_argument("--require_onnx_parity", action="store_true", default=False)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_onnx_parity_check(
    exporter: nn.Module,
    onnx_path: Path,
    obs_dim: int,
    num_layers: int,
    hidden_size: int,
    atol: float,
    rtol: float,
    require: bool,
) -> bool:
    try:
        import onnxruntime as ort
    except Exception as exc:
        message = f"onnxruntime is unavailable; skipping ONNX parity check: {exc}"
        if require:
            raise RuntimeError(message) from exc
        print(f"[WARN] {message}")
        return False

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator(device="cpu").manual_seed(72)
    test_cases = [
        ("zero", torch.zeros(1, obs_dim), torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size)),
        ("random", torch.randn(1, obs_dim, generator=generator), torch.randn(num_layers, 1, hidden_size, generator=generator) * 0.1, torch.randn(num_layers, 1, hidden_size, generator=generator) * 0.1),
    ]
    exporter.eval().cpu()
    with torch.no_grad():
        for name, obs, h, c in test_cases:
            pt_actions, pt_h, pt_c = exporter(obs.float(), h.float(), c.float())
            ort_actions, ort_h, ort_c = session.run(
                None,
                {
                    "obs": obs.numpy().astype("float32"),
                    "h_in": h.numpy().astype("float32"),
                    "c_in": c.numpy().astype("float32"),
                },
            )
            _assert_close(name, "actions", pt_actions.numpy(), ort_actions, atol, rtol)
            _assert_close(name, "h_out", pt_h.numpy(), ort_h, atol, rtol)
            _assert_close(name, "c_out", pt_c.numpy(), ort_c, atol, rtol)

        h = torch.randn(num_layers, 1, hidden_size, generator=generator) * 0.1
        c = torch.randn(num_layers, 1, hidden_size, generator=generator) * 0.1
        ort_h = h.numpy().astype("float32")
        ort_c = c.numpy().astype("float32")
        for step in range(4):
            obs = torch.randn(1, obs_dim, generator=generator).float()
            pt_actions, h, c = exporter(obs, h.float(), c.float())
            ort_actions, ort_h, ort_c = session.run(
                None,
                {"obs": obs.numpy().astype("float32"), "h_in": ort_h, "c_in": ort_c},
            )
            _assert_close(f"rollout_{step}", "actions", pt_actions.numpy(), ort_actions, atol, rtol)
            _assert_close(f"rollout_{step}", "h_out", h.numpy(), ort_h, atol, rtol)
            _assert_close(f"rollout_{step}", "c_out", c.numpy(), ort_c, atol, rtol)
    return True


def _assert_close(case: str, name: str, expected, actual, atol: float, rtol: float) -> None:
    import numpy as np

    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(expected - actual)))
        raise RuntimeError(f"ONNX parity failed for {case}/{name}: max_abs={max_abs:.6g}, atol={atol}, rtol={rtol}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    manifest_path = output_dir / args.manifest_name
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    tmp_manifest_path = manifest_path.with_name(manifest_path.name + ".tmp")

    for tmp_path in (tmp_output_path, tmp_manifest_path):
        if tmp_path.exists():
            tmp_path.unlink()

    policy = SceneTokenSruCheckpointPolicy(checkpoint_path, device="cpu")
    actor_critic = policy.actor_critic.eval().cpu()
    exporter = SceneTokenSruOnnxActor(actor_critic).eval().cpu()

    obs_dim = policy.cfg.num_actor_obs
    hidden_size = policy.cfg.rnn_hidden_size
    num_layers = policy.cfg.rnn_num_layers
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_h = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
    dummy_c = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        pt_actions, pt_h, pt_c = exporter(dummy_obs, dummy_h, dummy_c)

    try:
        torch.onnx.export(
            exporter,
            (dummy_obs, dummy_h, dummy_c),
            str(tmp_output_path),
            input_names=["obs", "h_in", "c_in"],
            output_names=["actions", "h_out", "c_out"],
            dynamic_axes={
                "obs": {0: "batch"},
                "h_in": {1: "batch"},
                "c_in": {1: "batch"},
                "actions": {0: "batch"},
                "h_out": {1: "batch"},
                "c_out": {1: "batch"},
            },
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

        manifest = {
            "format_version": 1,
            "policy_onnx": output_path.name,
            "policy_onnx_sha256": _sha256(tmp_output_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_iteration": int(policy.checkpoint_iteration),
            "obs_dim": int(obs_dim),
            "num_actions": 3,
            "rnn_num_layers": int(num_layers),
            "rnn_hidden_size": int(hidden_size),
            "scene_token_dim": int(getattr(actor_critic, "num_actor_scene_token_features", 0)),
            "encoded_depth_dim": int(getattr(actor_critic, "num_image_features", 0)),
            "proprio_dim": int(getattr(actor_critic, "actor_proprioceptive_input_dim", 0)),
            "image_input_dims": [int(v) for v in getattr(actor_critic, "image_input_dims", [])],
            "opset": int(args.opset),
        }
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        parity_ok = _run_onnx_parity_check(
            exporter=exporter,
            onnx_path=tmp_output_path,
            obs_dim=obs_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            atol=float(args.parity_atol),
            rtol=float(args.parity_rtol),
            require=bool(args.require_onnx_parity),
        )

        tmp_output_path.replace(output_path)
        tmp_manifest_path.replace(manifest_path)
    except Exception:
        for tmp_path in (tmp_output_path, tmp_manifest_path):
            if tmp_path.exists():
                tmp_path.unlink()
        raise

    print(f"[INFO] Exported high-level SRU ONNX: {output_path}")
    print(f"[INFO] Exported deploy manifest: {manifest_path}")
    print(f"[INFO] ONNX parity check: {'passed' if parity_ok else 'not run'}")
    print(f"[INFO] checkpoint iteration: {policy.checkpoint_iteration}")
    print(f"[INFO] obs shape: (1, {obs_dim})")
    print(f"[INFO] h/c shape: ({num_layers}, 1, {hidden_size})")
    print(f"[INFO] sample actions: {pt_actions.detach().cpu().numpy().reshape(-1).tolist()}")
    print(f"[INFO] sample h_norm={float(pt_h.norm()):.6f}, c_norm={float(pt_c.norm()):.6f}")

if __name__ == "__main__":
    main()
