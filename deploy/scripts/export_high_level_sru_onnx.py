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
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

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

    torch.onnx.export(
        exporter,
        (dummy_obs, dummy_h, dummy_c),
        str(output_path),
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

    print(f"[INFO] Exported high-level SRU ONNX: {output_path}")
    print(f"[INFO] checkpoint iteration: {policy.checkpoint_iteration}")
    print(f"[INFO] obs shape: (1, {obs_dim})")
    print(f"[INFO] h/c shape: ({num_layers}, 1, {hidden_size})")
    print(f"[INFO] sample actions: {pt_actions.detach().cpu().numpy().reshape(-1).tolist()}")
    print(f"[INFO] sample h_norm={float(pt_h.norm()):.6f}, c_norm={float(pt_c.norm()):.6f}")


if __name__ == "__main__":
    main()
