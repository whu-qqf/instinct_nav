from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch


def _ensure_local_rsl_rl_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rsl_rl_source = repo_root / "source" / "rsl_rl"
    if rsl_rl_source.exists():
        source_str = str(rsl_rl_source)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)


@dataclass(frozen=True)
class SceneTokenSruPolicyConfig:
    """Network shape for the current shared-lowcam scene-token SRU checkpoint."""

    num_actor_obs: int = 16 + 64 * 5 * 8 + 64 * 6
    num_critic_obs: int = 16 + 64 * 7 * 7 + 64 * 5 * 8 + 64 * 6 + 1
    num_actions: int = 3
    actor_hidden_dims: tuple[int, int, int] = (512, 256, 128)
    critic_hidden_dims: tuple[int, int, int] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    image_input_dims: tuple[int, int, int] = (64, 5, 8)
    height_input_dims: tuple[int, int, int] = (64, 7, 7)
    rnn_type: str = "lstm_sru"
    dropout: float = 0.2
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    time_embed_dim: int = 8
    num_cameras: int = 1
    use_actor_visibility_map: bool = False
    use_actor_visibility_early_fusion: bool = False
    use_actor_visibility_query_attention: bool = False
    use_actor_scene_tokens: bool = True
    use_critic_scene_tokens: bool = True
    scene_token_num_tokens: int = 64
    scene_token_dim: int = 6
    scene_token_feature_dim: int = 64
    scene_token_query_num_heads: int = 4
    use_dynamic_map_cross_attention: bool = False
    use_dynamic_scene_recurrent_fusion: bool = False
    dynamic_scene_recurrent_feature_dim: int = 64


class SceneTokenSruCheckpointPolicy(torch.nn.Module):
    """Direct Python runtime for trained scene-token ActorCriticSRU checkpoints.

    The current scene-token policy cannot be exported with the old SRU JIT helper,
    because that helper only knows the legacy depth-only actor observation layout.
    This wrapper keeps the original training module in Python and calls
    ``act_inference`` with persistent SRU hidden state.
    """

    def __init__(self, checkpoint_path: str | Path, cfg: SceneTokenSruPolicyConfig | None = None, device: str = "cpu"):
        super().__init__()
        _ensure_local_rsl_rl_on_path()
        from rsl_rl.modules.actor_critic_sru import ActorCriticSRU

        checkpoint_path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if cfg is None:
            cfg = SceneTokenSruPolicyConfig()
            if any("history_map_projector" in key for key in state_dict):
                current_weight = state_dict.get("current_map_projector.0.weight")
                if current_weight is None:
                    raise RuntimeError("DynamicMapCross checkpoint is missing current_map_projector.0.weight")
                scene_token_dim = int(current_weight.shape[1]) + 1
                recurrent_fusion = any("dynamic_scene_recurrent_projector" in key for key in state_dict)
                recurrent_feature_dim = 64
                if recurrent_fusion:
                    recurrent_weight = state_dict.get("dynamic_scene_recurrent_projector.0.weight")
                    if recurrent_weight is None:
                        raise RuntimeError(
                            "Recurrent-fusion checkpoint is missing dynamic_scene_recurrent_projector.0.weight"
                        )
                    recurrent_feature_dim = int(recurrent_weight.shape[0])
                cfg = replace(
                    cfg,
                    num_actor_obs=16 + 64 * 5 * 8 + 64 * scene_token_dim,
                    num_critic_obs=16 + 64 * 7 * 7 + 64 * 5 * 8 + 64 * scene_token_dim + 1,
                    scene_token_dim=scene_token_dim,
                    use_dynamic_map_cross_attention=True,
                    use_dynamic_scene_recurrent_fusion=recurrent_fusion,
                    dynamic_scene_recurrent_feature_dim=recurrent_feature_dim,
                )
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor_critic = ActorCriticSRU(
            num_actor_obs=self.cfg.num_actor_obs,
            num_critic_obs=self.cfg.num_critic_obs,
            num_actions=self.cfg.num_actions,
            actor_hidden_dims=list(self.cfg.actor_hidden_dims),
            critic_hidden_dims=list(self.cfg.critic_hidden_dims),
            activation=self.cfg.activation,
            init_noise_std=self.cfg.init_noise_std,
            image_input_dims=self.cfg.image_input_dims,
            height_input_dims=self.cfg.height_input_dims,
            rnn_type=self.cfg.rnn_type,
            dropout=self.cfg.dropout,
            rnn_hidden_size=self.cfg.rnn_hidden_size,
            rnn_num_layers=self.cfg.rnn_num_layers,
            time_embed_dim=self.cfg.time_embed_dim,
            num_cameras=self.cfg.num_cameras,
            use_actor_visibility_map=self.cfg.use_actor_visibility_map,
            use_actor_visibility_early_fusion=self.cfg.use_actor_visibility_early_fusion,
            use_actor_visibility_query_attention=self.cfg.use_actor_visibility_query_attention,
            use_actor_scene_tokens=self.cfg.use_actor_scene_tokens,
            use_critic_scene_tokens=self.cfg.use_critic_scene_tokens,
            scene_token_num_tokens=self.cfg.scene_token_num_tokens,
            scene_token_dim=self.cfg.scene_token_dim,
            scene_token_feature_dim=self.cfg.scene_token_feature_dim,
            scene_token_query_num_heads=self.cfg.scene_token_query_num_heads,
            use_dynamic_map_cross_attention=self.cfg.use_dynamic_map_cross_attention,
            use_dynamic_scene_recurrent_fusion=self.cfg.use_dynamic_scene_recurrent_fusion,
            dynamic_scene_recurrent_feature_dim=self.cfg.dynamic_scene_recurrent_feature_dim,
        )
        missing, unexpected = self.actor_critic.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"SRU checkpoint architecture mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.actor_critic.to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_iteration = checkpoint.get("iter")

    def reset(self) -> None:
        self.actor_critic.reset()
        self.actor_critic.reset_dropout_masks()

    def forward(self, obs: torch.Tensor, reset: bool = False) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.cfg.num_actor_obs:
            raise ValueError(f"Expected actor obs dim {self.cfg.num_actor_obs}, got {obs.shape[-1]}")
        if reset:
            self.reset()
        with torch.no_grad():
            return self.actor_critic.act_inference(obs)
