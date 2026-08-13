#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Created by Fan Yang, Per Frivik, Robotic Systems Lab, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic module with SRU (Structured Recurrent Unit) memory and attention.

Architecture follows the paper: Self-attention → Cross-attention → SRU → MLP with TC-Dropout.
Attention is applied in batched mode (outside RNN loop) for efficiency.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from rsl_rl.networks.sru_memory import LSTM_SRU, CrossAttentionFuseModule
from rsl_rl.utils import unpad_trajectories


class ActorCriticSRU(nn.Module):
    """Actor-Critic network with SRU memory and cross-attention for image processing.

    This module is designed for visual navigation tasks where the actor receives
    image observations and the critic receives both image and height map observations.

    Args:
        num_actor_obs: Total dimension of actor observations.
        num_critic_obs: Total dimension of critic observations.
        num_actions: Number of action dimensions.
        actor_hidden_dims: List of hidden layer dimensions for actor MLP.
        critic_hidden_dims: List of hidden layer dimensions for critic MLP.
        activation: Activation function name.
        init_noise_std: Initial standard deviation for action noise.
        image_input_dims: Tuple of (C, H, W) for image input.
        height_input_dims: Tuple of (C, H, W) for height map input.
        rnn_type: Type of RNN (currently only 'lstm' supported).
        dropout: Dropout probability.
        rnn_hidden_size: Hidden size of the RNN.
        rnn_num_layers: Number of RNN layers.
        time_embed_dim: Dimension of time embedding for critic.
        num_cameras: Number of cameras (1 or 2).
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Optional[list[int]] = None,
        critic_hidden_dims: Optional[list[int]] = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        image_input_dims: tuple[int, int, int] = (64, 5, 8),
        height_input_dims: tuple[int, int, int] = (64, 7, 7),
        rnn_type: str = "lstm_sru",
        dropout: float = 0.2,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        time_embed_dim: int = 8,
        num_cameras: int = 1,
        use_actor_visibility_map: bool = False,
        use_actor_visibility_early_fusion: bool = False,
        visibility_map_input_dims: tuple[int, int, int] = (1, 64, 64),
        visibility_map_feature_dim: int = 64,
        use_actor_visibility_query_attention: bool = False,
        visibility_map_query_num_heads: int = 4,
        use_actor_scene_tokens: bool = False,
        use_critic_scene_tokens: bool = False,
        scene_token_num_tokens: int = 64,
        scene_token_dim: int = 6,
        scene_token_feature_dim: int = 64,
        scene_token_query_num_heads: int = 4,
        use_dynamic_map_cross_attention: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticSRU] Warning: got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        if rnn_type != "lstm_sru":
            print(f"[ActorCriticSRU] Warning: rnn_type='{rnn_type}' is ignored. ActorCriticSRU always uses LSTM_SRU.")
        super().__init__()

        # Handle mutable default arguments
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]

        self.image_input_dims = image_input_dims
        self.height_input_dims = height_input_dims
        self.num_cameras = num_cameras
        self.use_actor_visibility_map = bool(use_actor_visibility_map)
        self.use_actor_visibility_early_fusion = (
            bool(use_actor_visibility_early_fusion) and self.use_actor_visibility_map
        )
        if self.use_actor_visibility_early_fusion and use_actor_visibility_query_attention:
            print(
                "[ActorCriticSRU] Warning: early fusion and query attention are mutually exclusive; "
                "using early fusion."
            )
        self.use_actor_visibility_query_attention = (
            bool(use_actor_visibility_query_attention)
            and self.use_actor_visibility_map
            and not self.use_actor_visibility_early_fusion
        )
        self.visibility_map_input_dims = visibility_map_input_dims
        self.visibility_map_feature_dim = int(visibility_map_feature_dim) if self.use_actor_visibility_map else 0
        self.visibility_map_query_num_heads = int(visibility_map_query_num_heads)
        self.use_actor_scene_tokens = bool(use_actor_scene_tokens)
        self.use_critic_scene_tokens = use_critic_scene_tokens
        if self.use_actor_scene_tokens and self.use_actor_visibility_map:
            raise ValueError("Actor scene tokens and actor visibility map are mutually exclusive in this implementation")
        self.scene_token_num_tokens = int(scene_token_num_tokens)
        self.scene_token_dim = int(scene_token_dim)
        self.scene_token_feature_dim = int(scene_token_feature_dim) if (self.use_actor_scene_tokens or self.use_critic_scene_tokens) else 0
        self.scene_token_query_num_heads = int(scene_token_query_num_heads)
        self.use_dynamic_map_cross_attention = bool(use_dynamic_map_cross_attention)
        if self.use_dynamic_map_cross_attention and self.scene_token_dim != 5:
            raise ValueError("dynamic map cross-attention expects scene_token_dim=5")

        # Compute the total number of features from image, height, visibility, and scene-token inputs.
        self.num_image_features = image_input_dims[0] * image_input_dims[1] * image_input_dims[2]
        self.num_height_features = height_input_dims[0] * height_input_dims[1] * height_input_dims[2]
        self.num_visibility_map_features = (
            visibility_map_input_dims[0] * visibility_map_input_dims[1] * visibility_map_input_dims[2]
            if self.use_actor_visibility_map
            else 0
        )
        self.num_scene_token_features = (
            self.scene_token_num_tokens * self.scene_token_dim if (self.use_actor_scene_tokens or self.use_critic_scene_tokens) else 0
        )
        self.num_actor_scene_token_features = self.num_scene_token_features if self.use_actor_scene_tokens else 0
        self.num_critic_scene_token_features = self.num_scene_token_features if self.use_critic_scene_tokens else 0

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # For actor: proprioceptive data + image data + optional visibility map data
        # For critic: proprioceptive data + height data + image data + time data
        self.actor_proprioceptive_input_dim = (
            num_actor_obs
            - self.num_image_features * self.num_cameras
            - self.num_visibility_map_features
            - self.num_actor_scene_token_features
        )
        self.critic_proprioceptive_input_dim = (
            num_critic_obs
            - self.num_height_features
            - self.num_image_features * self.num_cameras
            - self.num_critic_scene_token_features
            - 1
        )

        assert self.actor_proprioceptive_input_dim == self.critic_proprioceptive_input_dim, (
            "Actor and Critic proprioceptive input dims must match"
        )

        # Baseline actor stream is proprioception+depth. Early fusion appends
        # visibility-map features before the SRU; late/query fusion append them
        # after it so explicit spatial memory stays out of the recurrent state.
        self.mlp_input_dim_actor = self.actor_proprioceptive_input_dim + image_input_dims[0]
        if self.use_actor_visibility_early_fusion:
            self.mlp_input_dim_actor += self.visibility_map_feature_dim
            self.actor_post_recurrent_input_dim = rnn_hidden_size
        else:
            self.actor_post_recurrent_input_dim = rnn_hidden_size + self.visibility_map_feature_dim
        if self.use_actor_scene_tokens:
            self.actor_post_recurrent_input_dim += self.scene_token_feature_dim
        self.mlp_input_dim_critic = (
            self.critic_proprioceptive_input_dim + height_input_dims[0] + image_input_dims[0]
        )
        self.critic_post_recurrent_input_dim = rnn_hidden_size + time_embed_dim
        if self.use_critic_scene_tokens:
            self.critic_post_recurrent_input_dim += self.scene_token_feature_dim

        # Attention modules (applied in batch, outside RNN loop)
        # spatial_dims = (D, H, W) where D=num_cameras for image, D=1 for height
        self.attn_image_net = CrossAttentionFuseModule(
            image_dim=image_input_dims[0],
            info_dim=self.actor_proprioceptive_input_dim,
            num_heads=4,
            spatial_dims=(num_cameras, image_input_dims[1], image_input_dims[2]),
        )
        self.attn_height_net = CrossAttentionFuseModule(
            image_dim=height_input_dims[0],
            info_dim=self.critic_proprioceptive_input_dim,
            num_heads=4,
            spatial_dims=(1, height_input_dims[1], height_input_dims[2]),
        )
        self.attn_critic_image_net = CrossAttentionFuseModule(
            image_dim=image_input_dims[0],
            info_dim=self.critic_proprioceptive_input_dim,
            num_heads=4,
            spatial_dims=(num_cameras, image_input_dims[1], image_input_dims[2]),
        )
        if self.use_actor_visibility_map:
            if self.use_actor_visibility_query_attention:
                assert self.visibility_map_feature_dim % self.visibility_map_query_num_heads == 0, (
                    "visibility_map_feature_dim must be divisible by visibility_map_query_num_heads"
                )
                self.visibility_map_num_tokens = 8 * 8
                self.visibility_map_encoder = nn.Sequential(
                    nn.Conv2d(visibility_map_input_dims[0], 16, kernel_size=5, stride=2, padding=2),
                    get_activation(activation),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    get_activation(activation),
                    nn.Conv2d(32, self.visibility_map_feature_dim, kernel_size=3, stride=2, padding=1),
                    get_activation(activation),
                    nn.AdaptiveAvgPool2d((8, 8)),
                )
                self.visibility_map_position_embedding = nn.Parameter(
                    torch.zeros(1, self.visibility_map_num_tokens, self.visibility_map_feature_dim)
                )
                self.visibility_query_proj = nn.Sequential(
                    nn.Linear(rnn_hidden_size + self.actor_proprioceptive_input_dim, self.visibility_map_feature_dim),
                    get_activation(activation),
                    nn.Linear(self.visibility_map_feature_dim, self.visibility_map_feature_dim),
                )
                self.visibility_token_norm = nn.LayerNorm(self.visibility_map_feature_dim)
                self.visibility_query_norm = nn.LayerNorm(self.visibility_map_feature_dim)
                self.visibility_cross_attention = nn.MultiheadAttention(
                    embed_dim=self.visibility_map_feature_dim,
                    num_heads=self.visibility_map_query_num_heads,
                    batch_first=True,
                )
                self.visibility_attention_ffn = nn.Sequential(
                    nn.LayerNorm(self.visibility_map_feature_dim),
                    nn.Linear(self.visibility_map_feature_dim, self.visibility_map_feature_dim * 2),
                    get_activation(activation),
                    nn.Linear(self.visibility_map_feature_dim * 2, self.visibility_map_feature_dim),
                )
            else:
                self.visibility_map_num_tokens = 0
                self.visibility_query_proj = None
                self.visibility_token_norm = None
                self.visibility_query_norm = None
                self.visibility_cross_attention = None
                self.visibility_attention_ffn = None
                self.visibility_map_position_embedding = None
                activation_layer = get_activation(activation)
                self.visibility_map_encoder = nn.Sequential(
                    nn.Conv2d(visibility_map_input_dims[0], 16, kernel_size=5, stride=2, padding=2),
                    activation_layer,
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    get_activation(activation),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                    get_activation(activation),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, self.visibility_map_feature_dim),
                    get_activation(activation),
                )
        else:
            self.visibility_map_encoder = None
            self.visibility_map_num_tokens = 0
            self.visibility_query_proj = None
            self.visibility_token_norm = None
            self.visibility_query_norm = None
            self.visibility_cross_attention = None
            self.visibility_attention_ffn = None
            self.visibility_map_position_embedding = None

        if self.use_actor_scene_tokens or self.use_critic_scene_tokens:
            if self.scene_token_feature_dim % self.scene_token_query_num_heads != 0:
                raise ValueError("scene_token_feature_dim must be divisible by scene_token_query_num_heads")
            self.scene_token_projector = nn.Sequential(
                nn.Linear(self.scene_token_dim, self.scene_token_feature_dim),
                get_activation(activation),
                nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim),
            )
            self.scene_token_position_embedding = nn.Parameter(
                torch.zeros(1, self.scene_token_num_tokens, self.scene_token_feature_dim)
            )
            self.scene_token_norm = nn.LayerNorm(self.scene_token_feature_dim)
            if self.use_dynamic_map_cross_attention:
                self.history_map_projector = nn.Sequential(
                    nn.Linear(1, self.scene_token_feature_dim), get_activation(activation),
                    nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim),
                )
                self.current_map_projector = nn.Sequential(
                    nn.Linear(4, self.scene_token_feature_dim), get_activation(activation),
                    nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim),
                )
                self.map_cross_attention = nn.MultiheadAttention(
                    self.scene_token_feature_dim, self.scene_token_query_num_heads, batch_first=True
                )
                self.map_cross_norm = nn.LayerNorm(self.scene_token_feature_dim)
                self.map_cross_ffn = nn.Sequential(
                    nn.LayerNorm(self.scene_token_feature_dim),
                    nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim * 2),
                    get_activation(activation),
                    nn.Linear(self.scene_token_feature_dim * 2, self.scene_token_feature_dim),
                )
            else:
                self.history_map_projector = self.current_map_projector = None
                self.map_cross_attention = self.map_cross_norm = self.map_cross_ffn = None
        else:
            self.scene_token_projector = None
            self.scene_token_position_embedding = None
            self.scene_token_norm = None
            self.history_map_projector = self.current_map_projector = None
            self.map_cross_attention = self.map_cross_norm = self.map_cross_ffn = None

        if self.use_actor_scene_tokens:
            self.scene_query_proj = nn.Sequential(
                nn.Linear(rnn_hidden_size + self.actor_proprioceptive_input_dim, self.scene_token_feature_dim),
                get_activation(activation),
                nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim),
            )
            self.scene_query_norm = nn.LayerNorm(self.scene_token_feature_dim)
            self.scene_cross_attention = nn.MultiheadAttention(
                embed_dim=self.scene_token_feature_dim,
                num_heads=self.scene_token_query_num_heads,
                batch_first=True,
            )
            self.scene_attention_ffn = nn.Sequential(
                nn.LayerNorm(self.scene_token_feature_dim),
                nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim * 2),
                get_activation(activation),
                nn.Linear(self.scene_token_feature_dim * 2, self.scene_token_feature_dim),
            )
        else:
            self.scene_query_proj = None
            self.scene_query_norm = None
            self.scene_cross_attention = None
            self.scene_attention_ffn = None

        if self.use_critic_scene_tokens:
            self.critic_scene_query_proj = nn.Sequential(
                nn.Linear(rnn_hidden_size + self.critic_proprioceptive_input_dim, self.scene_token_feature_dim),
                get_activation(activation),
                nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim),
            )
            self.critic_scene_query_norm = nn.LayerNorm(self.scene_token_feature_dim)
            self.critic_scene_cross_attention = nn.MultiheadAttention(
                embed_dim=self.scene_token_feature_dim,
                num_heads=self.scene_token_query_num_heads,
                batch_first=True,
            )
            self.critic_scene_attention_ffn = nn.Sequential(
                nn.LayerNorm(self.scene_token_feature_dim),
                nn.Linear(self.scene_token_feature_dim, self.scene_token_feature_dim * 2),
                get_activation(activation),
                nn.Linear(self.scene_token_feature_dim * 2, self.scene_token_feature_dim),
            )
        else:
            self.critic_scene_query_proj = None
            self.critic_scene_query_norm = None
            self.critic_scene_cross_attention = None
            self.critic_scene_attention_ffn = None

        # RNN memory modules (plain LSTM_SRU without attention)
        self.memory_a = MemorySRU(
            input_size=self.mlp_input_dim_actor,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )
        self.memory_c = MemorySRU(
            input_size=self.mlp_input_dim_critic,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )

        # Time embedding layer for critic
        self.time_layer = nn.Linear(1, time_embed_dim)

        # Policy network (Actor)
        self.linear_dropout_actor = LinearConstDropout(
            self.actor_post_recurrent_input_dim, actor_hidden_dims[0], dropout_p=dropout, activation_name=activation
        )
        actor_layers = []
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(get_activation(activation))
        self.actor = nn.Sequential(*actor_layers)

        # Value network (Critic)
        self.linear_dropout_critic = LinearConstDropout(
            self.critic_post_recurrent_input_dim, critic_hidden_dims[0], dropout_p=dropout, activation_name=activation
        )
        critic_layers = []
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        print(f"[ActorCriticSRU] Actor MLP: {self.actor}")
        print(f"[ActorCriticSRU] Critic MLP: {self.critic}")
        print(f"[ActorCriticSRU] Actor RNN: {self.memory_a}")
        print(f"[ActorCriticSRU] Critic RNN: {self.memory_c}")
        print(f"[ActorCriticSRU] Num cameras: {self.num_cameras}")
        if self.use_actor_visibility_map:
            if self.use_actor_visibility_early_fusion:
                branch_name = "early fusion"
            elif self.use_actor_visibility_query_attention:
                branch_name = "query-attention late fusion"
            else:
                branch_name = "late fusion"
            print(
                f"[ActorCriticSRU] Actor visibility map branch: {branch_name}, "
                f"input={self.visibility_map_input_dims}, feature_dim={self.visibility_map_feature_dim}"
            )
        if self.use_actor_scene_tokens:
            print(
                f"[ActorCriticSRU] Actor scene-token branch: query-attention late fusion, "
                f"tokens={self.scene_token_num_tokens}, token_dim={self.scene_token_dim}, "
                f"feature_dim={self.scene_token_feature_dim}"
            )
            if self.use_dynamic_map_cross_attention:
                print("[ActorCriticSRU] Scene tokens learned by current-to-history map cross-attention")
        # Action noise: using log_std parameter
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        Distribution.set_default_validate_args(False)

    def get_actor_parameters(self):
        visibility_parameters = []
        if self.visibility_map_encoder is not None:
            visibility_parameters += list(self.visibility_map_encoder.parameters())
        if self.use_actor_visibility_query_attention:
            visibility_parameters += [self.visibility_map_position_embedding]
            visibility_parameters += list(self.visibility_query_proj.parameters())
            visibility_parameters += list(self.visibility_token_norm.parameters())
            visibility_parameters += list(self.visibility_query_norm.parameters())
            visibility_parameters += list(self.visibility_cross_attention.parameters())
            visibility_parameters += list(self.visibility_attention_ffn.parameters())
        scene_parameters = []
        if self.use_actor_scene_tokens or self.use_critic_scene_tokens:
            scene_parameters += list(self.scene_token_projector.parameters())
            scene_parameters += [self.scene_token_position_embedding]
            scene_parameters += list(self.scene_token_norm.parameters())
            if self.use_dynamic_map_cross_attention:
                scene_parameters += list(self.history_map_projector.parameters())
                scene_parameters += list(self.current_map_projector.parameters())
                scene_parameters += list(self.map_cross_attention.parameters())
                scene_parameters += list(self.map_cross_norm.parameters())
                scene_parameters += list(self.map_cross_ffn.parameters())
            if self.use_actor_scene_tokens:
                scene_parameters += list(self.scene_query_proj.parameters())
                scene_parameters += list(self.scene_query_norm.parameters())
                scene_parameters += list(self.scene_cross_attention.parameters())
                scene_parameters += list(self.scene_attention_ffn.parameters())
            if self.use_critic_scene_tokens:
                scene_parameters += list(self.critic_scene_query_proj.parameters())
                scene_parameters += list(self.critic_scene_query_norm.parameters())
                scene_parameters += list(self.critic_scene_cross_attention.parameters())
                scene_parameters += list(self.critic_scene_attention_ffn.parameters())
        return (
            list(self.attn_image_net.parameters())
            + visibility_parameters
            + scene_parameters
            + list(self.linear_dropout_actor.parameters())
            + list(self.actor.parameters())
            + list(self.memory_a.parameters())
            + [self.log_std]
        )

    def get_critic_parameters(self):
        return (
            list(self.attn_height_net.parameters())
            + list(self.attn_critic_image_net.parameters())
            + list(self.linear_dropout_critic.parameters())
            + list(self.time_layer.parameters())
            + list(self.critic.parameters())
            + list(self.memory_c.parameters())
        )

    @staticmethod
    def init_weights(sequential, scales):
        """Initialize all Linear layers in a sequential module using orthogonal initialization."""
        linear_modules = [module for module in sequential if isinstance(module, nn.Linear)]
        for idx, module in enumerate(linear_modules):
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])

    def reset(self, dones=None):
        """Reset the hidden states of both RNN memories."""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError("Forward method not implemented.")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _extract_image_observations(
        self, observations: torch.Tensor, trailing_features: int = 0
    ) -> list[torch.Tensor]:
        """Extract and reshape image observations based on number of cameras.

        Args:
            observations: Input observations tensor.
            trailing_features: Number of non-image features appended after the image block.

        Returns:
            List of reshaped image tensors.
        """
        image_end = None if trailing_features == 0 else -trailing_features
        if self.num_cameras == 2:
            image_start = -(self.num_image_features * 2 + trailing_features)
            image_mid = -(self.num_image_features + trailing_features)
            image_obs_front = observations[..., image_start:image_mid]
            image_obs_back = observations[..., image_mid:image_end]
            return [
                image_obs_front.reshape(-1, *self.image_input_dims),
                image_obs_back.reshape(-1, *self.image_input_dims),
            ]
        elif self.num_cameras == 1:
            image_start = -(self.num_image_features + trailing_features)
            image_obs_single = observations[..., image_start:image_end]
            return [image_obs_single.reshape(-1, *self.image_input_dims)]
        else:
            raise ValueError(f"Unsupported num_cameras: {self.num_cameras}")

    def _extract_actor_visibility_map(self, observations: torch.Tensor) -> torch.Tensor | None:
        if not self.use_actor_visibility_map:
            return None
        visibility_obs = observations[..., -self.num_visibility_map_features :]
        return visibility_obs.reshape(-1, *self.visibility_map_input_dims)

    def _extract_actor_scene_tokens(self, observations: torch.Tensor) -> torch.Tensor | None:
        if not self.use_actor_scene_tokens:
            return None
        scene_obs = observations[..., -self.num_actor_scene_token_features :]
        return scene_obs.reshape(-1, self.scene_token_num_tokens, self.scene_token_dim)

    def _extract_critic_scene_tokens(self, observations: torch.Tensor) -> torch.Tensor | None:
        if not self.use_critic_scene_tokens:
            return None
        scene_obs = observations[..., -self.num_critic_scene_token_features :]
        return scene_obs.reshape(-1, self.scene_token_num_tokens, self.scene_token_dim)

    @staticmethod
    def _match_recurrent_shape(features: torch.Tensor, recurrent_features: torch.Tensor) -> torch.Tensor:
        while features.dim() > recurrent_features.dim() and features.shape[0] == 1:
            features = features.squeeze(0)
        return features

    def _query_actor_visibility_map(
        self, visibility_tokens: torch.Tensor, recurrent_features: torch.Tensor, proprio_features: torch.Tensor
    ) -> torch.Tensor:
        leading_shape = recurrent_features.shape[:-1]
        flat_tokens = visibility_tokens.reshape(-1, self.visibility_map_num_tokens, self.visibility_map_feature_dim)
        flat_context = torch.cat([recurrent_features, proprio_features], dim=-1).reshape(
            -1, recurrent_features.shape[-1] + proprio_features.shape[-1]
        )

        tokens = flat_tokens + self.visibility_map_position_embedding.to(
            device=flat_tokens.device, dtype=flat_tokens.dtype
        )
        query = self.visibility_query_proj(flat_context).unsqueeze(1)
        query = self.visibility_query_norm(query)
        tokens = self.visibility_token_norm(tokens)
        attended, _ = self.visibility_cross_attention(query, tokens, tokens, need_weights=False)
        visibility_features = attended.squeeze(1)
        visibility_features = visibility_features + self.visibility_attention_ffn(visibility_features)
        return visibility_features.reshape(*leading_shape, self.visibility_map_feature_dim)

    def _query_actor_scene_tokens(
        self, scene_tokens: torch.Tensor, recurrent_features: torch.Tensor, proprio_features: torch.Tensor
    ) -> torch.Tensor:
        leading_shape = recurrent_features.shape[:-1]
        flat_tokens = scene_tokens.reshape(-1, self.scene_token_num_tokens, self.scene_token_dim)
        flat_context = torch.cat([recurrent_features, proprio_features], dim=-1).reshape(
            -1, recurrent_features.shape[-1] + proprio_features.shape[-1]
        )

        tokens = self._encode_scene_tokens(flat_tokens)
        tokens = tokens + self.scene_token_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
        query = self.scene_query_proj(flat_context).unsqueeze(1)
        query = self.scene_query_norm(query)
        tokens = self.scene_token_norm(tokens)
        attended, _ = self.scene_cross_attention(query, tokens, tokens, need_weights=False)
        scene_features = attended.squeeze(1)
        scene_features = scene_features + self.scene_attention_ffn(scene_features)
        return scene_features.reshape(*leading_shape, self.scene_token_feature_dim)

    def _query_critic_scene_tokens(
        self, scene_tokens: torch.Tensor, recurrent_features: torch.Tensor, proprio_features: torch.Tensor
    ) -> torch.Tensor:
        leading_shape = recurrent_features.shape[:-1]
        flat_tokens = scene_tokens.reshape(-1, self.scene_token_num_tokens, self.scene_token_dim)
        flat_context = torch.cat([recurrent_features, proprio_features], dim=-1).reshape(
            -1, recurrent_features.shape[-1] + proprio_features.shape[-1]
        )

        tokens = self._encode_scene_tokens(flat_tokens)
        tokens = tokens + self.scene_token_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
        query = self.critic_scene_query_proj(flat_context).unsqueeze(1)
        query = self.critic_scene_query_norm(query)
        tokens = self.scene_token_norm(tokens)
        attended, _ = self.critic_scene_cross_attention(query, tokens, tokens, need_weights=False)
        scene_features = attended.squeeze(1)
        scene_features = scene_features + self.critic_scene_attention_ffn(scene_features)
        return scene_features.reshape(*leading_shape, self.scene_token_feature_dim)

    def _encode_scene_tokens(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamic_map_cross_attention:
            return self.scene_token_projector(flat_tokens)
        position = self.scene_token_position_embedding.to(device=flat_tokens.device, dtype=flat_tokens.dtype)
        history = self.history_map_projector(flat_tokens[..., :1]) + position
        current = self.current_map_projector(flat_tokens[..., 1:]) + position
        attended, _ = self.map_cross_attention(current, history, history, need_weights=False)
        learned = self.map_cross_norm(current + attended)
        return learned + self.map_cross_ffn(learned)

    def process_actor_input(self, observations: torch.Tensor, masks, hidden_states) -> torch.Tensor:
        """Process actor observations through attention and RNN.

        Architecture: Self-attention → Cross-attention → SRU → MLP
        Attention is applied in batched mode (all timesteps at once) for efficiency.

        Args:
            observations: Actor observations tensor.
            masks: Mask tensor for batch mode (None for inference).
            hidden_states: Hidden states from previous step.

        Returns:
            Combined features after RNN processing.
        """
        batch_mode = masks is not None

        # Split observations into proprioceptive, image, and optional visibility-map parts.
        visual_tail_features = (
            self.num_image_features * self.num_cameras
            + self.num_visibility_map_features
            + self.num_actor_scene_token_features
        )
        other_obs = observations[..., : -visual_tail_features]
        image_list = self._extract_image_observations(
            observations, trailing_features=self.num_visibility_map_features + self.num_actor_scene_token_features
        )
        visibility_obs = self._extract_actor_visibility_map(observations)
        scene_tokens = self._extract_actor_scene_tokens(observations)
        other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)

        # Stack images if multiple cameras, otherwise use single image
        if self.num_cameras == 2:
            image_input = image_list  # List of tensors for multi-camera
        else:
            image_input = image_list[0]  # Single tensor

        # Batched attention (process all timesteps at once)
        image_features = self.attn_image_net(image_input, other_obs)
        visibility_features = None
        visibility_tokens = None
        if visibility_obs is not None:
            visibility_encoded = self.visibility_map_encoder(visibility_obs)
            if self.use_actor_visibility_query_attention:
                visibility_tokens = visibility_encoded.flatten(2).transpose(1, 2)
            else:
                visibility_features = visibility_encoded

        # Reshape for batch mode
        if batch_mode:
            seq_len, batch_size, _ = observations.size()
            image_features = image_features.view(seq_len, batch_size, -1)
            if visibility_features is not None:
                visibility_features = visibility_features.view(seq_len, batch_size, -1)
            if visibility_tokens is not None:
                visibility_tokens = visibility_tokens.reshape(seq_len, batch_size, -1)
            if scene_tokens is not None:
                scene_tokens = scene_tokens.reshape(seq_len, batch_size, -1)
            other_obs = other_obs.view(seq_len, batch_size, -1)

        # The recurrent actor state is based on the baseline observation stream,
        # except in early fusion where visibility features are recurrent input.
        if self.use_actor_visibility_early_fusion and visibility_features is not None:
            combined_features = torch.cat([image_features, other_obs, visibility_features], dim=-1)
            visibility_features = None
        else:
            combined_features = torch.cat([image_features, other_obs], dim=-1)

        # RNN processing
        combined_features = self.memory_a(combined_features, masks, hidden_states)
        combined_features = combined_features.squeeze(0)

        if visibility_features is not None:
            if batch_mode:
                visibility_features = unpad_trajectories(visibility_features, masks)
                visibility_features = self._match_recurrent_shape(visibility_features, combined_features)
            combined_features = torch.cat([combined_features, visibility_features], dim=-1)
        elif visibility_tokens is not None:
            proprio_features = other_obs
            if batch_mode:
                proprio_features = unpad_trajectories(proprio_features, masks)
                proprio_features = self._match_recurrent_shape(proprio_features, combined_features)
                visibility_tokens = unpad_trajectories(visibility_tokens, masks)
                visibility_tokens = self._match_recurrent_shape(visibility_tokens, combined_features)
            visibility_tokens = visibility_tokens.reshape(
                *combined_features.shape[:-1], self.visibility_map_num_tokens, self.visibility_map_feature_dim
            )
            visibility_features = self._query_actor_visibility_map(
                visibility_tokens, combined_features, proprio_features
            )
            combined_features = torch.cat([combined_features, visibility_features], dim=-1)

        if scene_tokens is not None:
            proprio_features = other_obs
            if batch_mode:
                proprio_features = unpad_trajectories(proprio_features, masks)
                proprio_features = self._match_recurrent_shape(proprio_features, combined_features)
                scene_tokens = unpad_trajectories(scene_tokens, masks)
                scene_tokens = self._match_recurrent_shape(scene_tokens, combined_features)
            scene_tokens = scene_tokens.reshape(
                *combined_features.shape[:-1], self.scene_token_num_tokens, self.scene_token_dim
            )
            scene_features = self._query_actor_scene_tokens(scene_tokens, combined_features, proprio_features)
            combined_features = torch.cat([combined_features, scene_features], dim=-1)

        return combined_features

    def process_critic_input(self, observations: torch.Tensor, masks, hidden_states) -> torch.Tensor:
        """Process critic observations through attention and RNN.

        Architecture: Self-attention → Cross-attention → SRU → MLP
        Attention is applied in batched mode (all timesteps at once) for efficiency.

        Args:
            observations: Critic observations tensor.
            masks: Mask tensor for batch mode (None for inference).
            hidden_states: Hidden states from previous step.

        Returns:
            Combined features after RNN processing with time embedding.
        """
        batch_mode = masks is not None

        # Critic observation layout is: proprio | time | height | depth | optional scene_tokens.
        scene_tokens = self._extract_critic_scene_tokens(observations)
        trailing_features = self.num_critic_scene_token_features
        depth_start_from_end = self.num_image_features * self.num_cameras + trailing_features
        height_start_from_end = self.num_height_features + depth_start_from_end
        time_index_from_end = height_start_from_end + 1

        other_obs = observations[..., :-time_index_from_end]
        time_obs = observations[..., -time_index_from_end].unsqueeze(-1)
        height_obs = observations[..., -height_start_from_end:-depth_start_from_end]
        image_list = self._extract_image_observations(observations, trailing_features=trailing_features)

        # Reshape tensors
        other_obs = other_obs.reshape(-1, self.critic_proprioceptive_input_dim)
        height_obs = height_obs.reshape(-1, *self.height_input_dims)

        # Stack images if multiple cameras, otherwise use single image
        if self.num_cameras == 2:
            image_input = image_list  # List of tensors for multi-camera
        else:
            image_input = image_list[0]  # Single tensor

        # Batched attention (process all timesteps at once)
        height_features = self.attn_height_net(height_obs, other_obs)
        image_features = self.attn_critic_image_net(image_input, other_obs)

        # Reshape for batch mode
        if batch_mode:
            seq_len, batch_size, _ = observations.size()
            height_features = height_features.view(seq_len, batch_size, -1)
            image_features = image_features.view(seq_len, batch_size, -1)
            other_obs = other_obs.view(seq_len, batch_size, -1)
            if scene_tokens is not None:
                scene_tokens = scene_tokens.reshape(seq_len, batch_size, -1)

        # Time embedding for the critic
        time_embed = self.time_layer(time_obs)

        # Concatenate height, image, and proprioceptive features
        combined_features = torch.cat([height_features, image_features, other_obs], dim=-1)

        # RNN processing
        combined_features = self.memory_c(combined_features, masks, hidden_states).squeeze(0)

        if batch_mode:
            time_embed = unpad_trajectories(time_embed, masks)
            time_embed = self._match_recurrent_shape(time_embed, combined_features)

        if scene_tokens is not None:
            proprio_features = other_obs
            if batch_mode:
                proprio_features = unpad_trajectories(proprio_features, masks)
                proprio_features = self._match_recurrent_shape(proprio_features, combined_features)
                scene_tokens = unpad_trajectories(scene_tokens, masks)
                scene_tokens = self._match_recurrent_shape(scene_tokens, combined_features)
            scene_tokens = scene_tokens.reshape(
                *combined_features.shape[:-1], self.scene_token_num_tokens, self.scene_token_dim
            )
            scene_features = self._query_critic_scene_tokens(scene_tokens, combined_features, proprio_features)
            return torch.cat([combined_features, scene_features, time_embed], dim=-1)

        return torch.cat([combined_features, time_embed], dim=-1)

    def update_distribution(self, combined_features):
        """Update the action distribution based on actor output."""
        mean = self.actor(combined_features)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        """Sample actions from the policy."""
        combined_features = self.process_actor_input(observations, masks, hidden_states)
        combined_features = self.linear_dropout_actor(combined_features, dropout_masks)
        self.update_distribution(combined_features)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        """Get deterministic actions for inference."""
        combined_features = self.process_actor_input(observations, masks, hidden_states)
        combined_features = self.linear_dropout_actor(combined_features, dropout_masks)
        actions_mean = self.actor(combined_features)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        """Evaluate state values."""
        combined_features = self.process_critic_input(critic_observations, masks, hidden_states)
        combined_features = self.linear_dropout_critic(combined_features, dropout_masks)
        value = self.critic(combined_features)
        return value

    def get_hidden_states(self):
        """Get current hidden states from actor and critic memories."""
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_dropout_masks(self):
        """Get current dropout masks."""
        return self.linear_dropout_actor.get_dropout_mask(), self.linear_dropout_critic.get_dropout_mask()

    def reset_dropout_masks(self):
        """Reset dropout masks for new episode."""
        self.linear_dropout_actor.reset_dropout_mask()
        self.linear_dropout_critic.reset_dropout_mask()

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        """Export policy as a JIT-scripted module for inference.

        The exported module includes the complete actor pipeline:
        CrossAttention → MemorySRU → LinearDropout → Actor MLP

        Uses specialized exporters for single vs dual camera configurations
        to avoid runtime branching and .item() calls for optimal performance.

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.pt".
            normalizer: Optional normalizer module for observations.
        """
        import os

        # Select optimized exporter based on camera count
        if self.num_cameras == 2:
            exporter = _ActorCriticSRUExporterDualCam(
                attn_image_net=self.attn_image_net,
                memory_a=self.memory_a,
                linear_dropout_actor=self.linear_dropout_actor,
                actor=self.actor,
                image_input_dims=self.image_input_dims,
                num_image_features=self.num_image_features,
                actor_proprioceptive_input_dim=self.actor_proprioceptive_input_dim,
                normalizer=normalizer,
            )
        else:
            exporter = _ActorCriticSRUExporterSingleCam(
                attn_image_net=self.attn_image_net,
                memory_a=self.memory_a,
                linear_dropout_actor=self.linear_dropout_actor,
                actor=self.actor,
                image_input_dims=self.image_input_dims,
                num_image_features=self.num_image_features,
                actor_proprioceptive_input_dim=self.actor_proprioceptive_input_dim,
                normalizer=normalizer,
            )

        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        scripted = torch.jit.script(exporter)
        scripted.save(filepath)
        print(f"[ActorCriticSRU] Exported JIT policy ({self.num_cameras} camera) to: {filepath}")

    def export_onnx(self, path: str, filename: str = "policy.onnx", normalizer=None):
        """Export policy as an ONNX model for inference.

        The exported module includes the complete actor pipeline:
        CrossAttention → MemorySRU → LinearDropout → Actor MLP

        For recurrent networks, hidden states are exposed as explicit inputs and outputs
        since ONNX doesn't support stateful ops.

        Uses specialized exporters for single vs dual camera configurations.

        The ONNX model has the following interface:
            Inputs:
                - obs: Observation tensor of shape (1, num_actor_obs)
                - h_in: Hidden state tensor of shape (num_layers, 1, hidden_size)
                - c_in: Cell state tensor of shape (num_layers, 1, hidden_size)
            Outputs:
                - actions: Action tensor of shape (1, num_actions)
                - h_out: Updated hidden state
                - c_out: Updated cell state

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.onnx".
            normalizer: Optional normalizer module for observations.
        """
        import os

        rnn = self.memory_a.rnn

        # Select optimized exporter based on camera count
        if self.num_cameras == 2:
            exporter = _ActorCriticSRUONNXExporterDualCam(
                attn_image_net=self.attn_image_net,
                memory_a=self.memory_a,
                linear_dropout_actor=self.linear_dropout_actor,
                actor=self.actor,
                image_input_dims=self.image_input_dims,
                num_image_features=self.num_image_features,
                actor_proprioceptive_input_dim=self.actor_proprioceptive_input_dim,
                normalizer=normalizer,
            )
        else:
            exporter = _ActorCriticSRUONNXExporterSingleCam(
                attn_image_net=self.attn_image_net,
                memory_a=self.memory_a,
                linear_dropout_actor=self.linear_dropout_actor,
                actor=self.actor,
                image_input_dims=self.image_input_dims,
                num_image_features=self.num_image_features,
                actor_proprioceptive_input_dim=self.actor_proprioceptive_input_dim,
                normalizer=normalizer,
            )

        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        # Create dummy inputs
        dummy_obs = torch.zeros(1, self.num_actor_obs)
        dummy_hidden = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
        dummy_cell = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
        dummy_inputs = (dummy_obs, dummy_hidden, dummy_cell)

        input_names = ["obs", "h_in", "c_in"]
        output_names = ["actions", "h_out", "c_out"]
        dynamic_axes = {
            "obs": {0: "batch_size"},
            "h_in": {1: "batch_size"},
            "c_in": {1: "batch_size"},
            "actions": {0: "batch_size"},
            "h_out": {1: "batch_size"},
            "c_out": {1: "batch_size"},
        }

        # Export to ONNX
        torch.onnx.export(
            exporter,
            dummy_inputs,
            filepath,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[ActorCriticSRU] Exported ONNX policy ({self.num_cameras} camera) to: {filepath}")
        print(f"  - Input 'obs': shape (1, {self.num_actor_obs})")
        print(f"  - Input 'h_in': shape ({rnn.num_layers}, 1, {rnn.hidden_size})")
        print(f"  - Input 'c_in': shape ({rnn.num_layers}, 1, {rnn.hidden_size})")
        print(f"  - Output 'actions', 'h_out', 'c_out'")
        print(f"  - Note: Initialize h_in/c_in to zeros at episode start")


class _ActorCriticSRUExporterSingleCam(nn.Module):
    """JIT-exportable wrapper for ActorCriticSRU with single camera (optimized)."""

    __constants__ = ['num_image_features', 'actor_proprioceptive_input_dim', 'image_c', 'image_h', 'image_w']

    def __init__(
        self,
        attn_image_net,
        memory_a,
        linear_dropout_actor,
        actor,
        image_input_dims: tuple,
        num_image_features: int,
        actor_proprioceptive_input_dim: int,
        normalizer=None,
    ):
        super().__init__()
        import copy

        self.attn_image_net = copy.deepcopy(attn_image_net)
        self.rnn = copy.deepcopy(memory_a.rnn)
        self.linear = copy.deepcopy(linear_dropout_actor.linear)
        self.activation = copy.deepcopy(linear_dropout_actor.activation)
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Store dimensions as constants (declared in __constants__ for JIT)
        self.num_image_features = num_image_features
        self.actor_proprioceptive_input_dim = actor_proprioceptive_input_dim
        self.image_c = image_input_dims[0]
        self.image_h = image_input_dims[1]
        self.image_w = image_input_dims[2]

        # Register hidden state buffers for LSTM_SRU
        self.register_buffer(
            "hidden_state",
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )
        self.register_buffer(
            "cell_state",
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )

    def forward(self, observations: torch.Tensor, reset: bool = False) -> torch.Tensor:
        # Reset hidden states if requested (new episode)
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()

        observations = self.normalizer(observations)

        # Split observations into proprioceptive and image parts
        other_obs = observations[..., :-self.num_image_features]
        other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)

        # Extract and reshape single camera image
        image_obs = observations[..., -self.num_image_features:]
        image_obs = image_obs.reshape(-1, self.image_c, self.image_h, self.image_w)

        # Attention processing
        image_features = self.attn_image_net(image_obs, other_obs)

        # Concatenate image features with proprioceptive observations
        combined_features = torch.cat([image_features, other_obs], dim=-1)

        # RNN processing
        combined_features, (h, c) = self.rnn(
            combined_features.unsqueeze(0),
            (self.hidden_state, self.cell_state)
        )
        self.hidden_state[:] = h
        self.cell_state[:] = c

        combined_features = combined_features.squeeze(0)

        # Linear + activation (no dropout during inference)
        combined_features = self.activation(self.linear(combined_features))

        # Actor MLP
        return self.actor(combined_features)

    @torch.jit.export
    def reset(self):
        """Reset hidden states for new episode."""
        self.hidden_state.zero_()
        self.cell_state.zero_()


class _ActorCriticSRUExporterDualCam(nn.Module):
    """JIT-exportable wrapper for ActorCriticSRU with dual cameras (optimized)."""

    __constants__ = ['num_image_features', 'actor_proprioceptive_input_dim', 'image_c', 'image_h', 'image_w']

    def __init__(
        self,
        attn_image_net,
        memory_a,
        linear_dropout_actor,
        actor,
        image_input_dims: tuple,
        num_image_features: int,
        actor_proprioceptive_input_dim: int,
        normalizer=None,
    ):
        super().__init__()
        import copy

        self.attn_image_net = copy.deepcopy(attn_image_net)
        self.rnn = copy.deepcopy(memory_a.rnn)
        self.linear = copy.deepcopy(linear_dropout_actor.linear)
        self.activation = copy.deepcopy(linear_dropout_actor.activation)
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Store dimensions as constants (declared in __constants__ for JIT)
        self.num_image_features = num_image_features
        self.actor_proprioceptive_input_dim = actor_proprioceptive_input_dim
        self.image_c = image_input_dims[0]
        self.image_h = image_input_dims[1]
        self.image_w = image_input_dims[2]

        # Register hidden state buffers for LSTM_SRU
        self.register_buffer(
            "hidden_state",
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )
        self.register_buffer(
            "cell_state",
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )

    def forward(self, observations: torch.Tensor, reset: bool = False) -> torch.Tensor:
        # Reset hidden states if requested (new episode)
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()

        observations = self.normalizer(observations)

        # Split observations into proprioceptive and image parts (2 cameras)
        other_obs = observations[..., :-self.num_image_features * 2]
        other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)

        # Extract and reshape dual camera images
        image_obs_front = observations[..., -self.num_image_features * 2: -self.num_image_features]
        image_obs_back = observations[..., -self.num_image_features:]
        image_list: List[torch.Tensor] = [
            image_obs_front.reshape(-1, self.image_c, self.image_h, self.image_w),
            image_obs_back.reshape(-1, self.image_c, self.image_h, self.image_w),
        ]

        # Attention processing
        image_features = self.attn_image_net(image_list, other_obs)

        # Concatenate image features with proprioceptive observations
        combined_features = torch.cat([image_features, other_obs], dim=-1)

        # RNN processing
        combined_features, (h, c) = self.rnn(
            combined_features.unsqueeze(0),
            (self.hidden_state, self.cell_state)
        )
        self.hidden_state[:] = h
        self.cell_state[:] = c

        combined_features = combined_features.squeeze(0)

        # Linear + activation (no dropout during inference)
        combined_features = self.activation(self.linear(combined_features))

        # Actor MLP
        return self.actor(combined_features)

    @torch.jit.export
    def reset(self):
        """Reset hidden states for new episode."""
        self.hidden_state.zero_()
        self.cell_state.zero_()


def get_activation(act_name: str) -> nn.Module:
    """Get activation function by name."""
    if act_name == "elu":
        return nn.ELU(inplace=True)
    elif act_name == "celu":
        return nn.CELU(inplace=True)
    elif act_name == "selu":
        return nn.SELU(inplace=True)
    elif act_name == "relu":
        return nn.ReLU(inplace=True)
    elif act_name == "lrelu":
        return nn.LeakyReLU(inplace=True)
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {act_name}")


class MemorySRU(torch.nn.Module):
    """Memory module using plain LSTM_SRU (without attention).

    Attention is applied externally in batched mode for efficiency.

    Args:
        input_size: Size of input features (after attention + proprioceptive).
        num_layers: Number of RNN layers.
        hidden_size: Hidden state size.
    """

    def __init__(self, input_size: int, num_layers: int = 1, hidden_size: int = 256):
        super().__init__()
        print(f"[MemorySRU] Init: input_size={input_size}, num_layers={num_layers}, hidden_size={hidden_size}")

        self.rnn = LSTM_SRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.hidden_states = None

    def forward(self, input_features, masks=None, hidden_states=None):
        """Forward pass through the memory module.

        Args:
            input_features: Combined features tensor (after attention).
            masks: Mask tensor for batch mode (None for inference).
            hidden_states: Hidden states from previous step.

        Returns:
            RNN output tensor.
        """
        batch_mode = masks is not None
        if batch_mode:
            # In batch mode, hidden states must be provided.
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input_features, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # In inference mode, use (or initialize) the module's hidden state.
            out, self.hidden_states = self.rnn(input_features.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones, use_random_init=True):
        """Reset hidden states for terminated episodes.

        Memory-optimized: only operates on terminated environments instead of
        cloning the entire hidden state tensor.
        """
        if self.hidden_states is None:
            return

        done_mask = dones == 1
        if not done_mask.any():
            return

        done_indices = done_mask.nonzero(as_tuple=True)[0]
        num_done = done_indices.size(0)
        batch_size = dones.size(0)

        for hidden_state in self.hidden_states:
            if use_random_init:
                # Sample random indices from entire batch (preserves original semantics)
                random_indices = torch.randint(0, batch_size, (num_done,), device=hidden_state.device)
                # Clone only the selected states to avoid in-place modification issues
                hidden_state[..., done_indices, :] = hidden_state[..., random_indices, :].clone()
            else:
                # Use zero initialization
                hidden_state[..., done_indices, :] = 0.0


class SimpleConsistentDropout(nn.Module):
    """Dropout that maintains consistent masks across a trajectory."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.scale_factor = 1.0 / (1.0 - p)

    def forward(self, x, dropout_masks=None):
        if self.training:
            if dropout_masks is None:
                # Pre-scale the mask to avoid chained multiplications
                dropout_masks = torch.empty_like(x).bernoulli_(1 - self.p).mul_(self.scale_factor)
            out = x * dropout_masks
            return out, dropout_masks
        else:
            return x, None


class LinearConstDropout(nn.Module):
    """Linear layer with activation and consistent dropout.

    Combines a Linear layer, an activation, and consistent dropout that
    maintains the same mask across a trajectory.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float, activation_name: str):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = get_activation(activation_name)
        self.dropout = SimpleConsistentDropout(p=dropout_p)
        self.dropout_masks = None

    def forward(self, x, dropout_masks=None):
        x = self.linear(x)
        x = self.activation(x)
        if dropout_masks is None:
            x, self.dropout_masks = self.dropout(x, dropout_masks=self.dropout_masks)
        else:
            x, _ = self.dropout(x, dropout_masks)
        return x

    def get_dropout_mask(self):
        return self.dropout_masks

    def reset_dropout_mask(self):
        self.dropout_masks = None


class _ActorCriticSRUONNXExporterSingleCam(nn.Module):
    """ONNX-exportable wrapper for ActorCriticSRU with single camera.

    Hidden states are explicit inputs/outputs since ONNX doesn't support stateful ops.
    Positional encodings are lazily cached on first forward pass.
    """

    def __init__(
        self,
        attn_image_net,
        memory_a,
        linear_dropout_actor,
        actor,
        image_input_dims: tuple,
        num_image_features: int,
        actor_proprioceptive_input_dim: int,
        normalizer=None,
    ):
        super().__init__()
        import copy

        self.attn_image_net = copy.deepcopy(attn_image_net)
        self.rnn = copy.deepcopy(memory_a.rnn)
        self.linear = copy.deepcopy(linear_dropout_actor.linear)
        self.activation = copy.deepcopy(linear_dropout_actor.activation)
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Store dimensions
        self.num_image_features = num_image_features
        self.actor_proprioceptive_input_dim = actor_proprioceptive_input_dim
        self.image_c = image_input_dims[0]
        self.image_h = image_input_dims[1]
        self.image_w = image_input_dims[2]

    def forward(
        self, observations: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit hidden state I/O."""
        observations = self.normalizer(observations)

        # Split observations into proprioceptive and image parts
        other_obs = observations[..., :-self.num_image_features]
        other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)

        # Extract and reshape single camera image, add depth dimension
        image_obs = observations[..., -self.num_image_features:]
        image_obs = image_obs.reshape(-1, self.image_c, self.image_h, self.image_w)
        image_obs = image_obs.unsqueeze(2)  # (B, C, 1, H, W)

        # Attention processing (positional encoding cached on first call)
        image_features = self.attn_image_net(image_obs, other_obs)

        # Concatenate image features with proprioceptive observations
        combined_features = torch.cat([image_features, other_obs], dim=-1)

        # RNN processing with explicit state I/O
        combined_features, (h_out, c_out) = self.rnn(
            combined_features.unsqueeze(0),
            (hidden_state, cell_state)
        )
        combined_features = combined_features.squeeze(0)

        # Linear + activation (no dropout during inference)
        combined_features = self.activation(self.linear(combined_features))

        # Actor MLP
        actions = self.actor(combined_features)
        return actions, h_out, c_out


class _ActorCriticSRUONNXExporterDualCam(nn.Module):
    """ONNX-exportable wrapper for ActorCriticSRU with dual cameras.

    Hidden states are explicit inputs/outputs since ONNX doesn't support stateful ops.
    Positional encodings are lazily cached on first forward pass.
    """

    def __init__(
        self,
        attn_image_net,
        memory_a,
        linear_dropout_actor,
        actor,
        image_input_dims: tuple,
        num_image_features: int,
        actor_proprioceptive_input_dim: int,
        normalizer=None,
    ):
        super().__init__()
        import copy

        self.attn_image_net = copy.deepcopy(attn_image_net)
        self.rnn = copy.deepcopy(memory_a.rnn)
        self.linear = copy.deepcopy(linear_dropout_actor.linear)
        self.activation = copy.deepcopy(linear_dropout_actor.activation)
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Store dimensions
        self.num_image_features = num_image_features
        self.actor_proprioceptive_input_dim = actor_proprioceptive_input_dim
        self.image_c = image_input_dims[0]
        self.image_h = image_input_dims[1]
        self.image_w = image_input_dims[2]

    def forward(
        self, observations: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit hidden state I/O."""
        observations = self.normalizer(observations)

        # Split observations into proprioceptive and image parts (2 cameras)
        other_obs = observations[..., :-self.num_image_features * 2]
        other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)

        # Extract and reshape dual camera images, then stack along depth dimension
        image_obs_front = observations[..., -self.num_image_features * 2: -self.num_image_features]
        image_obs_back = observations[..., -self.num_image_features:]
        image_front = image_obs_front.reshape(-1, self.image_c, self.image_h, self.image_w)
        image_back = image_obs_back.reshape(-1, self.image_c, self.image_h, self.image_w)
        # Stack along depth dimension: (B, C, 2, H, W)
        image_stacked = torch.stack([image_front, image_back], dim=2)

        # Attention processing (positional encoding cached on first call)
        image_features = self.attn_image_net(image_stacked, other_obs)

        # Concatenate image features with proprioceptive observations
        combined_features = torch.cat([image_features, other_obs], dim=-1)

        # RNN processing with explicit state I/O
        combined_features, (h_out, c_out) = self.rnn(
            combined_features.unsqueeze(0),
            (hidden_state, cell_state)
        )
        combined_features = combined_features.squeeze(0)

        # Linear + activation (no dropout during inference)
        combined_features = self.activation(self.linear(combined_features))

        # Actor MLP
        actions = self.actor(combined_features)
        return actions, h_out, c_out
