# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom RSL-RL configuration classes for navigation tasks.

These config classes replace the standard Isaac Lab RL configs to support
custom network architectures used in navigation tasks with depth camera inputs.
"""

from dataclasses import MISSING
from typing import Literal, Optional

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with navigation extensions."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    rnn_type: str = "lstm"
    """The type of RNN to use."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    dropout: float = 0.0
    """The dropout rate for the first layer of the actor and critic networks."""

    # Visual inputs
    num_cameras: int = 1
    """Number of depth cameras encoded into the observation (1 or 2)."""
    
    image_input_dims: tuple[int, int, int] = (64, 5, 8)
    """Encoded depth feature shape as (C, H, W)."""
    
    height_input_dims: tuple[int, int, int] = (64, 7, 7)
    """Encoded height scan feature shape as (C, H, W). Default is (64, 7, 7) for 64*7*7=3136 features."""

    use_actor_visibility_map: bool = False
    """Whether the actor observation contains a local visibility map after the depth features."""

    use_critic_visibility_map: bool = False
    """Whether the critic observation contains the same local visibility map after depth."""

    use_actor_visibility_early_fusion: bool = False
    """Whether to concatenate the visibility-map feature into the actor recurrent input."""

    visibility_map_input_dims: tuple[int, int, int] = (1, 64, 64)
    """Actor visibility map shape as (C, H, W)."""

    visibility_map_feature_dim: int = 64
    """Output feature dimension of the actor visibility CNN branch."""

    use_actor_visibility_query_attention: bool = False
    """Whether to query visibility-map tokens with the actor recurrent context before fusion."""

    use_critic_visibility_query_attention: bool = False
    """Whether to query visibility-map tokens with the critic recurrent context before fusion."""

    visibility_map_query_num_heads: int = 4
    """Number of attention heads for actor visibility-map query fusion."""

    use_actor_scene_tokens: bool = False
    """Whether the actor observation contains scene tokens after the depth features."""

    use_critic_scene_tokens: bool = False
    """Whether the critic observation contains scene tokens after the depth features."""

    scene_token_num_tokens: int = 64
    """Number of fixed scene tokens in the actor observation."""

    scene_token_dim: int = 6
    """Raw feature dimension of each scene token."""

    scene_token_feature_dim: int = 64
    """Projected feature dimension used by scene-token query attention."""

    scene_token_query_num_heads: int = 4
    """Number of attention heads for actor scene-token query fusion."""

    use_dynamic_map_cross_attention: bool = False
    """Whether raw scene patches are fused by current-to-history cross-attention."""

    visibility_map_preserve_spatial_resolution: bool = False
    """Keep 32x32 query-map features at 8x8 instead of expanding a 4x4 bottleneck."""

    use_future_occupancy: bool = False
    """Train a two-horizon occupancy decoder from actor BEV features only."""

    use_dynamic_scene_recurrent_fusion: bool = False
    """Whether dynamic BEV scene-token context is also fused into the actor/critic SRU inputs."""

    dynamic_scene_recurrent_feature_dim: int = 64
    """Feature dimension of the presence-weighted dynamic context entering each SRU."""

    use_dynamic_mask: bool = False
    """Whether actor/critic observations contain a trailing dynamic-mask input block."""

    dynamic_mask_input_mode: str = "predict_from_depth"
    """Mask input mode: predict_from_depth learns segmentation, direct_mask flattens
    an external mask, and encoded_mask applies a spatial CNN before recurrent fusion."""

    mask_input_dims: tuple[int, int, int] = (1, 36, 64)
    """Trailing input shape as (C, H, W): raw depth for predicted mode or compact mask for direct mode."""

    mask_feature_dim: int = 64
    """Predicted-mode bottleneck size; direct mode always uses the flattened mask size."""

    dynamic_mask_fusion_mode: Literal["concat", "gated_add"] = "concat"
    """How the mask feature enters the recurrent stream. ``concat`` preserves the
    existing checkpoints; ``gated_add`` projects it into each 64D depth feature."""

    dynamic_mask_gate_init: float = 0.1
    """Initial sigmoid gate value for gated mask-to-depth residual fusion."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    value_clip_param: float = 0.2
    """The value clipping parameter. Default is 0.2."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    lambda_mask: float = 0.0
    """Auxiliary segmentation coefficient; zero disables the depth-to-mask loss."""

    lambda_future: float = 0.0
    """Auxiliary future occupancy coefficient; only future-aware tasks enable it."""
    future_horizon_steps: tuple[int, int] = (2, 4)
    """0.4/0.8 seconds at the navigation policy's 5 Hz rate."""
    future_batch_size: int = 256
    """Maximum sampled source frames per PPO minibatch auxiliary loss."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: Optional[int] = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    reward_shifting_value: float = 0.0
    """The value to shift the reward by. Default is 0.0."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string."""

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all)."""
