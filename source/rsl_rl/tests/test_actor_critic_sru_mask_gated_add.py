import pytest
import torch

from rsl_rl.modules.actor_critic_sru import ActorCriticSRU


def _make_policy(fusion_mode: str = "gated_add", mask_feature_dim: int = 8) -> ActorCriticSRU:
    proprio_dim = 3
    image_dims = (8, 2, 2)
    height_dims = (8, 2, 2)
    mask_dims = (2, 16, 16)
    image_features = 8 * 2 * 2
    height_features = 8 * 2 * 2
    mask_features = 2 * 16 * 16
    return ActorCriticSRU(
        num_actor_obs=proprio_dim + image_features + mask_features,
        num_critic_obs=proprio_dim + 1 + height_features + image_features + mask_features,
        num_actions=3,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        activation="elu",
        init_noise_std=0.5,
        image_input_dims=image_dims,
        height_input_dims=height_dims,
        rnn_hidden_size=16,
        use_dynamic_mask=True,
        dynamic_mask_input_mode="encoded_mask",
        mask_input_dims=mask_dims,
        mask_feature_dim=mask_feature_dim,
        dynamic_mask_fusion_mode=fusion_mode,
        dynamic_mask_gate_init=0.1,
    )


def test_gated_add_restores_baseline_sru_input_widths_and_backpropagates():
    policy = _make_policy()

    assert policy.memory_a.rnn.input_size == 3 + 8
    assert policy.memory_c.rnn.input_size == 3 + 8 + 8

    actor_obs = torch.randn(2, policy.num_actor_obs)
    critic_obs = torch.randn(2, policy.num_critic_obs)
    mask_feature = policy.mask_encoder(torch.randn(2, 2, 16, 16))
    assert torch.allclose(
        policy.actor_mask_gate(mask_feature),
        torch.full((2, 1), 0.1),
        atol=1.0e-6,
    )

    actions = policy.act_inference(actor_obs)
    values = policy.evaluate(critic_obs)
    (actions.square().mean() + values.square().mean()).backward()

    assert actions.shape == (2, 3)
    assert values.shape == (2, 1)
    assert policy.actor_mask_adapter[1].weight.grad is not None
    assert policy.critic_mask_adapter[1].weight.grad is not None
    assert policy.mask_encoder.spatial_encoder[0].weight.grad is not None


def test_concat_mode_keeps_existing_recurrent_dimensions():
    policy = _make_policy(fusion_mode="concat")

    assert policy.memory_a.rnn.input_size == 3 + 8 + 8
    assert policy.memory_c.rnn.input_size == 3 + 8 + 8 + 8
    assert policy.actor_mask_adapter is None
    assert policy.critic_mask_adapter is None


def test_compact_concat_uses_only_its_feature_width_and_backpropagates():
    policy = _make_policy(fusion_mode="concat", mask_feature_dim=4)
    assert policy.memory_a.rnn.input_size == 3 + 8 + 4
    assert policy.memory_c.rnn.input_size == 3 + 8 + 8 + 4
    assert policy.actor_mask_gate is None
    assert policy.critic_mask_gate is None

    actor_obs = torch.randn(2, policy.num_actor_obs)
    critic_obs = torch.randn(2, policy.num_critic_obs)
    loss = policy.act_inference(actor_obs).square().mean() + policy.evaluate(critic_obs).square().mean()
    loss.backward()
    assert policy.mask_encoder.projector[-1].weight.grad is not None


def test_no_mask_control_keeps_scene_tokens_and_baseline_recurrent_widths():
    policy = ActorCriticSRU(
        num_actor_obs=3 + 8 * 2 * 2 + 4 * 5,
        num_critic_obs=3 + 1 + 8 * 2 * 2 + 8 * 2 * 2 + 4 * 5,
        num_actions=3,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        image_input_dims=(8, 2, 2),
        height_input_dims=(8, 2, 2),
        rnn_hidden_size=16,
        use_actor_scene_tokens=True,
        use_critic_scene_tokens=True,
        scene_token_num_tokens=4,
        scene_token_dim=5,
        scene_token_feature_dim=8,
        scene_token_query_num_heads=2,
        use_dynamic_map_cross_attention=True,
        use_dynamic_mask=False,
        mask_input_dims=(2, 16, 16),
        mask_feature_dim=4,
    )

    assert policy.memory_a.rnn.input_size == 3 + 8
    assert policy.memory_c.rnn.input_size == 3 + 8 + 8
    assert policy.mask_encoder is None
    assert policy.num_mask_input_features == 0

    actor_obs = torch.randn(2, policy.num_actor_obs)
    critic_obs = torch.randn(2, policy.num_critic_obs)
    loss = policy.act_inference(actor_obs).square().mean() + policy.evaluate(critic_obs).square().mean()
    loss.backward()

    assert policy.current_map_projector[0].weight.grad is not None


def test_invalid_gate_initialization_is_rejected():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        policy = _make_policy()
        ActorCriticSRU(
            num_actor_obs=policy.num_actor_obs,
            num_critic_obs=policy.num_critic_obs,
            num_actions=3,
            actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8],
            image_input_dims=(8, 2, 2),
            height_input_dims=(8, 2, 2),
            rnn_hidden_size=16,
            use_dynamic_mask=True,
            dynamic_mask_input_mode="encoded_mask",
            mask_input_dims=(2, 16, 16),
            mask_feature_dim=8,
            dynamic_mask_fusion_mode="gated_add",
            dynamic_mask_gate_init=0.0,
        )
