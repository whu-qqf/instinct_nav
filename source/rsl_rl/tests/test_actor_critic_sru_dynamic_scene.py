import torch

from rsl_rl.modules.actor_critic_sru import ActorCriticSRU


def _make_policy() -> ActorCriticSRU:
    return ActorCriticSRU(
        num_actor_obs=3 + 8 * 2 * 2 + 4 * 6,
        num_critic_obs=3 + 1 + 8 * 2 * 2 + 8 * 2 * 2 + 4 * 6,
        num_actions=3,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        activation="elu",
        init_noise_std=0.5,
        image_input_dims=(8, 2, 2),
        height_input_dims=(8, 2, 2),
        rnn_hidden_size=16,
        use_actor_scene_tokens=True,
        use_critic_scene_tokens=True,
        scene_token_num_tokens=4,
        scene_token_dim=6,
        scene_token_feature_dim=8,
        scene_token_query_num_heads=2,
        use_dynamic_map_cross_attention=True,
        use_dynamic_scene_recurrent_fusion=True,
        dynamic_scene_recurrent_feature_dim=8,
    )


def test_dynamic_scene_context_enters_actor_and_critic_sru():
    policy = _make_policy()

    assert policy.memory_a.rnn.input_size == 3 + 8 + 8
    assert policy.memory_c.rnn.input_size == 3 + 8 + 8 + 8

    actor_obs = torch.randn(2, policy.num_actor_obs)
    critic_obs = torch.randn(2, policy.num_critic_obs)
    actor_tokens = actor_obs[:, -24:].reshape(2, 4, 6)
    critic_tokens = critic_obs[:, -24:].reshape(2, 4, 6)
    actor_tokens[..., 5] = 0.0
    critic_tokens[..., 5] = 0.0

    empty_context = policy._pool_dynamic_scene_tokens(actor_tokens)
    actor_tokens[:, 1, 5] = 1.0
    critic_tokens[:, 1, 5] = 1.0
    dynamic_context = policy._pool_dynamic_scene_tokens(actor_tokens)

    assert not torch.allclose(empty_context, dynamic_context)
    actions = policy.act_inference(actor_obs)
    values = policy.evaluate(critic_obs)
    loss = actions.square().mean() + values.square().mean() + dynamic_context.square().mean()
    loss.backward()

    assert actions.shape == (2, 3)
    assert values.shape == (2, 1)
    assert policy.dynamic_scene_token_projector[0].weight.grad is not None


def test_dynamic_scene_context_supports_ppo_sequence_layout():
    policy = _make_policy()
    time_steps, batch_size = 2, 2
    actor_obs = torch.randn(time_steps, batch_size, policy.num_actor_obs)
    critic_obs = torch.randn(time_steps, batch_size, policy.num_critic_obs)
    actor_obs[..., -24:].reshape(time_steps, batch_size, 4, 6)[..., 5] = 0.0
    critic_obs[..., -24:].reshape(time_steps, batch_size, 4, 6)[..., 5] = 0.0
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool)
    hidden = (
        torch.zeros(1, batch_size, 16),
        torch.zeros(1, batch_size, 16),
    )

    actor_features = policy.process_actor_input(actor_obs, masks, hidden)
    critic_features = policy.process_critic_input(critic_obs, masks, hidden)

    assert actor_features.shape == (time_steps, batch_size, 16 + 8)
    assert critic_features.shape == (time_steps, batch_size, 16 + 8 + 8)
