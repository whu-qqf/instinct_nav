import pytest
import torch

from rsl_rl.modules.actor_critic_sru import ActorCriticSRU


@pytest.mark.parametrize("preserve_spatial", [False, True])
def test_learned_bev_query_fuses_after_actor_and_critic_sru(preserve_spatial):
    proprio_dim = 16
    image_dims = (4, 2, 2)
    height_dims = (4, 2, 2)
    visibility_dims = (6, 32, 32)
    image_features = 4 * 2 * 2
    height_features = 4 * 2 * 2
    visibility_features = 6 * 32 * 32

    policy = ActorCriticSRU(
        num_actor_obs=proprio_dim + image_features + visibility_features,
        num_critic_obs=proprio_dim + 1 + height_features + image_features + visibility_features,
        num_actions=3,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        image_input_dims=image_dims,
        height_input_dims=height_dims,
        rnn_hidden_size=32,
        visibility_map_input_dims=visibility_dims,
        visibility_map_feature_dim=16,
        visibility_map_query_num_heads=4,
        use_actor_visibility_map=True,
        use_critic_visibility_map=True,
        use_actor_visibility_query_attention=True,
        use_critic_visibility_query_attention=True,
        visibility_map_preserve_spatial_resolution=preserve_spatial,
    )

    # Late query fusion must not enlarge either recurrent input.
    assert policy.mlp_input_dim_actor == proprio_dim + image_dims[0]
    assert policy.mlp_input_dim_critic == proprio_dim + image_dims[0] + height_dims[0]
    assert policy.actor_post_recurrent_input_dim == 32 + 16
    assert policy.critic_post_recurrent_input_dim == 32 + 16 + 8
    conv_layers = [layer for layer in policy.visibility_map_encoder if isinstance(layer, torch.nn.Conv2d)]
    assert conv_layers[-1].stride == ((1, 1) if preserve_spatial else (2, 2))
    features = torch.randn(2, *visibility_dims)
    for layer in list(policy.visibility_map_encoder)[:-1]:
        features = layer(features)
    assert features.shape[-2:] == ((8, 8) if preserve_spatial else (4, 4))

    actor_obs = torch.randn(2, proprio_dim + image_features + visibility_features)
    critic_obs = torch.randn(
        2, proprio_dim + 1 + height_features + image_features + visibility_features
    )
    actions = policy.act_inference(actor_obs)
    values = policy.evaluate(critic_obs)

    assert actions.shape == (2, 3)
    assert values.shape == (2, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()
    (actions.square().mean() + values.square().mean()).backward()
    assert conv_layers[0].weight.grad is not None
