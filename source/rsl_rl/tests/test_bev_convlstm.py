import math

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_bev_convlstm import ActorCriticBEVConvLSTM, EgoConvLSTM


def policy():
    return ActorCriticBEVConvLSTM(
        num_actor_obs=16 + 16 + 6144 + 3, num_critic_obs=16 + 1 + 16 + 16 + 6144,
        num_actions=3, actor_hidden_dims=[32], critic_hidden_dims=[32],
        image_input_dims=(4, 2, 2), height_input_dims=(4, 2, 2),
        rnn_hidden_size=32, use_actor_visibility_map=True, use_critic_visibility_map=True,
        visibility_map_input_dims=(6, 32, 32), visibility_map_feature_dim=16,
        use_actor_visibility_query_attention=True, use_critic_visibility_query_attention=True,
        visibility_map_preserve_spatial_resolution=True, use_future_occupancy=True,
    )


def test_ego_warp_translation_rotation_and_identity():
    memory = EgoConvLSTM(channels=1)
    value = torch.zeros(1, 1, 8, 8)
    value[0, 0, 2, 3] = 1
    origin = torch.zeros(1, 3)
    torch.testing.assert_close(memory.warp(value, origin, origin), value, atol=1e-6, rtol=0)
    moved = memory.warp(value, origin, torch.tensor([[0.8, 0.0, 0.0]]))
    assert moved[0, 0, 3, 3] > 0.999
    rotated = memory.warp(value, origin, torch.tensor([[0.0, 0.0, math.pi / 2]]))
    assert rotated[0, 0, 3, 5] > 0.999


def test_online_and_padded_replay_match_and_navigation_gradients_reach_memory():
    torch.manual_seed(72)
    model = policy().eval()
    observations = torch.randn(4, 2, model.num_actor_obs)
    observations[..., -3:] = 0
    with torch.no_grad():
        online = torch.stack([model.act_inference(o) for o in observations])
    initial = (torch.zeros(1, 2, 32), torch.zeros(1, 2, 32),
               torch.zeros(1, 2, 2048), torch.zeros(1, 2, 2048), torch.zeros(1, 2, 3))
    before = tuple(x.clone() for x in model.bev_hidden_states)
    replay = model.act_inference(observations, masks=torch.ones(4, 2, dtype=torch.bool), hidden_states=initial)
    torch.testing.assert_close(replay, online, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(model.bev_hidden_states, before):
        torch.testing.assert_close(actual, expected)
    replay.square().mean().backward()
    assert model.dynamic_memory.gates.weight.grad.abs().sum() > 0
    assert model.memory_a.rnn.input_size == 20
    assert model.memory_c.rnn.input_size == 24
    model.reset(torch.tensor([1, 0]))
    assert model.bev_hidden_states[0][0].count_nonzero() == 0
    assert model.bev_hidden_states[1][0].count_nonzero() == 0
    torch.testing.assert_close(model.bev_hidden_states[0][1], before[0][1])


def test_ppo_replays_spatial_states_and_trains_temporal_forecasting():
    torch.manual_seed(72)
    model = policy()
    algorithm = PPO(model, device="cpu", lambda_future=0.05, future_batch_size=2,
                    num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    algorithm.init_storage(2, 6, [model.num_actor_obs], [model.num_critic_obs], [3], future_teacher_shape=[7])
    for t in range(6):
        obs = torch.randn(2, model.num_actor_obs)
        obs[:, -3:] = 0
        critic = torch.randn(2, model.num_critic_obs)
        teacher = torch.tensor([[0., 0., 0., 0.5 + t * 0.1, 0.5, 1., 1.]]).expand(2, -1)
        with torch.no_grad():
            algorithm.act(obs, critic)
            algorithm.transition.future_teacher = teacher
            algorithm.process_env_step(torch.ones(2), torch.tensor([t == 2, False]),
                                       {"observations": {"future_teacher": teacher}})
    with torch.no_grad():
        algorithm.compute_returns(critic)
    assert len(algorithm.storage.saved_hidden_states_a) == 5
    assert len(algorithm.storage.saved_hidden_states_c) == 2
    before = model.dynamic_memory.gates.weight.detach().clone()
    assert len(algorithm.update(0, 1)) == 7
    assert not torch.equal(before, model.dynamic_memory.gates.weight)
    assert algorithm.last_future_metrics["Loss/future_occupancy"] > 0


def test_forecast_uses_past_dynamic_frames_but_not_static_channels_and_resets():
    model = policy()
    bev = torch.randn(3, 6, 32, 32, requires_grad=True)
    poses = torch.zeros(3, 1, 3)
    features, _ = model._temporal_features(bev, poses, include_static=False)
    model.future_occupancy_head(features[-1]).sum().backward()
    assert bev.grad[0, 3:].abs().sum() > 0
    assert bev.grad[:, :3].count_nonzero() == 0
    dones = torch.tensor([True, False, False]).reshape(3, 1, 1)
    full, _ = model._temporal_features(bev.detach(), poses, dones=dones, include_static=False)
    fresh, _ = model._temporal_features(bev.detach()[1:], poses[1:], include_static=False)
    torch.testing.assert_close(full[1:], fresh)
