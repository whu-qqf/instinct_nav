import math

import pytest
import torch

from rsl_rl.algorithms.future_occupancy import future_occupancy_targets
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_sru import ActorCriticSRU


def _states():
    states = torch.zeros(6, 2, 11)
    states[:, 1, :3] = torch.tensor([10.0, 20.0, math.pi / 2])
    for step in range(6):
        x = 0.5 + step * 0.5
        states[step, 0, 3:7] = torch.tensor([x, 0.5, 1.0, 1.0])
        states[step, 1, 3:7] = torch.tensor([9.5, 20.0 + x, 1.0, 1.0])
        # An unobserved object must not leak into forecasting supervision.
        states[step, :, 7:11] = torch.tensor([-0.5, -0.5, 1.0, 0.0])
    return states


def test_targets_use_actual_future_states_in_source_frame_not_future_robot_frame():
    states = _states()
    states[1:, :, :3] = 100.0
    target, valid = future_occupancy_targets(states, torch.zeros(6, 2, 1),
                                            torch.tensor([0, 0]), torch.tensor([0, 1]),
                                            (2, 4), grid_size=8, size_m=8.0)
    assert valid.all()
    assert torch.all(target[:, 0, 2, 3] == 1)
    assert torch.all(target[:, 1, 1, 3] == 1)
    assert target.sum() == 4


def test_episode_reset_and_slot_recycle_invalidate_horizons():
    states = _states()
    states[3, 1, 5] = 0.0
    dones = torch.zeros(6, 2, 1)
    dones[1, 0] = 1
    _, valid = future_occupancy_targets(states, dones, torch.tensor([0, 0]),
                                       torch.tensor([0, 1]), (2, 4))
    assert valid.tolist() == [[False, False], [True, False]]


def test_static_or_unobserved_sources_have_no_auxiliary_supervision():
    states = _states()
    states[:, :, 6] = 0
    targets, valid = future_occupancy_targets(states, torch.zeros(6, 2, 1),
                                             torch.tensor([0]), torch.tensor([0]), (2, 4))
    assert not valid.any()
    assert targets.sum() == 0


def _policy():
    return ActorCriticSRU(
        num_actor_obs=16 + 16 + 6144, num_critic_obs=16 + 1 + 16 + 16 + 6144,
        num_actions=3, actor_hidden_dims=[32], critic_hidden_dims=[32],
        image_input_dims=(4, 2, 2), height_input_dims=(4, 2, 2),
        rnn_hidden_size=32, use_actor_visibility_map=True, use_critic_visibility_map=True,
        visibility_map_input_dims=(6, 32, 32), visibility_map_feature_dim=16,
        use_actor_visibility_query_attention=True, use_critic_visibility_query_attention=True,
        visibility_map_preserve_spatial_resolution=True, use_future_occupancy=True,
    )


def test_forecast_head_is_actor_only_and_backpropagates_to_shared_bev_encoder():
    policy = _policy()
    obs = torch.randn(3, policy.num_actor_obs)
    logits = policy.predict_future_occupancy(obs)
    assert logits.shape == (3, 2, 32, 32)
    logits.square().mean().backward()
    assert policy.visibility_map_encoder[0].weight.grad is not None
    assert policy.future_occupancy_head[-1].weight.grad is not None
    assert all(p.grad is None for p in policy.attn_height_net.parameters())
    assert policy.memory_a.rnn.input_size == 20
    assert policy.memory_c.rnn.input_size == 24
    head_ids = {id(p) for p in policy.future_occupancy_head.parameters()}
    assert head_ids <= {id(p) for p in policy.get_actor_parameters()}


@pytest.mark.parametrize("observed", [True, False])
def test_ppo_auxiliary_update_keeps_legacy_return_signature_and_trains_head(observed):
    torch.manual_seed(72)
    policy = _policy()
    alg = PPO(policy, device="cpu", lambda_future=0.05, future_batch_size=8,
              num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    alg.init_storage(2, 6, [policy.num_actor_obs], [policy.num_critic_obs], [3],
                     future_teacher_shape=[11])
    states = _states()
    if not observed:
        states[:, :, 6] = 0.0
    for step in range(6):
        alg.act(torch.randn(2, policy.num_actor_obs), torch.randn(2, policy.num_critic_obs))
        alg.transition.future_teacher = states[step]
        alg.process_env_step(torch.randn(2), torch.zeros(2, dtype=torch.bool), {})
    alg.compute_returns(torch.randn(2, policy.num_critic_obs))
    before = policy.future_occupancy_head[-1].weight.detach().clone()
    result = alg.update(0, 1)
    assert len(result) == 7
    assert all(math.isfinite(x) for x in result)
    if observed:
        assert alg.last_future_metrics["Loss/future_occupancy"] > 0
        assert alg.last_future_metrics["Metrics/future/label_positive_rate"] > 0
        assert not torch.equal(before, policy.future_occupancy_head[-1].weight)
    else:
        assert alg.last_future_metrics["Loss/future_occupancy"] == 0
        assert torch.equal(before, policy.future_occupancy_head[-1].weight)
    assert alg.storage.step == 0


def test_future_loss_configuration_requires_teacher_and_head():
    policy = _policy()
    alg = PPO(policy, lambda_future=0.05)
    with pytest.raises(ValueError):
        alg.init_storage(2, 6, [policy.num_actor_obs], [policy.num_critic_obs], [3])
