"""Opt-in spatial temporal BEV memory; the navigation SRU remains unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from .actor_critic_sru import ActorCriticSRU


class EgoConvLSTM(nn.Module):
    def __init__(self, channels=32, grid_size=8, map_size=6.4):
        super().__init__()
        self.channels, self.grid_size, self.map_size = channels, grid_size, map_size
        self.gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[channels:2 * channels].fill_(1.0)
        centers = map_size / 2 - (torch.arange(grid_size) + 0.5) * map_size / grid_size
        x, y = torch.meshgrid(centers, centers, indexing="ij")
        self.register_buffer("centers", torch.stack((x, y), -1), persistent=False)

    def warp(self, value, previous_pose, pose):
        local = self.centers.to(value)[None]
        c, s = pose[:, 2].cos()[:, None, None], pose[:, 2].sin()[:, None, None]
        dx = c * local[..., 0] - s * local[..., 1] + pose[:, 0, None, None] - previous_pose[:, 0, None, None]
        dy = s * local[..., 0] + c * local[..., 1] + pose[:, 1, None, None] - previous_pose[:, 1, None, None]
        c, s = previous_pose[:, 2].cos()[:, None, None], previous_pose[:, 2].sin()[:, None, None]
        old_x, old_y = c * dx + s * dy, -s * dx + c * dy
        extent = self.map_size * (self.grid_size - 1) / self.grid_size / 2
        grid = torch.stack((-old_y / extent, -old_x / extent), -1)
        return F.grid_sample(value, grid, align_corners=True)

    def forward(self, inputs, pose, state=None):
        if state is None:
            h = c = torch.zeros_like(inputs)
        else:
            h, c, previous_pose = state
            h = self.warp(h, previous_pose, pose)
            c = self.warp(c, previous_pose, pose)
        i, f, o, g = self.gates(torch.cat((inputs, h), 1)).chunk(4, 1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c, pose)


class ActorCriticBEVConvLSTM(ActorCriticSRU):
    """Actor-only dynamic memory, with odometry used exclusively for ego warp.

    Actor observations append x/y/yaw after the original six-channel BEV.
    Critic architecture and all SRU input dimensions retain the old layout.
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        super().__init__(num_actor_obs - 3, num_critic_obs, num_actions, **kwargs)
        self.num_actor_obs = num_actor_obs
        if not self.use_future_occupancy or not self.use_actor_visibility_query_attention:
            raise ValueError("ConvLSTM task requires spatial BEV query and future occupancy")
        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ELU(),
        )
        self.dynamic_memory = EgoConvLSTM()
        self.dynamic_projection = nn.Conv2d(32, self.visibility_map_feature_dim, 1)
        self.bev_hidden_states = None

    def get_actor_parameters(self):
        return (super().get_actor_parameters() + list(self.dynamic_encoder.parameters())
                + list(self.dynamic_memory.parameters()) + list(self.dynamic_projection.parameters()))

    def _temporal_features(self, bev, poses, state=None, dones=None, include_static=True):
        time, batch = poses.shape[:2]
        dynamic = self.dynamic_encoder(bev[:, 3:]).reshape(time, batch, 32, 8, 8)
        static = None
        if include_static:
            static_bev = torch.cat((bev[:, :3], torch.zeros_like(bev[:, 3:])), 1)
            static = self.visibility_map_encoder(static_bev).reshape(time, batch, -1, 8, 8)
        features = []
        for t in range(time):
            h, state = self.dynamic_memory(dynamic[t], poses[t], state)
            projected = self.dynamic_projection(h)
            features.append(projected if static is None else static[t] + projected)
            if dones is not None:
                keep = (~dones[t].reshape(batch).bool()).to(h.dtype)
                state = (state[0] * keep[:, None, None, None],
                         state[1] * keep[:, None, None, None], state[2])
        return torch.stack(features), state

    @staticmethod
    def _unpack_state(hidden_states):
        if hidden_states is None:
            return None
        return (hidden_states[2][0].reshape(-1, 32, 8, 8),
                hidden_states[3][0].reshape(-1, 32, 8, 8), hidden_states[4][0])

    def process_actor_input(self, observations, masks, hidden_states):
        self._bev_poses = observations[..., -3:]
        self._bev_replay = masks is not None
        self._bev_initial = self._unpack_state(hidden_states) if self._bev_replay else self.bev_hidden_states
        sru_states = hidden_states[:2] if hidden_states is not None else None
        return super().process_actor_input(observations[..., :-3], masks, sru_states)

    def encode_actor_visibility_map(self, visibility_obs):
        poses = self._bev_poses if self._bev_replay else self._bev_poses.unsqueeze(0)
        features, state = self._temporal_features(visibility_obs, poses, self._bev_initial)
        if not self._bev_replay:
            self.bev_hidden_states = tuple(item.detach() for item in state)
        return features.flatten(0, 1)

    def process_critic_input(self, observations, masks, hidden_states):
        return super().process_critic_input(observations, masks,
                                           hidden_states[:2] if hidden_states is not None else None)

    def get_hidden_states(self):
        actor, critic = super().get_hidden_states()
        if actor is None:
            return actor, critic
        h, c, pose = self.bev_hidden_states
        spatial = (h.flatten(1).unsqueeze(0), c.flatten(1).unsqueeze(0), pose.unsqueeze(0))
        return tuple(actor) + spatial, critic

    def reset(self, dones=None):
        super().reset(dones)
        if self.bev_hidden_states is not None and dones is not None:
            keep = (~dones.reshape(-1).bool()).to(self.bev_hidden_states[0].dtype)
            h, c, pose = self.bev_hidden_states
            self.bev_hidden_states = (h * keep[:, None, None, None], c * keep[:, None, None, None], pose)

    def predict_future_from_rollout(self, storage, times, envs):
        unique_envs, inverse = torch.unique(envs, return_inverse=True)
        observations = storage.observations[:, unique_envs]
        states = tuple(state[0, :, unique_envs] for state in storage.saved_hidden_states_a)
        bev = self._extract_actor_visibility_map(observations[..., :-3])
        features, _ = self._temporal_features(bev, observations[..., -3:], self._unpack_state(states),
                                              storage.dones[:, unique_envs], include_static=False)
        selected = features[times, inverse]
        return self.future_occupancy_head(selected)

    def predict_future_occupancy(self, actor_observations):
        raise RuntimeError("ConvLSTM supervision requires continuous rollout replay, not independent frames")
