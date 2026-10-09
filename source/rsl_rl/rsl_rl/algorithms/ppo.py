#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from .future_occupancy import future_occupancy_targets

EPSILON = 1e-7

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        value_clip_param=0.5,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        lambda_mask=0.0,
        lambda_future=0.0,
        future_horizon_steps=(2, 4),
        future_batch_size=256,
        **kwargs,
    ):
        self.device = device
        self.lambda_mask = lambda_mask
        self.lambda_future = float(lambda_future)
        self.future_horizon_steps = tuple(int(h) for h in future_horizon_steps)
        self.future_batch_size = int(future_batch_size)
        self.last_future_metrics = {}
        if self.lambda_future > 0:
            if self.future_horizon_steps != (2, 4) or self.future_batch_size < 1:
                raise ValueError("Future occupancy uses horizons (2, 4) and a positive batch size")
            if not getattr(actor_critic, "use_future_occupancy", False):
                raise ValueError("lambda_future requires an enabled prediction head")

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, mask_label_shape=None, future_teacher_shape=None):
        if self.lambda_future > 0 and (future_teacher_shape is None or num_transitions_per_env <= max(self.future_horizon_steps)):
            raise ValueError("Future loss requires teacher states and a rollout longer than its horizon")
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
            mask_label_shape=mask_label_shape,
            future_teacher_shape=future_teacher_shape,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # print("obs", obs)
        # print("obs shape", obs.shape)
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
    def update_dropout_masks(self):
        # update the dropout masks
        self.storage.saved_dropout_masks_a, self.storage.saved_dropout_masks_c = self.actor_critic.get_dropout_masks()
        
    def reset_dropout_masks(self):
        self.actor_critic.reset_dropout_masks()

    def update(self, iter, max_iters):
        self.last_future_metrics = {tag: 0.0 for tag in (
            "Loss/future_occupancy", "Loss/future_bce", "Loss/future_dice",
            "Metrics/future/soft_iou", "Metrics/future/label_positive_rate",
            "Metrics/future/pred_positive_rate", "Metrics/future/valid_fraction",
        )} if self.lambda_future > 0 else {}
        future_updates = 0
        future_candidates = None
        if self.lambda_future > 0:
            teacher = self.storage.future_teacher
            slots = teacher[:-max(self.future_horizon_steps), :, 3:].reshape(
                teacher.shape[0] - max(self.future_horizon_steps), teacher.shape[1], -1, 4)
            eligible = ((slots[..., 2] > 0.5) & (slots[..., 3] > 0.5)).any(-1)
            future_candidates = eligible.nonzero()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_mask_loss = 0
        mean_mask_dice_loss = 0
        mean_mask_positive_rate = 0
        mean_mask_pred_positive_rate = 0
        mean_mask_soft_iou = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            dropout_masks_a,
            dropout_masks_c,
            mask_labels_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0], dropout_masks=dropout_masks_a)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1],
                dropout_masks=dropout_masks_c
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + EPSILON) + EPSILON)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                        
            # normalize advantages
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + EPSILON)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # print("PPO max ratio: ", ratio.max().item())

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.value_clip_param, self.value_clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean() * 0.5
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean() * 0.5

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if future_candidates is not None and future_candidates.shape[0] > 0:
                selected = future_candidates[torch.randint(
                    future_candidates.shape[0], (self.future_batch_size,), device=self.device)]
                times, envs = selected.unbind(1)
                targets, valid = future_occupancy_targets(
                    self.storage.future_teacher, self.storage.dones, times, envs,
                    self.future_horizon_steps,
                )
                if valid.any():
                    if hasattr(self.actor_critic, "predict_future_from_rollout"):
                        predictions = self.actor_critic.predict_future_from_rollout(self.storage, times, envs)
                    else:
                        predictions = self.actor_critic.predict_future_occupancy(self.storage.observations[times, envs])
                    logits, labels = predictions[valid], targets[valid]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=logits.new_tensor(20.0))
                    probabilities = logits.sigmoid()
                    intersection = (probabilities * labels).sum((-2, -1))
                    union = probabilities.sum((-2, -1)) + labels.sum((-2, -1))
                    dice = (1 - (2 * intersection + 1) / (union + 1)).mean()
                    loss = loss + self.lambda_future * (bce + dice)
                    metrics = {
                        "Loss/future_occupancy": bce + dice, "Loss/future_bce": bce,
                        "Loss/future_dice": dice,
                        "Metrics/future/soft_iou": (intersection / (union - intersection).clamp_min(1e-6)).mean(),
                        "Metrics/future/label_positive_rate": labels.mean(),
                        "Metrics/future/pred_positive_rate": (probabilities >= 0.5).float().mean(),
                        "Metrics/future/valid_fraction": valid.float().mean(),
                    }
                    for tag, value in metrics.items():
                        self.last_future_metrics[tag] += value.detach().item()
                    future_updates += 1

            # Auxiliary dynamic-obstacle segmentation loss (BCE + Dice) against the
            # training-time mask label. The mask prediction is cached by the actor
            # forward pass (self.actor_critic._mask_pred).
            if self.lambda_mask > 0.0 and mask_labels_batch is not None and self.actor_critic._mask_pred is not None:
                mask_pred = self.actor_critic._mask_pred.reshape(mask_labels_batch.shape)
                if masks_batch is not None:
                    valid = masks_batch.bool()
                    mask_pred = mask_pred[valid]
                    mask_labels_batch = mask_labels_batch[valid]
                bce = torch.nn.functional.binary_cross_entropy(mask_pred, mask_labels_batch)
                smooth = 1.0
                intersection = (mask_pred * mask_labels_batch).sum()
                dice = 1.0 - (2.0 * intersection + smooth) / (mask_pred.sum() + mask_labels_batch.sum() + smooth)
                loss = loss + self.lambda_mask * (bce + dice)
                soft_union = mask_pred.sum() + mask_labels_batch.sum() - intersection
                # Treat empty prediction/target batches as zero IoU so early
                # no-obstacle minibatches do not inflate the training metric.
                soft_iou = intersection / soft_union.clamp_min(1.0e-6)
                mean_mask_loss += (bce + dice).item()
                mean_mask_dice_loss += dice.item()
                mean_mask_positive_rate += mask_labels_batch.mean().item()
                mean_mask_pred_positive_rate += (mask_pred >= 0.5).float().mean().item()
                mean_mask_soft_iou += soft_iou.item()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip the gradients for actor and critic separately
            nn.utils.clip_grad_norm_(self.actor_critic.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.get_critic_parameters(), self.max_grad_norm)
            
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_mask_loss /= num_updates
        mean_mask_dice_loss /= num_updates
        mean_mask_positive_rate /= num_updates
        mean_mask_pred_positive_rate /= num_updates
        mean_mask_soft_iou /= num_updates
        for tag in self.last_future_metrics:
            self.last_future_metrics[tag] /= max(1, future_updates)
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_mask_loss,
            mean_mask_dice_loss,
            mean_mask_positive_rate,
            mean_mask_pred_positive_rate,
            mean_mask_soft_iou,
        )
