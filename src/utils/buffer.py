# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch
from models.modules.distributions import MyDist


class RolloutBuffer:
    def __init__(self, step_start: int, step_end: int, step_current: int) -> None:
        """
        saves prediction, step [step_start,...,step_end]
        """
        self.step_start = step_start  # first step 0 in the rollout buffer corresponds to step_start in absolute time
        self.step_end = step_end  # last step in the rollout buffer corresponds to step_end in absolute time
        # this step in the rollout buffer corresponds to step_current+1 in absolute time
        self.step_future_start = step_current + 1 - step_start

        self.override_masks: List[Tensor] = []  # n_step * [n_batch, n_agent], bool

        # for logging
        self.valid: List[Tensor] = []  # n_step * [n_batch, n_agent], bool
        self.preds: List[Tensor] = []  # n_step * [n_batch, n_agent, 4], grad, x,y,yaw,v

        # for diffbar reward
        self.diffbar_rewards: List[Tensor] = []  # n_step * [n_batch, n_agent], grad
        self.diffbar_rewards_valid: List[Tensor] = []  # n_step * [n_batch, n_agent], bool

        # for VAE KL loss, len = n_step (traffic_sim) or 1(others that do not update latent)
        self.latents: List[Tuple[MyDist, MyDist]] = []
        self.latent_log_probs: List[Tensor] = []
        self.action_log_probs: List[Tensor] = []

        # for simulation metrics
        self.violations = {}  # Dict: n_step * [n_batch, n_agent], no_grad

        #  for visualizing video
        self.vis_dicts = {}

    def add(
        self,
        valid: Tensor,
        pred: Tensor,
        override_mask: Tensor,
        violation: Dict[str, Tensor],
        diffbar_reward: Optional[Tensor],
        diffbar_reward_valid: Optional[Tensor],
        latent_log_prob: Tensor,
        action_log_prob: Tensor,
        vis_dict: Dict[str, Tensor],
    ) -> None:
        self.valid.append(valid)  # [n_batch, n_agent]
        self.preds.append(pred)  # [n_batch, n_agent, 4]
        self.override_masks.append(override_mask)  # [n_batch, n_agent]

        if len(self.violations) == 0:
            self.violations = {k: [] for k in violation.keys()}
        for k, v in violation.items():
            self.violations[k].append(v)

        if diffbar_reward is not None:
            self.diffbar_rewards.append(diffbar_reward)  # [n_batch, n_agent]
            self.diffbar_rewards_valid.append(diffbar_reward_valid)

        self.latent_log_probs.append(latent_log_prob)  # [n_batch, n_agent]
        self.action_log_probs.append(action_log_prob)  # [n_batch, n_agent]

        if len(self.vis_dicts) == 0:
            self.vis_dicts = {k: [] for k in vis_dict.keys()}
        for k, v in vis_dict.items():
            self.vis_dicts[k].append(v)

    def finish(self) -> None:
        self.valid = torch.stack(self.valid, dim=2)  # [n_batch, n_agent, n_step]
        self.preds = torch.stack(self.preds, dim=2)  # [n_batch, n_agent, n_step, 4]
        self.override_masks = torch.stack(self.override_masks, dim=2)  # [n_batch, n_agent, n_step]

        # [n_batch, n_agent, n_step], no_grad
        for k in self.violations.keys():
            self.violations[k] = torch.stack(self.violations[k], dim=2)

        # [n_batch, n_agent, n_step]
        if len(self.diffbar_rewards) > 0:  # grad
            self.diffbar_rewards = torch.stack(self.diffbar_rewards, dim=2)
            self.diffbar_rewards_valid = torch.stack(self.diffbar_rewards_valid, dim=2)

        self.latent_log_probs = torch.stack(self.latent_log_probs, dim=2)  # [n_batch, n_agent, n_step]
        self.action_log_probs = torch.stack(self.action_log_probs, dim=2)  # [n_batch, n_agent, n_step]

        for k in self.vis_dicts.keys():
            self.vis_dicts[k] = torch.stack(self.vis_dicts[k], dim=2)

    def flatten_repeat(self, n_repeat: int) -> None:
        n_batch, n_agent, n_step, d_preds = self.preds.shape
        n_batch = n_batch // n_repeat

        # [n_batch, n_agent, n_repeat, n_step]
        self.valid = self.valid.view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)
        self.override_masks = self.override_masks.view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)

        # [n_batch, n_agent, n_repeat, n_step, 4]
        self.preds = self.preds.view(n_batch, n_repeat, n_agent, n_step, d_preds).transpose(1, 2)

        # [n_batch, n_agent, n_repeat, n_step], no_grad
        for k in self.violations.keys():
            self.violations[k] = self.violations[k].view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)

        # [n_batch, n_agent, n_repeat, n_step]
        if len(self.diffbar_rewards) > 0:  # grad
            self.diffbar_rewards = self.diffbar_rewards.view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)
            self.diffbar_rewards_valid = self.diffbar_rewards_valid.view(n_batch, n_repeat, n_agent, n_step).transpose(
                1, 2
            )

        # [n_batch, n_agent, n_repeat, n_step]
        self.latent_log_probs = self.latent_log_probs.view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)
        self.action_log_probs = self.action_log_probs.view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)

        for k in self.vis_dicts.keys():
            if "valid" in k:
                self.vis_dicts[k] = self.vis_dicts[k].view(n_batch, n_repeat, n_agent, n_step).transpose(1, 2)
            else:
                _dim = self.vis_dicts[k].shape[-1]
                self.vis_dicts[k] = self.vis_dicts[k].view(n_batch, n_repeat, n_agent, n_step, _dim).transpose(1, 2)
