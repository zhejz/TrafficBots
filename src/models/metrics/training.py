# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Optional
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from .loss import BalancedKL
from models.modules.distributions import MyDist


class TrainingMetrics(Metric):
    def __init__(
        self,
        prefix: str,
        w_vae_kl: float = 0,
        kl_balance_scale: float = 0,
        kl_free_nats: float = -1,
        kl_for_unseen_agent: bool = True,
        w_diffbar_reward: float = 0,
        w_goal: float = 0,
        w_relevant_agent: float = 0,
        loss_for_teacher_forcing: bool = True,
        p_loss_for_irrelevant: float = -1.0,
        step_training_start: int = 0,
    ) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.loss_for_teacher_forcing = loss_for_teacher_forcing
        self.p_loss_for_irrelevant = p_loss_for_irrelevant
        self.w_relevant_agent = w_relevant_agent  # set to greater than 0 to enable extra weights on relevant agents
        self.step_training_start = step_training_start

        # CVAE KL divergence
        self.w_vae_kl = w_vae_kl
        self.kl_for_unseen_agent = kl_for_unseen_agent
        if self.w_vae_kl > 0:
            self.use_vae_kl = True
            self.add_state("vae_kl_counter", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("vae_kl", default=tensor(0.0), dist_reduce_fx="sum")
            self.l_vae_kl = BalancedKL(kl_balance_scale=kl_balance_scale, kl_free_nats=kl_free_nats)
        else:
            self.use_vae_kl = False

        # diffbar reward
        self.w_diffbar_reward = w_diffbar_reward
        if self.w_diffbar_reward > 0:
            self.use_diffbar_reward = True
            # for training
            self.add_state("diffbar_reward_counter", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("diffbar_reward", default=tensor(0.0), dist_reduce_fx="sum")
        else:
            self.use_diffbar_reward = False

        self.w_goal = w_goal
        if self.w_goal > 0:
            self.add_state("goal_loss", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("goal_counter", default=tensor(0.0), dist_reduce_fx="sum")
            self.train_goal_predictor = True
        else:
            self.train_goal_predictor = False

    def update(
        self,
        pred_valid: Tensor,
        diffbar_rewards_valid: Tensor,
        diffbar_rewards: Tensor,
        override_masks: Tensor,
        agent_role: Tensor,
        goal_valid: Optional[MyDist],
        goal_pred: Optional[MyDist],
        goal_gt: Optional[Tensor],
        latent_post: MyDist,
        latent_prior: MyDist,
    ) -> None:
        """
        pred_valid: [n_batch, n_agent, n_step_rollout]
        diffbar_rewards_valid: [n_batch, n_agent, n_step_rollout]
        diffbar_rewards: [n_batch, n_agent, n_step_rollout]
        override_masks: [n_batch, n_agent, n_step_rollout]
        agent_role: [n_batch, n_agent, 4] one_hot [sdc=0, interest=1, predict=2]
        goal_valid: [n_batch, n_agent]
        goal_gt: [n_batch, n_agent, 4] float32 [x, y, theta, v] or [n_batch, n_agent] int64: index to n_pl, or None
        goal_pred: MyDist or None  # distribution, gaussian or categorical
        """
        with torch.no_grad():
            if self.p_loss_for_irrelevant > 0:
                mask_relevant = agent_role.any(-1).unsqueeze(-1)  # [n_batch, n_agent, 1]
                pred_valid = pred_valid & mask_relevant
                mask_irrelevant = torch.bernoulli(torch.ones_like(mask_relevant) * self.p_loss_for_irrelevant).bool()
                pred_valid = pred_valid | mask_irrelevant
            if not self.loss_for_teacher_forcing:
                pred_valid = pred_valid & (~override_masks)
            if self.step_training_start > 0:
                pred_valid[:, :, : self.step_training_start] &= False

            if self.w_relevant_agent > 0:
                # [n_batch, n_agent]
                w_mask_rel = pred_valid.any(-1) + agent_role.any(-1) * self.w_relevant_agent
            else:
                w_mask_rel = None

        # ! CVAE KL divergence
        if self.use_vae_kl:
            # [n_batch, n_agent]
            if self.kl_for_unseen_agent:
                # posterior valid: unseen agent has unit prior.
                vae_kl_valid = latent_post.valid
            else:
                # prior valid: no kl loss for unseen agent.
                vae_kl_valid = latent_prior.valid
            vae_kl_valid = vae_kl_valid & (pred_valid.any(-1))
            error_vae = self.l_vae_kl.compute(latent_post.distribution, latent_prior.distribution)
            self.vae_kl_counter += vae_kl_valid.sum()
            if w_mask_rel is not None:
                error_vae *= w_mask_rel
            self.vae_kl += error_vae.masked_fill(~vae_kl_valid, 0.0).sum()

        # ! diffbar reward
        if self.use_diffbar_reward:
            reward_valid = pred_valid & diffbar_rewards_valid  # [n_batch, n_agent, n_step]
            # [n_batch, n_agent, n_step]
            error_rewards_dr = diffbar_rewards.masked_fill(~reward_valid, 0.0)
            if w_mask_rel is not None:
                error_rewards_dr *= w_mask_rel.unsqueeze(1)
            self.diffbar_reward -= error_rewards_dr.sum()
            self.diffbar_reward_counter += reward_valid.sum()

        # ! goal/dest prediction
        if self.train_goal_predictor:
            # [n_batch, n_agent]
            goal_valid = goal_pred.valid & (pred_valid.any(-1))
            # same as F.cross_entropy(self.distribution.logits.transpose(1, 2), goal_gt, reduction="none")
            goal_nll = -goal_pred.log_prob(goal_gt).masked_fill(~goal_valid, 0)
            if w_mask_rel is not None:
                goal_nll *= w_mask_rel
            self.goal_loss += goal_nll.sum()
            self.goal_counter += goal_valid.sum()

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {f"{self.prefix}/loss": 0}

        # CVAE KL divergence
        if self.use_vae_kl:
            out_dict[f"{self.prefix}/vae_kl"] = self.w_vae_kl * self.vae_kl / self.vae_kl_counter
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/vae_kl"]

        # diffbar reward
        if self.use_diffbar_reward:
            # training
            out_dict[f"{self.prefix}/diffbar_reward"] = (
                self.w_diffbar_reward * self.diffbar_reward / self.diffbar_reward_counter
            )
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/diffbar_reward"]

        # goal/dest prediction
        if self.train_goal_predictor:
            out_dict[f"{self.prefix}/goal_loss"] = self.w_goal * self.goal_loss / self.goal_counter
            out_dict[f"{self.prefix}/loss"] += out_dict[f"{self.prefix}/goal_loss"]
        return out_dict
