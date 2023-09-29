# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple
from torch import Tensor
import torch
from omegaconf import DictConfig
from models.metrics.loss import AngularError


class DifferentiableReward:
    def __init__(
        self,
        l_pos: DictConfig,
        l_rot: DictConfig,
        l_spd: DictConfig,
        w_collision: float,
        use_il_loss: bool,
        reduce_collsion_with_max: bool,
    ):
        # traffic_rule
        self.w_collision = w_collision
        self.reduce_collsion_with_max = reduce_collsion_with_max

        # imitation
        self.use_il_loss = use_il_loss
        if self.use_il_loss:
            self.il_l_pos = getattr(torch.nn, l_pos.criterion)(reduction="none")
            self.il_w_pos = l_pos.weight
            self.il_l_rot = AngularError(l_rot.criterion, l_rot.angular_type)
            self.il_w_rot = l_rot.weight
            self.il_l_spd = getattr(torch.nn, l_spd.criterion)(reduction="none")
            self.il_w_spd = l_spd.weight

    def get(
        self, agent_valid: Tensor, agent_state: Tensor, gt_valid: Tensor, gt_state: Tensor, agent_size: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            agent_valid: [n_batch, n_agent] bool
            agent_state: [n_batch, n_agent, 4]: x,y,theta,spd
            gt_valid: [n_batch, n_agent] bool
            gt_state: [n_batch, n_agent, 4] x,y,theta,spd
            agent_size: [n_batch, n_agent, 3], length, width, height
        Returns:
            reward: [n_batch, n_agent]
            reward_valid : [n_batch, n_agent]
        """
        reward = torch.zeros_like(agent_state[:, :, 0])
        reward_valid = agent_valid
        if self.w_collision > 0:
            n_batch, n_agent = agent_valid.shape
            # [n_batch, n_agent, 2]
            agent_xy = agent_state[..., :2]
            # [n_batch, n_agent]
            agent_yaw = agent_state[..., 2]
            # [n_batch, n_agent, 2]
            agent_heading = torch.stack([torch.cos(agent_yaw), torch.sin(agent_yaw)], axis=-1)

            # [n_batch, n_agent]
            agent_w = agent_size[:, :, :2].amin(-1)
            agent_l = agent_size[:, :, :2].amax(-1)
            agent_d = (agent_l - agent_w) / 4.0
            # [n_batch, n_agent, 2]
            agent_d = agent_d.unsqueeze(-1).expand(-1, -1, 2)

            # [n_batch, n_agent, 5, 2] centroid of 5 circles
            centroids = agent_xy.unsqueeze(2).expand(-1, -1, 5, -1) + torch.stack(
                [
                    -2 * agent_heading * agent_d,
                    -1 * agent_heading * agent_d,
                    0 * agent_heading * agent_d,
                    1 * agent_heading * agent_d,
                    2 * agent_heading * agent_d,
                ],
                dim=2,
            )

            # [n_batch, n_agent, 5, 2] -> [n_batch, n_agent, n_agent, 5, 2]
            centroids_0 = centroids.unsqueeze(2).expand(-1, -1, n_agent, -1, -1)
            centroids_1 = centroids_0.transpose(1, 2)
            # [n_batch, n_agent] -> [n_batch, n_agent, n_agent]
            agent_r = agent_w.unsqueeze(-1).expand(-1, -1, n_agent) / 2.0 + torch.finfo(agent_state.dtype).eps
            agent_r_sum = agent_r.transpose(1, 2) + agent_r

            distances = torch.zeros([n_batch, n_agent, n_agent, 5, 5], device=agent_valid.device)

            for i in range(5):
                for j in range(5):
                    # [n_batch, n_agent, n_agent, 2]
                    diff = centroids_0[:, :, :, i, :] - centroids_1[:, :, :, j, :]
                    # [n_batch, n_agent, n_agent]
                    _dist = torch.norm(diff, dim=-1) + torch.finfo(agent_state.dtype).eps
                    distances[:, :, :, i, j] = _dist

            # [n_batch, n_agent, n_agent]
            distances = torch.min(distances.flatten(start_dim=3, end_dim=4), dim=-1)[0]
            # relaxed collision: 1 for fully overlapped, 0 for not overlapped
            collision = torch.clamp(1 - distances / agent_r_sum, min=0)

            # [n_batch, n_agent, n_agent]
            ego_mask = torch.eye(n_agent, device=agent_valid.device, dtype=torch.bool)[None, :, :].expand(
                n_batch, -1, -1
            )
            ego_mask = ego_mask | (~agent_valid[:, :, None])
            ego_mask = ego_mask | (~agent_valid[:, None, :])
            collision.masked_fill_(ego_mask, 0.0)

            if self.reduce_collsion_with_max:
                # [n_batch, n_agent, n_agent] -> [n_batch, n_agent]: reduce dim 2
                collision = collision.amax(2)
            else:
                # [n_batch, n_agent, n_agent] -> [n_batch, n_agent, n_agent]
                collision = torch.clamp(collision, max=1)
                # [n_batch, n_agent]: reduce n_agent, as_valid: [n_batch, n_agent]
                collision = collision.sum(-1) / (agent_valid.sum(-1, keepdim=True))
            reward = reward - self.w_collision * collision.masked_fill(~agent_valid, 0.0)

        if self.use_il_loss and (gt_valid is not None):
            # [n_batch, n_agent, 1]
            il_invalid = ~(agent_valid & gt_valid).unsqueeze(-1)
            # [n_batch, n_agent, 4]
            gt_state = gt_state.masked_fill(il_invalid, 0)
            agent_state = agent_state.masked_fill(il_invalid, 0)

            error_pos = self.il_l_pos(gt_state[..., :2], agent_state[..., :2]).sum(-1)
            error_rot = self.il_l_rot.compute(gt_state[..., 2], agent_state[..., 2])
            error_spd = self.il_l_spd(gt_state[..., 3], agent_state[..., 3])
            il_loss = self.il_w_pos * error_pos + self.il_w_rot * error_rot + self.il_w_spd * error_spd
            reward = reward - il_loss
            reward_valid = agent_valid & gt_valid

        return reward.masked_fill(~reward_valid, 0.0), reward_valid
