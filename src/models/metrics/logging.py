# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from utils.transform_utils import cast_rad


class ErrorMetrics(Metric):
    def __init__(self, prefix: str, loss_for_teacher_forcing: bool = False) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.loss_for_teacher_forcing = loss_for_teacher_forcing

        self.add_state("err_counter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_pos_meter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_rot_deg", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_spd_m_per_s", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_valid: Tensor,
        pred_states: Tensor,
        gt_valid: Tensor,
        gt_states: Tensor,
        override_masks: Tensor,
        agent_role: Tensor,
    ) -> None:
        """
        pred_valid: [n_batch, n_agent, (n_future), n_step_rollout]
        pred_states: [n_batch, n_agent, (n_future), n_step_rollout, 4]
        gt_valid: [n_batch, n_agent, n_step_rollout]
        gt_states: [n_batch, n_agent, n_step_rollout, 4]
        override_masks: [n_batch, n_agent, (n_future), n_step_rollout]
        agent_role: [n_batch, n_agent, 4] one_hot [sdc=0, interest=1, predict=2]
        """
        with torch.no_grad():
            mask_relevant = agent_role.any(-1).unsqueeze(-1).unsqueeze(-1)  # [n_batch, n_agent, 1, 1]
            gt_valid = gt_valid.unsqueeze(2)
            gt_states = gt_states.unsqueeze(2)

            pred_valid = pred_valid & mask_relevant
            if not self.loss_for_teacher_forcing:
                pred_valid = pred_valid & (~override_masks)

            # [n_batch, n_step, n_agent]
            err_valid = gt_valid & pred_valid
            # [n_batch, n_step, n_agent, 4]
            gt_states = gt_states.masked_fill(~err_valid.unsqueeze(-1), 0.0)
            states = pred_states.masked_fill(~err_valid.unsqueeze(-1), 0.0)

            self.err_counter += err_valid.sum()
            self.err_pos_meter += torch.norm(gt_states[..., :2] - states[..., :2], dim=-1).sum()
            self.err_rot_deg += torch.abs(torch.rad2deg(cast_rad(gt_states[..., 2] - states[..., 2]))).sum()
            self.err_spd_m_per_s += torch.abs(gt_states[..., 3] - states[..., 3]).sum()

    def compute(self) -> Dict[str, Tensor]:
        # logging
        out_dict = {
            f"{self.prefix}/err/pos_meter": self.err_pos_meter / self.err_counter,
            f"{self.prefix}/err/rot_deg": self.err_rot_deg / self.err_counter,
            f"{self.prefix}/err/spd_m_per_s": self.err_spd_m_per_s / self.err_counter,
        }

        return out_dict


class TrafficRuleMetrics(Metric):
    """
    Log traffic_rule_violations, Not based on ground truth trajectory.
    n_agent_collided / n_agent_valid, collided if collision happened at any time step.
    """

    def __init__(self, prefix: str, loss_for_teacher_forcing: bool = False) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.loss_for_teacher_forcing = loss_for_teacher_forcing
        self.add_state("counter_agent", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter_veh", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("outside_map", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("collided", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("run_red_light", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("goal_reached", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dest_reached", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("run_road_edge", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("passive", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        valid: Tensor,
        override_masks: Tensor,
        outside_map: Tensor,
        collided: Tensor,
        run_road_edge: Tensor,
        run_red_light: Tensor,
        passive: Tensor,
        goal_reached: Tensor,
        dest_reached: Tensor,
        agent_type: Tensor,
    ) -> None:
        with torch.no_grad():
            if self.loss_for_teacher_forcing:
                agent_valid = valid.any(-1)
            else:
                # exclude teacher_forcing agents
                # [n_batch, n_agent, n_step]
                agent_valid = valid & (~override_masks)
                invalid_mask = ~agent_valid
                outside_map = outside_map.masked_fill(invalid_mask, 0)
                collided = collided.masked_fill(invalid_mask, 0)
                run_road_edge = run_road_edge.masked_fill(invalid_mask, 0)
                run_red_light = run_red_light.masked_fill(invalid_mask, 0)
                passive = passive.masked_fill(invalid_mask, 0)
                goal_reached = goal_reached.masked_fill(invalid_mask, 0)
                dest_reached = dest_reached.masked_fill(invalid_mask, 0)
                # [n_batch, n_agent]
                agent_valid = agent_valid.any(-1)

        self.counter_agent += agent_valid.sum()
        mask_veh = agent_type[:, :, 0:1]
        self.counter_veh += (agent_valid & mask_veh).sum()
        self.outside_map += outside_map.any(-1).sum()
        self.collided += collided.any(-1).sum()
        self.run_road_edge += run_road_edge.any(-1).sum()
        self.run_red_light += run_red_light.any(-1).sum()
        self.passive += passive.any(-1).sum()
        self.goal_reached += goal_reached.any(-1).sum()
        self.dest_reached += dest_reached.any(-1).sum()

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {
            f"{self.prefix}/traffic_rule/outside_map": self.outside_map / self.counter_agent,
            f"{self.prefix}/traffic_rule/collided": self.collided / self.counter_agent,
            f"{self.prefix}/traffic_rule/run_road_edge": self.run_road_edge / self.counter_veh,
            f"{self.prefix}/traffic_rule/run_red_light": self.run_red_light / self.counter_veh,
            f"{self.prefix}/traffic_rule/passive": self.passive / self.counter_veh,
            f"{self.prefix}/traffic_rule/goal_reached": self.goal_reached / self.counter_agent,
            f"{self.prefix}/traffic_rule/dest_reached": self.dest_reached / self.counter_agent,
        }
        return out_dict
