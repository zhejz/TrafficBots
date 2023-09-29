# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from torch import Tensor
import torch


class TeacherForcing:
    def __init__(
        self,
        step_spawn_agent: int = 10,
        step_warm_start: int = 10,
        step_horizon: int = 0,
        step_horizon_decrease_per_epoch: int = 0,
        prob_forcing_agent: float = 0,
        prob_forcing_agent_decrease_per_epoch: float = 0,
    ):
        """
        Args:
            step_spawn_agent: spawn agents up to this time step
            step_warm_start: teacher forcing all agents up to this time step.
            step_horizon: from step 0 to step_horizon all agents will be teacher-forced
            step_horizon_decrease_per_epoch: decrease `step_horizon` every epoch. e.g. 10
            prob_forcing_agent: some agents will always be teacher-forced. e.g. 0.5
            prob_forcing_agent_decrease_per_epoch: decrease `prob_forcing_agent` every epoch. e.g. 0.1
        """
        self.step_spawn_agent = step_spawn_agent
        self.step_warm_start = step_warm_start
        self.step_horizon = step_horizon
        self.step_horizon_decrease_per_epoch = step_horizon_decrease_per_epoch
        self.prob_forcing_agent = prob_forcing_agent
        self.prob_forcing_agent_decrease_per_epoch = prob_forcing_agent_decrease_per_epoch

    @torch.no_grad()
    def get(self, as_valid: Tensor, current_epoch: int = 0, gt_sdc: bool = False) -> Tensor:
        """
        Args:
            as_valid: [n_batch, step_gt+1, n_agent] bool
            current_epoch: current training epoch

        Returns:
            mask_teacher_forcing: [n_batch, step_gt+1, n_agent]
        """
        mask_teacher_forcing = torch.zeros_like(as_valid)

        # always spawn at step 0
        mask_teacher_forcing[:, 0] |= as_valid[:, 0]
        if self.step_spawn_agent > 0:
            # spawn when valid change from False to True, because traj is interpolated.
            mask_spawn_agent = (~as_valid[:, :-1]) & as_valid[:, 1:]
            mask_spawn_agent[:, self.step_spawn_agent :] = False
            mask_teacher_forcing[:, 1:] |= mask_spawn_agent

        # warm start
        if self.step_warm_start >= 0:
            mask_teacher_forcing[:, : self.step_warm_start + 1] |= as_valid[:, : self.step_warm_start + 1]

        # horizon schedule
        step_horizon = self.step_horizon - self.step_horizon_decrease_per_epoch * current_epoch
        if step_horizon > 0:
            mask_teacher_forcing[:, :step_horizon] |= as_valid[:, :step_horizon]

        # agent schedule
        prob_forcing_agent = self.prob_forcing_agent - self.prob_forcing_agent_decrease_per_epoch * current_epoch
        if prob_forcing_agent > 0:
            # [n_batch, n_agent]
            mask_forcing_agent = torch.bernoulli(torch.ones_like(as_valid[:, 0]) * prob_forcing_agent).bool()
            mask_teacher_forcing |= mask_forcing_agent.unsqueeze(1) & as_valid

        # what-if motion prediction
        if gt_sdc:
            assert as_valid[:, :, 0].all()
            # mask_teacher_forcing[:, :, 0] = True
            mask_teacher_forcing[:, :, 0] |= as_valid[:, :, 0]

        return mask_teacher_forcing
