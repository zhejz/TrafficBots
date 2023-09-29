# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple
from omegaconf import DictConfig
import hydra
from models.modules.distributions import MyDist
from utils.transform_utils import cast_rad


class Dynamics:
    def __init__(
        self, veh: DictConfig, ped: DictConfig, cyc: DictConfig, dt: float = 0.1, use_veh_dynamics_for_all: bool = False
    ) -> None:
        self.dt = dt
        self.action_dim = 2
        self.use_veh_dynamics_for_all = use_veh_dynamics_for_all
        self.state_keys = ["agent_state", "vel", "acc", "yaw_rate"]

        if self.use_veh_dynamics_for_all:
            self.agent_dynamics = hydra.utils.instantiate(veh, dt=dt)
        else:
            self.agent_dynamics = (
                hydra.utils.instantiate(veh, dt=dt),
                hydra.utils.instantiate(ped, dt=dt),
                hydra.utils.instantiate(cyc, dt=dt),
            )

    def init(
        self,
        agent_valid: Tensor,
        agent_state: Tensor,
        agent_size: Tensor,
        agent_type: Tensor,
        vel: Tensor,
        acc: Tensor,
        yaw_rate: Tensor,
    ) -> None:
        # constant
        self.agent_size = agent_size
        self.agent_type = agent_type
        # updating
        self.agent_valid = agent_valid  # [n_batch, n_agent], bool
        self.agent_killed = torch.zeros_like(agent_valid)  # [n_batch, n_agent], bool
        self.agent_state = agent_state  # [n_batch, n_agent, 4], x,y,yaw,v
        self.vel = vel  # [n_batch, n_agent, 2]
        self.acc = acc  # [n_batch, n_agent, 1]
        self.yaw_rate = yaw_rate  # [n_batch, n_agent, 1]

    def update(
        self,
        action_dist: MyDist,
        action_override: Optional[Tensor] = None,
        mask_action_override: Optional[Tensor] = None,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            action_dist: Gaussian or DiagGaussian, unbounded value range.
            action_override: [n_batch, n_agent, 2] in physical metrics.
            mask_action_override: [n_batch, n_agent] bool, for gym API.
        Updating attrs:
            self.agent_state: [n_batch, n_agent, 4], x,y,yaw,spd
            self.vel: [n_batch, n_agent, 2]
            self.acc: [n_batch, n_agent, 1]
            self.yaw_rate: [n_batch, n_agent, 1]

        Returns:
            agent_state: [n_batch, n_agent, 4], x,y,yaw,spd
            agent_valid: [n_batch, n_agent], bool
            action: [n_batch, n_agent, 2]
            action_log_prob: [n_batch, n_agent]
        """
        mask_type = ~self.agent_type  # [n_batch, n_agent, 3] reversed agent_type for masked_fill
        agent_invalid = ~self.agent_valid.unsqueeze(-1)
        # [n_batch, n_agent, 2], unbounded
        action_unbounded = action_dist.sample(deterministic)

        # [n_batch, n_agent]
        action_log_prob = action_dist.log_prob(action_unbounded.detach()).masked_fill(agent_invalid.squeeze(-1), 0)

        # action: unbounded value to value with physical metrics.
        if self.use_veh_dynamics_for_all:
            action = self.agent_dynamics.process_action(action_unbounded)
        else:
            action = 0
            # mask_type: [n_batch, n_agent, 3]
            for i in range(3):
                action += self.agent_dynamics[i].process_action(action_unbounded).masked_fill(mask_type[:, :, [i]], 0)
        action = action.masked_fill(agent_invalid, 0)

        # override actions: used for gym, for robot-vehicle. not used for now
        if action_override is not None:
            mask_action_override = mask_action_override & self.agent_valid  # [n_batch, n_agent], bool
            action = action.masked_fill(mask_action_override.unsqueeze(-1), 0)
            action[mask_action_override] = action_override[mask_action_override]  # physical metrics

        # dynamics update states using actions
        if self.use_veh_dynamics_for_all:
            state, vel, acc, yaw_rate = self.agent_dynamics.update(self.agent_state, action)
        else:
            state = 0  # [n_batch, n_agent, 4], x,y,yaw,spd
            vel = 0  # [n_batch, n_agent, 2]
            acc = 0  # [n_batch, n_agent, 1]
            yaw_rate = 0  # [n_batch, n_agent, 1]
            # mask_type: [n_batch, n_agent, 3]
            for i in range(3):
                _state, _vel, _acc, _yaw_rate = self.agent_dynamics[i].update(self.agent_state, action)
                state += _state.masked_fill(mask_type[:, :, [i]], 0)
                vel += _vel.masked_fill(mask_type[:, :, [i]], 0)
                acc += _acc.masked_fill(mask_type[:, :, [i]], 0)
                yaw_rate += _yaw_rate.masked_fill(mask_type[:, :, [i]], 0)

        # mask out invalid
        self.agent_state = state.masked_fill(agent_invalid, 0)
        self.agent_vel = vel.masked_fill(agent_invalid, 0)
        self.agent_acc = acc.masked_fill(agent_invalid, 0)
        self.agent_yaw_rate = yaw_rate.masked_fill(agent_invalid, 0)
        return self.agent_state, self.agent_valid, action, action_log_prob

    def override_states(
        self, state_override: Optional[Dict[str, Tensor]] = None, mask_state_override: Optional[Tensor] = None
    ) -> None:
        """For teacher forcing (existing agent) and spawn (new agent).
        Args:
            mask_state_override: [n_batch, n_agent] bool
            state_override["agent_state"]: [n_batch, n_agent, 4]
            state_override["vel]: [n_batch, n_agent, 2]
            state_override["acc"]: [n_batch, n_agent, 1]
            state_override["yaw_rate]: [n_batch, n_agent, 1]
        """
        if mask_state_override is not None:
            mask_state_override = mask_state_override & (~self.agent_killed)
            if mask_state_override.any():
                self.agent_valid = self.agent_valid | mask_state_override
                mask_state_override = mask_state_override.unsqueeze(-1)
                mask_old_state = ~mask_state_override
                if "agent_state" in state_override:
                    self.agent_state = self.agent_state.masked_fill(mask_state_override, 0)
                    self.agent_state = self.agent_state + state_override["agent_state"].masked_fill(mask_old_state, 0)
                if "vel" in state_override:
                    self.vel = self.vel.masked_fill(mask_state_override, 0)
                    self.vel = self.vel + state_override["vel"].masked_fill(mask_old_state, 0)
                if "acc" in state_override:
                    self.acc = self.acc.masked_fill(mask_state_override, 0)
                    self.acc = self.acc + state_override["acc"].masked_fill(mask_old_state, 0)
                if "yaw_rate" in state_override:
                    self.yaw_rate = self.yaw_rate.masked_fill(mask_state_override, 0)
                    self.yaw_rate = self.yaw_rate + state_override["yaw_rate"].masked_fill(mask_old_state, 0)

    @torch.no_grad()
    def kill(self, traffic_rule_violations: Dict[str, Tensor], gt_valid: Optional[Tensor] = None) -> None:
        """
        Args:
            traffic_rule_violations: at t
            gt_valid: [n_batch, n_agent] at t

        Update: will take affect at next update(), at t
            self.as_valid: [n_batch, n_agent], bool
        """
        mask_kill = traffic_rule_violations["outside_map_this_step"]
        if gt_valid is not None:  # do not kill agent that has gt_valid, such that the value can be computed at t+1.
            mask_kill = mask_kill & (~gt_valid)
        if mask_kill.any():
            self.agent_killed = self.agent_killed | mask_kill
            mask_survive = ~mask_kill
            self.agent_valid = self.agent_valid & mask_survive


class MultiPathPP:
    def __init__(self, dt: float, max_acc: float = 4, max_yaw_rate: float = 1, disable_neg_spd: bool = False) -> None:
        """
        max_acc:
        max_yaw_rate: veh=1rad/s, cyc=2.5rad/s, ped=5rad/s
        max_yaw_rate: veh=1.2rad/s, cyc=3rad/s, ped=6rad/s
        delta_theta per step (0.1sec), veh: 5m, 0.1rad, cyc: 2m, 0.25rad, ped:1m, 0.5rad
        """
        self.dt = dt
        self._max_acc = max_acc
        self._max_yaw_rate = max_yaw_rate
        self.disable_neg_spd = disable_neg_spd

    def init(self, *args, **kwargs) -> None:
        """Set prarameters for vehicle dynamics, called before each rollout."""
        pass

    def process_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: [n_batch, n_agent, 2] unbounded sample from Gaussian
        Returns:
            action: [n_batch, n_agent, 2], acc (m/s2), theta_rad (rad/s)
        """
        action = torch.tanh(action)
        action = torch.stack([action[..., 0] * self._max_acc, action[..., 1] * self._max_yaw_rate], dim=-1)
        return action

    def update(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            state: [n_batch, n_agent, 4] x,y,theta,spd
            action: [n_batch, n_agent, 2] acc(m/s2), yaw_rate(rad/s)

        Returns:
            as_pred: [n_batch, n_agent, 4], x,y,theta,spd
            vel: [n_batch, n_agent, 2]
            acc: [n_batch, n_agent, 1]
            yaw_rate [n_batch, n_agent, 1]
        """
        acc = action[:, :, 0]
        yaw_rate = action[:, :, 1]

        # [n_batch, n_agent]
        v_tilde = state[:, :, 3] + 0.5 * self.dt * acc
        theta_tilde = state[:, :, 2] + 0.5 * self.dt * yaw_rate
        cos_theta_tilde = torch.cos(theta_tilde)
        sin_theta_tilde = torch.sin(theta_tilde)

        # [n_batch, n_agent, 4]
        delta_state = torch.stack([v_tilde * cos_theta_tilde, v_tilde * sin_theta_tilde, yaw_rate, acc], dim=-1)
        as_pred = state + self.dt * delta_state
        if self.disable_neg_spd:
            as_pred[..., -1] = torch.relu(state[..., -1])

        vel = (as_pred[:, :, :2] - state[:, :, :2]) / self.dt
        acc = acc.unsqueeze(-1)
        yaw_rate = yaw_rate.unsqueeze(-1)
        return as_pred, vel, acc, yaw_rate


class StateIntegrator:
    def __init__(self, dt: float, max_v: float = 3) -> None:
        self.dt = dt
        self._max_v = max_v  # ped=3m/s

    def init(self, *args, **kwargs) -> None:
        pass

    def process_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: [n_batch, n_agent, 2] unbounded sample from Gaussian
        Returns:
            action: [n_batch, n_agent, 2], vx (m/s), vy (m/s)
        """
        action = torch.tanh(action) * self._max_v
        return action

    def update(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            state: [n_batch, n_agent, 4] x,y,theta,spd
            action: [n_batch, n_agent, 2] vx,vy in m/s

        Returns:
            as_pred: [n_batch, n_agent, 4], x,y,theta,spd
            vel: [n_batch, n_agent, 2]
            acc: [n_batch, n_agent, 1]
            yaw_rate [n_batch, n_agent, 1]
        """
        mask_theta_v = torch.zeros_like(state, dtype=torch.bool)
        mask_theta_v[:, :, 2:] = True

        # [n_batch, n_agent]
        vx = action[:, :, 0]
        vy = action[:, :, 1]
        theta = torch.atan2(vy, vx).detach()
        spd = torch.norm(action, dim=-1).detach()

        # [n_batch, n_agent, 4]
        delta_state = torch.stack([vx * self.dt, vy * self.dt, theta, spd], dim=-1)
        as_pred = state.masked_fill(mask_theta_v, 0) + delta_state

        acc = (spd - state[:, :, 3]) / self.dt
        acc = acc.unsqueeze(-1).detach()
        yaw_rate = cast_rad(theta - state[:, :, 2]) / self.dt
        yaw_rate = yaw_rate.unsqueeze(-1).detach()
        return as_pred, action, acc, yaw_rate
