# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Tuple, Optional
from torch import Tensor
import numpy as np
import torch
from utils.transform_utils import cast_rad


class TrafficRuleChecker:
    def __init__(
        self,
        map_boundary: Tensor,
        map_valid: Tensor,
        map_type: Tensor,
        map_pos: Tensor,
        map_dir: Tensor,
        tl_stop_valid: Tensor,
        tl_stop_pos: Tensor,
        tl_stop_state: Tensor,
        agent_type: Tensor,
        agent_size: Tensor,
        agent_goal: Optional[Tensor],
        agent_dest: Optional[Tensor],
        enable_check_collided: bool,
        enable_check_run_road_edge: bool,
        enable_check_run_red_light: bool,
        enable_check_passive: bool,
        collision_size_scale: float = 1.1,
    ) -> None:
        self.agent_size = agent_size[..., :2] * collision_size_scale
        self.map_boundary = map_boundary
        self.tl_stop_valid = tl_stop_valid
        self.tl_stop_pos = tl_stop_pos
        self.tl_stop_state = tl_stop_state
        self.agent_goal = agent_goal
        self.agent_dest = agent_dest

        self.enable_check_collided = enable_check_collided
        self.enable_check_run_road_edge = enable_check_run_road_edge
        self.enable_check_run_red_light = enable_check_run_red_light
        self.enable_check_passive = enable_check_passive

        mask = agent_type[:, :, 0]
        n_batch, n_agent = mask.shape
        self.outside_map = torch.zeros_like(mask)
        self.collided = torch.zeros_like(mask)
        self.run_red_light = torch.zeros_like(mask)
        self.goal_reached = torch.zeros_like(mask)
        self.dest_reached = torch.zeros_like(mask)
        self.run_road_edge = torch.zeros_like(mask)
        self.passive = torch.zeros_like(mask)
        self.passive_counter = torch.zeros_like(mask, dtype=torch.float32)

        # for self._check_collided
        # [n_batch, n_agent, n_agent]: no self collision
        self.agent_ego_mask = torch.eye(n_agent, dtype=torch.bool, device=mask.device)[None, :, :].expand(
            n_batch, -1, -1
        )
        ped_cyc_mask = agent_type[:, :, 1]
        collision_ped_cyc_mask = ped_cyc_mask.unsqueeze(1) & ped_cyc_mask.unsqueeze(2)
        self.collision_invalid_mask = self.agent_ego_mask | collision_ped_cyc_mask

        # for self._check_run_road_edge
        self.road_edge, self.road_edge_valid = self._get_road_edge(map_valid, map_type, map_pos, map_dir)

        # for self._check_run_red_light
        # [n_batch, n_agent, 1]
        self.run_red_light_agent_length = agent_size[:, :, [0]] * 0.5 * 0.6
        self.run_red_light_agent_width = agent_size[:, :, [1]] * 0.5 * 1.8
        self.veh_mask = agent_type[:, :, 0]  # [n_batch, n_agent]

        # for self._check_passive
        self.lane_center, self.lane_center_valid = self._get_lane_center(map_valid, map_type, map_pos)

        # for self._check_goal_reached
        # [n_batch, n_agent]
        if self.agent_dest is not None:
            self.goal_thresh_pos = agent_size[:, :, 0] * 8
            self.goal_thresh_rot = np.deg2rad(15)

        # for self._check_dest_reached
        if self.agent_dest is not None:
            # [n_batch, 1]
            batch_idx = torch.arange(map_valid.shape[0]).unsqueeze(1)
            # [n_batch, n_agent, 20]
            self.dest_valid = map_valid[batch_idx, agent_dest]
            # [n_batch, n_agent, n_type]: one_hot bool, LANE<=3, TYPE_ROAD_EDGE_BOUNDARY = 4
            self.dest_type = map_type[batch_idx, agent_dest]
            # [n_batch, n_agent, 20, 2]
            self.dest_pos = map_pos[batch_idx, agent_dest]
            # [n_batch, n_agent, 20, 2]
            self.dest_dir = map_dir[batch_idx, agent_dest]
            self.dest_dir = self.dest_dir / torch.norm(self.dest_dir, dim=-1, keepdim=True)
            # dest_thresh_rot only valid for lane dests
            self.dest_thresh_rot = np.deg2rad(30)
            # [n_batch, n_agent]: thresh_lane=50, thresh_edge=10
            self.dest_thresh_pos = torch.ones_like(agent_size[:, :, 0]) * 50
            self.dest_thresh_pos = self.dest_thresh_pos * (1 - self.dest_type[:, :, 4] * 0.8)

    @staticmethod
    def _check_outside_map(as_valid: Tensor, as_state: Tensor, map_boundary: Tensor) -> Tensor:
        """
        Args:
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
            map_boundary: [n_batch, 4], xmin,xmax,ymin,ymax
        Returns:
            outside_map_this_step: [n_batch, n_agent], bool
        """
        # [n_batch, n_agent]
        x = as_state[:, :, 0]
        y = as_state[:, :, 1]
        # [n_batch, 1]
        xmin = map_boundary[:, [0]]
        xmax = map_boundary[:, [1]]
        ymin = map_boundary[:, [2]]
        ymax = map_boundary[:, [3]]
        outside_map_this_step = ((x > xmax) | (x < xmin) | (y > ymax) | (y < ymin)) & as_valid
        return outside_map_this_step

    @staticmethod
    def _check_collided(as_valid: Tensor, agent_bbox: Tensor, collision_invalid_mask: Tensor) -> Tensor:
        """
        Args:
            as_valid: [n_batch, n_agent] bool
            agent_bbox: [n_batch, n_agent, 4, 2], 4 corners, (x,y)
            collision_invalid_mask: [n_batch, n_agent, n_agent]: no self collision
        Returns:
            collided_this_step: [n_batch, n_agent], bool
        """
        bbox_next = agent_bbox.roll(-1, dims=2)

        # [n_batch, n_agent, 4, 3]
        bbox_line = torch.cat(  # ax+by+c=0
            [
                bbox_next[..., [1]] - agent_bbox[..., [1]],  # a
                agent_bbox[..., [0]] - bbox_next[..., [0]],  # b
                bbox_next[..., [0]] * agent_bbox[..., [1]] - bbox_next[..., [1]] * agent_bbox[..., [0]],
            ],  # c
            axis=-1,
        )
        bbox_point = torch.cat([agent_bbox, torch.ones_like(agent_bbox[..., [0]])], axis=-1)

        # [n_batch, n_agent, n_agent, 4, 4, 3]
        n_agent = agent_bbox.shape[1]
        bbox_line = bbox_line[:, :, None, :, None, :].expand(-1, -1, n_agent, -1, 4, -1)
        bbox_point = bbox_point[:, None, :, None, :, :].expand(-1, n_agent, -1, 4, -1, -1)

        # [n_batch, n_agent, n_agent, 4, 4]
        is_outside = torch.sum(bbox_line * bbox_point, axis=-1) > 0

        # [n_batch, n_agent, n_agent]
        no_collision = torch.any(torch.all(is_outside, axis=-1), axis=-1)
        no_collision = no_collision | no_collision.transpose(1, 2)

        # [n_batch, n_agent, n_agent]: no collision for invalid agent
        invalid_mask = ~(as_valid[:, :, None] & as_valid[:, None, :])
        no_collision = no_collision | collision_invalid_mask | invalid_mask
        collided_this_step = ~(no_collision.all(-1))
        return collided_this_step

    @staticmethod
    def _check_run_road_edge(
        as_valid: Tensor, agent_bbox: Tensor, veh_mask: Tensor, road_edge: Tensor, road_edge_valid: Tensor
    ) -> Tensor:
        """Check vehicles only.
            as_valid: [n_batch, n_agent] bool
            agent_bbox: [n_batch, n_agent, 4, 2], 4 corners, (x,y)
            veh_mask: [n_batch, n_agent] bool
            road_edge: [n_batch, n_pl*20, 2, 2], (start/end), (x,y)
            road_edge_valid: [n_batch, n_pl*20], bool
        Returns:
            run_road_edge_this_step: [n_batch, n_agent], bool
        """
        bbox_next = agent_bbox.roll(-1, dims=2)  # [n_batch, n_agent, 4, 2]
        # [n_batch, n_agent, 1, 4, 2, 2]
        bbox_line = torch.stack([agent_bbox, bbox_next], dim=-2).unsqueeze(2)

        # [n_batch, n_pl*20, 2, 2] -> [n_batch, 1, n_pl*20, 1, 2, 2]
        road_edge_line = road_edge[:, None, :, None, :, :]

        # [n_batch, n_agent, n_pl*20, 4, 2]
        A = bbox_line[:, :, :, :, 0]
        B = bbox_line[:, :, :, :, 1]
        C = road_edge_line[:, :, :, :, 0]
        D = road_edge_line[:, :, :, :, 1]

        # [n_batch, n_agent, n_pl*20, 4]
        run_road_edge_this_step = (ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D))

        # [n_batch, n_agent, n_pl*20]
        run_road_edge_this_step = run_road_edge_this_step.any(-1) & road_edge_valid.unsqueeze(1)

        # [n_batch, n_agent]
        run_road_edge_this_step = run_road_edge_this_step.any(-1) & as_valid & veh_mask
        return run_road_edge_this_step

    @staticmethod
    def _check_run_red_light(
        as_valid: Tensor,
        as_state: Tensor,
        tl_valid: Tensor,
        tl_pos: Tensor,
        tl_state: Tensor,
        run_red_light_agent_length: Tensor,
        run_red_light_agent_width: Tensor,
        veh_mask: Tensor,
    ) -> Tensor:
        """Check up to step_gt
        Args:
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
            tl_valid: [n_batch, n_tl]
            tl_pos: [n_batch, n_tl, 2] xy of stop point
            tl_state: [n_batch, n_tl, n_tl_state=5] bool one_hot
            run_red_light_agent_length: [n_batch, n_agent, 1]
            run_red_light_agent_width: [n_batch, n_agent, 1]
            veh_mask: [n_batch, n_agent]
            # LANE_STATE_UNKNOWN = 0;
            # LANE_STATE_STOP = 1;
            # LANE_STATE_CAUTION = 2;
            # LANE_STATE_GO = 3;
            # LANE_STATE_FLASHING = 4;
        Returns:
            run_red_light_this_step: [n_batch, n_agent], bool
        """
        # [n_batch, n_agent]
        heading_cos = torch.cos(as_state[..., 2])
        heading_sin = torch.sin(as_state[..., 2])
        # [n_batch, n_agent, 1, 2]
        agent_heading_f = torch.stack([heading_cos, heading_sin], axis=-1).unsqueeze(2)
        agent_heading_r = torch.stack([heading_sin, -heading_cos], axis=-1).unsqueeze(2)

        # [n_batch, n_agent, 1, 2]
        agent_xy_0 = as_state[..., :2].unsqueeze(2)
        agent_xy_1 = agent_xy_0 + 0.1 * as_state[..., [3]].unsqueeze(2) * agent_heading_f

        # [n_batch, n_tl, 2] -> [n_batch, 1, n_tl, 2]
        tl_pos = tl_pos.unsqueeze(1)
        # [n_batch, n_agent, n_tl]
        inside_0 = torch.logical_and(
            torch.abs(torch.sum((tl_pos - agent_xy_0) * agent_heading_f, dim=-1)) < run_red_light_agent_length,
            torch.abs(torch.sum((tl_pos - agent_xy_0) * agent_heading_r, dim=-1)) < run_red_light_agent_width,
        )
        inside_1 = torch.logical_and(
            torch.abs(torch.sum((tl_pos - agent_xy_1) * agent_heading_f, dim=-1)) < run_red_light_agent_length,
            torch.abs(torch.sum((tl_pos - agent_xy_1) * agent_heading_r, dim=-1)) < run_red_light_agent_width,
        )

        # [n_batch, n_agent, 1]
        mask_valid_agent = (as_valid & veh_mask).unsqueeze(2)
        # [n_batch, 1, n_tl]
        mask_valid_tl = (tl_valid & tl_state[:, :, 1]).unsqueeze(1)
        # [n_batch, n_agent, n_tl]
        run_red_light_this_step = inside_0 & (~inside_1) & mask_valid_agent & mask_valid_tl
        # [n_batch, n_agent]
        run_red_light_this_step = run_red_light_this_step.any(-1)
        return run_red_light_this_step

    @staticmethod
    def _check_passive(
        as_valid: Tensor,
        as_state: Tensor,
        passive_counter: Tensor,
        tl_valid: Tensor,
        tl_pos: Tensor,
        tl_state: Tensor,
        lane_center: Tensor,
        lane_center_valid: Tensor,
        veh_mask: Tensor,
        agent_ego_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Check vehicles only.
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
            passive_counter: [n_batch, n_agent], float32
            tl_valid: [n_batch, n_tl]
            tl_pos: [n_batch, n_tl, 2] xy of stop point
            tl_state: [n_batch, n_tl, n_tl_state=5] bool one_hot
            lane_center: [n_batch, n_pl*20, 2]
            lane_center_valid: [n_batch, n_pl*20]
            veh_mask: [n_batch, n_agent] bool
            agent_ego_mask: [n_batch, n_agent, n_agent] bool, eye
        Returns:
            passive_this_step: [n_batch, n_agent], bool
            passive_counter: [n_batch, n_agent], float32
        """
        # [n_batch, n_agent, 1, 2] - [n_batch, 1, n_pl*20, 2] = [n_batch, n_agent, n_pl*20, 2]
        close_to_lane = torch.norm(as_state[:, :, :2].unsqueeze(2) - lane_center.unsqueeze(1), dim=-1)
        # [n_batch, n_agent, n_pl*20] bool
        close_to_lane = close_to_lane < 2  # m
        # [n_batch, n_agent]: [n_batch, n_agent, n_pl*20] & [n_batch, 1, n_pl*20]
        close_to_lane = (close_to_lane & lane_center_valid.unsqueeze(1)).any(-1)

        # [n_batch, n_agent]
        low_speed = as_state[:, :, 3] < 5  # m/s

        # [n_batch, n_agent, 1, 2]
        agent_heading_f = torch.stack([torch.cos(as_state[..., 2]), torch.sin(as_state[..., 2])], axis=-1).unsqueeze(2)

        # LANE_STATE_UNKNOWN = 0;
        # LANE_STATE_STOP = 1;
        # LANE_STATE_CAUTION = 2;
        # LANE_STATE_GO = 3;
        # LANE_STATE_FLASHING = 4;
        # check red (flashing,yellow) light ahead
        # [n_batch, 1, n_tl]
        mask_valid_tl = (tl_valid & tl_state[:, :, [0, 1, 2, 4]].any(-1)).unsqueeze(1)
        # [n_batch, n_agent, n_tl, 2]
        tl_vec = tl_pos.unsqueeze(1) - as_state[:, :, :2].unsqueeze(2)
        tl_vec_norm = torch.norm(tl_vec, dim=-1)
        # [n_batch, n_agent, n_tl]
        tl_is_close = tl_vec_norm < 10  # meter
        tl_is_ahead = ((agent_heading_f * tl_vec).sum(-1) / tl_vec_norm) > 0.95  # np.cos(np.deg2rad(18)) = 0.95
        # [n_batch, n_agent]
        red_tl_ahead = (tl_is_close & tl_is_ahead & mask_valid_tl).any(-1)

        # check other agents
        # [n_batch, n_agent, n_agent, 2]
        agent_vec = as_state[:, :, :2].unsqueeze(1) - as_state[:, :, :2].unsqueeze(2)
        agent_vec_norm = torch.norm(agent_vec, dim=-1)
        # [n_batch, n_agent, n_agent]
        agent_is_close = agent_vec_norm < 10  # meter
        agent_is_ahead = ((agent_heading_f * agent_vec).sum(-1) / agent_vec_norm) > 0.95
        # [n_batch, n_agent]
        agent_ahead = (
            agent_is_close & agent_is_ahead & as_valid.unsqueeze(1) & as_valid.unsqueeze(2) & (~agent_ego_mask)
        ).any(-1)

        passive_this_step = as_valid & veh_mask & close_to_lane & low_speed & (~red_tl_ahead) & (~agent_ahead)

        # accumulate, set to zero if not passive
        passive_counter = (passive_counter + passive_this_step) * passive_this_step
        passive_this_step = passive_counter > 20
        return passive_this_step, passive_counter

    @staticmethod
    def _check_goal_reached(
        as_valid: Tensor,
        as_state: Tensor,
        goal: Tensor,
        goal_reached: Tensor,
        goal_thresh_pos: Tensor,
        goal_thresh_rot: float,
    ) -> Tensor:
        """
        Args:
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
            goal: [n_batch, n_agent, 4], x,y,theta,v
            goal_reached: [n_batch, n_agent], bool
            goal_thresh_pos: [n_batch, n_agent] agent_length * 8
            goal_thresh_rot: float, 15 rad
        Returns:
            goal_reached_this_step: [n_batch, n_agent], bool
        """
        # [n_batch, n_agent]
        pos_reached = torch.norm(as_state[..., :2] - goal[..., :2], dim=-1) < goal_thresh_pos
        rot_reached = torch.abs(cast_rad(as_state[..., 2] - goal[..., 2])) < goal_thresh_rot
        goal_reached_this_step = pos_reached & rot_reached & as_valid & (~goal_reached)
        return goal_reached_this_step

    @staticmethod
    def _check_dest_reached(
        as_valid: Tensor,
        as_state: Tensor,
        dest_valid: Tensor,
        dest_type: Tensor,
        dest_pos: Tensor,
        dest_dir: Tensor,
        dest_reached: Tensor,
        dest_thresh_pos: Tensor,
        dest_thresh_rot: float,
    ) -> Tensor:
        """
        Args:
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
            dest_valid: [n_batch, n_agent, 20]
            dest_type: [n_batch, n_agent, n_pl_type] one_hot bool, LANE<=3, ROAD_EDGE = 4
            dest_pos: [n_batch, n_agent, 20, 2]
            dest_dir: [n_batch, n_agent, 20, 2], unit_vec
            dest_reached: [n_batch, n_agent], bool
            dest_thresh_pos: [n_batch, n_agent]
            dest_thresh_rot: float, in rad
        Returns:
            dest_reached_this_step: [n_batch, n_agent], bool
        """
        # [n_batch, n_agent, 20]
        dist_to_dest = torch.norm(as_state[..., :2].unsqueeze(2) - dest_pos, dim=-1).masked_fill(~dest_valid, 1e4)
        # [n_batch, n_agent]
        pos_reached = (dist_to_dest < dest_thresh_pos.unsqueeze(-1)).any(-1)

        heading_cos = torch.cos(as_state[..., 2])  # [n_batch, n_agent]
        heading_sin = torch.sin(as_state[..., 2])  # [n_batch, n_agent]
        agent_heading_f = torch.stack([heading_cos, heading_sin], axis=-1)  # [n_batch, n_agent, 2]

        # [n_batch, n_agent, 1, 2] * [n_batch, n_agent, 20, 2]
        # [n_batch, n_agent, 20]
        rot_diff_to_dest = (agent_heading_f.unsqueeze(2) * dest_dir).sum(-1).masked_fill(~dest_valid, 0)
        # [n_batch, n_agent]
        rot_reached = (rot_diff_to_dest > np.cos(dest_thresh_rot)).any(-1)

        # [n_batch, n_agent]: one_hot bool, LANE<=3, TYPE_ROAD_EDGE_BOUNDARY = 4
        mask_lane = dest_type[:, :, :4].any(-1)
        mask_edge = dest_type[:, :, 4]
        dest_reached_this_step = (
            (~dest_reached) & as_valid & ((mask_lane & pos_reached & rot_reached) | (mask_edge & pos_reached))
        )
        return dest_reached_this_step

    @torch.no_grad()
    def check(self, step: int, as_valid: Tensor, as_state: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            as_valid: [n_batch, n_agent] bool
            as_state: [n_batch, n_agent, 4] x,y,theta,v
        Returns: for this current step
            violations: Dict {str -> Tensor [n_batch, n_agent]}
        """
        agent_bbox = self._get_agent_bbox(as_state, self.agent_size)

        outside_map_this_step = self._check_outside_map(as_valid, as_state, self.map_boundary)
        self.outside_map = self.outside_map | outside_map_this_step

        if self.enable_check_collided:
            collided_this_step = self._check_collided(as_valid, agent_bbox, self.collision_invalid_mask)
            self.collided = self.collided | collided_this_step
        else:
            collided_this_step = self.collided

        if self.enable_check_run_road_edge:
            run_road_edge_this_step = self._check_run_road_edge(
                as_valid, agent_bbox, self.veh_mask, self.road_edge, self.road_edge_valid
            )
            self.run_road_edge = self.run_road_edge | run_road_edge_this_step
        else:
            run_road_edge_this_step = self.run_road_edge

        # step can be larger than ground truth tl_step
        if self.enable_check_run_red_light:
            tl_step = min(step, self.tl_stop_valid.shape[1] - 1)
            run_red_light_this_step = self._check_run_red_light(
                as_valid,
                as_state,
                self.tl_stop_valid[:, tl_step],
                self.tl_stop_pos[:, tl_step],
                self.tl_stop_state[:, tl_step],
                self.run_red_light_agent_length,
                self.run_red_light_agent_width,
                self.veh_mask,
            )
            self.run_red_light = self.run_red_light | run_red_light_this_step
        else:
            run_red_light_this_step = self.run_red_light

        if self.enable_check_passive:
            passive_this_step, self.passive_counter = self._check_passive(
                as_valid,
                as_state,
                self.passive_counter,
                self.tl_stop_valid[:, tl_step],
                self.tl_stop_pos[:, tl_step],
                self.tl_stop_state[:, tl_step],
                self.lane_center,
                self.lane_center_valid,
                self.veh_mask,
                self.agent_ego_mask,
            )
            self.passive = self.passive | passive_this_step
        else:
            passive_this_step = self.passive

        if self.agent_goal is None:
            goal_reached_this_step = torch.zeros_like(self.goal_reached)
        else:
            goal_reached_this_step = self._check_goal_reached(
                as_valid, as_state, self.agent_goal, self.goal_reached, self.goal_thresh_pos, self.goal_thresh_rot
            )
        self.goal_reached = self.goal_reached | goal_reached_this_step

        if self.agent_dest is None:
            dest_reached_this_step = torch.zeros_like(self.dest_reached)
        else:
            dest_reached_this_step = self._check_dest_reached(
                as_valid,
                as_state,
                self.dest_valid,
                self.dest_type,
                self.dest_pos,
                self.dest_dir,
                self.dest_reached,
                self.dest_thresh_pos,
                self.dest_thresh_rot,
            )
        self.dest_reached = self.dest_reached | dest_reached_this_step

        # [n_batch, n_agent], bool
        violations = {
            # "agent_to_kill": outside_map_this_step,
            "outside_map": self.outside_map,
            "outside_map_this_step": outside_map_this_step,
            "collided": self.collided,  # no collision ped2ped
            "collided_this_step": collided_this_step,
            "run_road_edge": self.run_road_edge,  # only for vehicles
            "run_road_edge_this_step": run_road_edge_this_step,
            "run_red_light": self.run_red_light,  # only for vehicles
            "run_red_light_this_step": run_red_light_this_step,
            "passive": self.passive,  # only for vehicles
            "passive_this_step": passive_this_step,
            "goal_reached": self.goal_reached,
            "goal_reached_this_step": goal_reached_this_step,
            "dest_reached": self.dest_reached,
            "dest_reached_this_step": dest_reached_this_step,
        }
        return violations

    @staticmethod
    def _get_agent_bbox(agent_states: Tensor, agent_size: Tensor) -> Tensor:
        """
        Args:
            agent_states: [n_batch, n_agent, 4] x,y,theta,v
            agent_size: [n_batch, n_agent, 2], length, width
        Returns:
            agent_bbox: [n_batch, n_agent, 4, 2]
        """
        heading_cos = torch.cos(agent_states[..., 2])  # [n_batch, n_agent]
        heading_sin = torch.sin(agent_states[..., 2])  # [n_batch, n_agent]
        agent_heading_f = torch.stack([heading_cos, heading_sin], axis=-1)  # [n_batch, n_agent, 2]
        agent_heading_r = torch.stack([heading_sin, -heading_cos], axis=-1)  # [n_batch, n_agent, 2]
        offset_forward = 0.5 * agent_size[..., [0]].expand(-1, -1, 2) * agent_heading_f  # [n_batch, n_agent, 2]
        offset_right = 0.5 * agent_size[..., [1]].expand(-1, -1, 2) * agent_heading_r  # [n_batch, n_agent, 2]
        vertex_offset = torch.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            dim=2,
        )
        agent_bbox = agent_states[:, :, None, :2].expand(-1, -1, 4, -1) + vertex_offset
        return agent_bbox

    # FREEWAY = 0
    # SURFACE_STREET = 1
    # STOP_SIGN = 2
    # BIKE_LANE = 3
    # TYPE_ROAD_EDGE_BOUNDARY = 4
    # TYPE_ROAD_EDGE_MEDIAN = 5
    # SOLID_SINGLE = 6
    # SOLID_DOUBLE = 7
    # PASSING_DOUBLE_YELLOW = 8
    # SPEED_BUMP = 9
    # CROSSWALK = 10

    @staticmethod
    def _get_road_edge(map_valid: Tensor, map_type: Tensor, map_pos: Tensor, map_dir: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            map_valid: [n_batch, n_pl, 20]
            map_type: [n_batch, n_pl, 11] one_hot bool
            map_pos: [n_batch, n_pl, 20, 2]
            map_dir: [n_batch, n_pl, 20, 2]
        Returns:
            road_edge: [n_batch, n_pl*20, 2, 2], (start/end), (x,y)
            road_edge_valid: [n_batch, n_pl*20], bool
        """
        # [n_batch, n_pl, 20]
        road_edge_valid = map_valid & map_type[:, :, [4, 5, 7]].any(dim=-1, keepdim=True)
        road_edge_valid = road_edge_valid.flatten(1, 2)  # [n_batch, n_pl*20]
        # [n_batch, n_pl, 20, 2] -> [n_batch, n_pl, 20, 2, 2] -> [n_batch, n_pl*20, 2, 2]
        road_edge = torch.stack([map_pos, map_pos + map_dir], dim=-2).flatten(1, 2)
        return road_edge, road_edge_valid

    @staticmethod
    def _get_lane_center(map_valid: Tensor, map_type: Tensor, map_pos: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            map_valid: [n_batch, n_pl, 20]
            map_type: [n_batch, n_pl]
            map_pos: [n_batch, n_pl, 20, 2]
        Returns:
            lane_center: [n_batch, n_pl*20, 2]
            lane_center_valid: [n_batch, n_pl*20]
        """
        lane_center_valid = map_type[:, :, :3].any(dim=-1, keepdim=True)  # [n_batch, n_pl, 1]
        lane_center_valid = map_valid & lane_center_valid  # [n_batch, n_pl, 20]
        lane_center_valid = lane_center_valid.flatten(1, 2)  # [n_batch, n_pl*20]
        lane_center = map_pos.flatten(1, 2)
        return lane_center, lane_center_valid


def ccw(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
