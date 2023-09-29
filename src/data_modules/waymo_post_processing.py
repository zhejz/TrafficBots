# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Tuple, List
from omegaconf import ListConfig
import torch
from torch import nn, Tensor


class WaymoPostProcessing(nn.Module):
    def __init__(
        self,
        k_pred: int,
        score_temperature: float,
        mpa_nms_thresh: ListConfig,
        mtr_nms_thresh: ListConfig,
        aggr_thresh: ListConfig,
        n_iter_em: int,
        use_ade: bool,
    ) -> None:
        """
        Args:
            score_temperature: set to > 0 to turn on
            mpa_nms_thresh, mtr_nms_thresh, aggr_thresh: list in meters
        """
        super().__init__()
        self.k_pred = k_pred
        self.score_temperature = score_temperature
        self.mpa_nms_thresh = list(mpa_nms_thresh)
        self.mtr_nms_thresh = list(mtr_nms_thresh)
        self.aggr_thresh = list(aggr_thresh)
        self.n_iter_em = n_iter_em
        self.use_ade = use_ade

    def forward(self, valid: Tensor, scores: Tensor, trajs: Tensor, agent_type: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            valid: [n_scene, n_agent]
            scores: [n_scene, n_agent, n_pred], not normalized!
            trajs: [n_scene, n_agent, n_pred, n_step_future, 2/4] (x,y, yaw...)
            agent_type: [n_scene, n_agent, 3]

        Returns: pred_dict
            "waymo_valid": [n_scene, n_step, n_agent]
            "waymo_trajs": [n_scene, n_step, n_agent, k_pred, 2]
            "waymo_scores": [n_scene, n_agent, k_pred], normalized prob.
            "waymo_yaw_bbox": [n_scene, n_step, n_agent, k_pred, 1] or None
            "waymo_spd": [n_scene, n_step, n_agent, k_pred, 1] or None
        """
        scores = scores / scores.sum(-1, keepdim=True)  # normalized to prob
        n_scene, n_agent, n_pred, n_step, d_traj = trajs.shape

        if n_pred > self.k_pred:
            if len(self.aggr_thresh) > 0:
                trajs, scores = self.traj_aggr(
                    trajs, scores, self.k_pred, self.aggr_thresh, self.n_iter_em, self.use_ade
                )
            elif len(self.mtr_nms_thresh) > 0:
                trajs, scores = self.mtr_nms(trajs, scores, self.k_pred, self.mtr_nms_thresh, self.use_ade, agent_type)
            else:
                trajs, scores = self.traj_topk(trajs, scores, self.k_pred)

        # ! manually scale scores if necessary: [n_scene, n_agent, n_pred]
        if len(self.mpa_nms_thresh) > 0:
            scores = self.mpa_nms(valid, trajs, scores, self.mpa_nms_thresh, self.use_ade, agent_type)
        if self.score_temperature > 0:
            scores = torch.softmax(torch.log(scores) / self.score_temperature, dim=-1)

        trajs = trajs.movedim(3, 1)  # [n_scene, n_step, n_agent, k_pred, :]
        pred_dict = {"waymo_trajs": trajs[..., :2]}
        if d_traj >= 3:
            pred_dict["waymo_yaw_bbox"] = trajs[..., 2:3]
        else:
            pred_dict["waymo_yaw_bbox"] = None
        if d_traj >= 4:
            pred_dict["waymo_spd"] = trajs[..., 3:4]
        else:
            pred_dict["waymo_spd"] = None
        pred_dict["waymo_scores"] = scores  # [n_scene, n_agent, k_pred]
        # [n_scene, n_step, n_agent]
        pred_dict["waymo_valid"] = valid.unsqueeze(1).expand(-1, n_step, -1)
        return pred_dict

    @staticmethod
    def mpa_nms(
        valid: Tensor, trajs: Tensor, scores: Tensor, type_thresh: List[float], use_ade: bool, agent_type: Tensor
    ) -> Tensor:
        """
        Args:
            valid: [n_scene, n_agent]
            trajs: [n_scene, n_agent, k_pred, n_step, 2]
            scores: [n_scene, n_agent, k_pred], normalized prob
            type_thresh: in meters, list, len=3 [veh, ped, cyc].
            agent_type: [n_scene, n_agent, 3], [veh, ped, cyc].

        Returns:
            scores: [n_scene, n_agent, k_pred], normalized prob
        """
        # ! type dependent thresh
        thresh = 0
        for i in range(len(type_thresh)):
            thresh += agent_type[:, :, i] * type_thresh[i]
        thresh = thresh[:, :, None, None]  # [n_scene, n_agent, 1, 1]
        # within_dist: [n_scene, n_agent, n_pred, n_pred]
        xy = trajs[..., :2]
        if use_ade:
            within_dist = (torch.norm(xy.unsqueeze(2) - xy.unsqueeze(3), dim=-1).mean(-1)) < thresh
        else:
            within_dist = torch.norm(xy[:, :, :, -1].unsqueeze(2) - xy[:, :, :, -1].unsqueeze(3), dim=-1) < thresh

        for i in range(valid.shape[0]):
            for j in range(valid.shape[1]):
                if valid[i, j]:
                    for k in scores[i, j].argsort(descending=True):
                        # [k_pred]
                        mask = within_dist[i, j, k] & (scores[i, j] > scores[i, j, k])
                        if mask.any():
                            scores[i, j, k] = 1e-3
        scores = scores / scores.sum(-1, keepdim=True)
        return scores

    @staticmethod
    def mtr_nms(
        trajs: Tensor, scores: Tensor, k_pred: int, type_thresh: float, use_ade: bool, agent_type: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            trajs: [n_scene, n_agent, n_pred, n_step, 2]
            scores: [n_scene, n_agent, n_pred], normalized prob
            k_pred: int
            type_thresh: in meters, list, len=3 [veh, ped, cyc].
            agent_type: [n_scene, n_agent, 3], [veh, ped, cyc].

        Returns:
            trajs_k: [n_scene, n_agent, k_pred, n_step, 2]
            scores_k: [n_scene, n_agent, k_pred], normalized prob
        """
        # ! type dependent thresh
        thresh = 0
        for i in range(len(type_thresh)):
            thresh += agent_type[:, :, i] * type_thresh[i]
        thresh = thresh[:, :, None, None]  # [n_scene, n_agent, 1, 1]
        # within_dist: [n_scene, n_agent, n_pred, n_pred]
        xy = trajs[..., :2]
        if use_ade:
            within_dist = (torch.norm(xy.unsqueeze(2) - xy.unsqueeze(3), dim=-1).mean(-1)) < thresh
        else:
            within_dist = torch.norm(xy[:, :, :, -1].unsqueeze(2) - xy[:, :, :, -1].unsqueeze(3), dim=-1) < thresh

        # ! compute mode_idx: [n_scene, n_agent, k_pred]
        scene_idx = torch.arange(scores.shape[0]).unsqueeze(1)  # [n_scene, 1]
        agent_idx = torch.arange(scores.shape[1]).unsqueeze(0)  # [1, n_agent]
        mode_idx = []
        scores_clone = scores.clone()
        for _ in range(k_pred):
            # [n_scene, n_agent]
            _idx = scores_clone.max(-1)[1]
            # [n_scene, n_agent, n_pred], True entry has w=0.01, False entry has w=1.0
            w_mask = ~(within_dist[scene_idx, agent_idx, _idx]) * 0.99 + 0.01
            # [n_scene, n_agent, n_pred], suppress all preds close to the selected one, by multiplying the prob by 0.1
            scores_clone *= w_mask
            scores_clone[scene_idx, agent_idx, _idx] = -1
            # append to mode_idx
            mode_idx.append(_idx)

        mode_idx = torch.stack(mode_idx, dim=-1)  # [n_scene, n_agent, k_pred]
        scene_idx = scene_idx.unsqueeze(-1)  # [n_scene, 1, 1]
        agent_idx = agent_idx.unsqueeze(-1)  # [1, n_agent, 1]
        trajs_k = trajs[scene_idx, agent_idx, mode_idx]
        scores_k = scores[scene_idx, agent_idx, mode_idx]
        scores_k = scores_k / scores_k.sum(-1, keepdim=True)
        return trajs_k, scores_k

    @staticmethod
    def traj_topk(trajs: Tensor, scores: Tensor, k_pred: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            trajs: [n_scene, n_agent, n_pred, n_step, 2]
            scores: [n_scene, n_agent, n_pred], normalized prob
            k_pred: int

        Returns:
            trajs_k: [n_scene, n_agent, k_pred, n_step, 2]
            scores_k: [n_scene, n_agent, k_pred], normalized prob
        """
        scene_idx = torch.arange(scores.shape[0])[:, None, None]  # [n_scene, 1, 1]
        agent_idx = torch.arange(scores.shape[1])[None, :, None]  # [1, n_agent, 1]
        mode_idx = scores.topk(k_pred, dim=-1, sorted=False)[1]  # [n_scene, n_agent, k_pred]
        trajs_k = trajs[scene_idx, agent_idx, mode_idx]
        scores_k = scores[scene_idx, agent_idx, mode_idx]

        scores_k = scores_k / scores_k.sum(-1, keepdim=True)
        return trajs_k, scores_k

    @staticmethod
    def traj_aggr(
        trajs: Tensor, scores: Tensor, k_pred: int, thresh: float, n_iter_em: int, use_ade: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            trajs: [n_scene, n_agent, n_pred, n_step, 2]
            scores: [n_scene, n_agent, n_pred], normalized prob
            k_pred: int
            thresh: in meters, if < 0, just topk scores
            n_iter_em: nubmer of iterations for k-means em
        Returns:
            trajs_k: [n_scene, n_agent, k_pred, n_step, 2]
            scores_k: [n_scene, n_agent, k_pred], normalized prob
        """
        scene_idx = torch.arange(scores.shape[0]).unsqueeze(1)  # [n_scene, 1]
        agent_idx = torch.arange(scores.shape[1]).unsqueeze(0)  # [1, n_agent]

        # ! compute mode_idx mpa: greedy k-means center
        # [n_scene, n_agent, n_pred, n_pred]
        # within_dist = torch.norm(trajs[:, :, :, -1].unsqueeze(2) - trajs[:, :, :, -1].unsqueeze(3), dim=-1) < thresh
        # mode_idx = []
        # for _ in range(k_pred):
        #     # [n_scene, n_agent]
        #     _idx = (scores.unsqueeze(2) * within_dist).sum(-1).max(-1)[1]
        #     # [n_scene, n_agent, n_pred], False at all nodes belonging to the selected center
        #     mask = ~(within_dist[scene_idx, agent_idx, _idx])
        #     # [n_scene, n_agent, n_pred, n_pred], remove selected nodes
        #     within_dist &= mask.unsqueeze(-1)
        #     # [n_scene, n_agent, n_pred, n_pred], set selected mode to 0 for remaning nodes
        #     within_dist &= mask.unsqueeze(-2)
        #     # append to mode_idx
        #     mode_idx.append(_idx)

        # ! compute mode_idx mtr: [n_scene, n_agent, k_pred]
        # [n_scene, n_agent, n_pred, n_pred]
        xy = trajs[..., :2]
        if use_ade:
            within_dist = (torch.norm(xy.unsqueeze(2) - xy.unsqueeze(3), dim=-1).mean(-1)) < thresh
        else:
            within_dist = torch.norm(xy[:, :, :, -1].unsqueeze(2) - xy[:, :, :, -1].unsqueeze(3), dim=-1) < thresh
        n_pred = scores.shape[-1]
        mode_idx = []
        scores_clone = scores.clone()
        for _ in range(k_pred):
            # [n_scene, n_agent]
            _idx = scores_clone.max(-1)[1]
            # [n_scene, n_agent, n_pred], True entry has w=0.1, False entry has w=1.0
            w_mask = ~(within_dist[scene_idx, agent_idx, _idx]) * 0.9 + 0.1
            # [n_scene, n_agent, n_pred], suppress all preds close to the selected one, by multiplying the prob by 0.1
            scores_clone *= w_mask
            # [n_scene, n_agent, n_pred], set prob to negative if selected
            w_mask_idx = nn.functional.one_hot(_idx, n_pred)
            scores_clone -= w_mask_idx
            # append to mode_idx
            mode_idx.append(_idx)

        mode_idx = torch.stack(mode_idx, dim=-1)  # [n_scene, n_agent, k_pred]
        scene_idx = scene_idx.unsqueeze(-1)  # [n_scene, 1, 1]
        agent_idx = agent_idx.unsqueeze(-1)  # [1, n_agent, 1]
        trajs_k = trajs[scene_idx, agent_idx, mode_idx]  # [n_scene, n_agent, k_pred, n_step, 2]
        scores_k = scores[scene_idx, agent_idx, mode_idx]  # [n_scene, n_agent, k_pred]

        # ! em for kmeans
        # trajs: [n_scene, n_agent, n_pred, n_step, 2]
        for _ in range(n_iter_em):
            # [n_scene, n_agent, n_pred, k_pred]
            xy_k = trajs_k[..., :2]
            xy = trajs[..., :2]
            if use_ade:
                dist = torch.norm(xy_k.unsqueeze(2) - xy.unsqueeze(3), dim=-1).mean(-1)
            else:
                dist = torch.norm(xy_k[:, :, :, -1].unsqueeze(2) - xy[:, :, :, -1].unsqueeze(3), dim=-1)
            # [n_scene, n_agent, n_pred]
            min_dist, assignment = dist.min(-1)
            # [n_scene, n_agent, n_pred, k_pred]
            assignment = nn.functional.one_hot(assignment, k_pred)
            empty_scene, empty_agent, empty_pred = torch.where(assignment.sum(2) == 0)

            if len(empty_scene) > 0:
                for i in range(len(empty_scene)):

                    # assign to furtherst
                    # max_i = min_dist[empty_scene[i], empty_agent[i]].argmax()
                    # min_dist[empty_scene[i], empty_agent[i], max_i] = -1
                    # assignment[empty_scene[i], empty_agent[i], max_i] = 0
                    # assignment[empty_scene[i], empty_agent[i], max_i, empty_pred[i]] = 1

                    # split
                    counter_n, max_i = assignment.sum(2)[empty_scene[i], empty_agent[i]].max(0)
                    _split = torch.where(assignment[empty_scene[i], empty_agent[i], :, max_i] == 1)[0][: counter_n // 2]
                    assignment[empty_scene[i], empty_agent[i], _split, max_i] = 0
                    assignment[empty_scene[i], empty_agent[i], _split, empty_pred[i]] = 1

            n_members = assignment.sum(2)  # [n_scene, n_agent, k_pred]
            # update trajs_k: [n_scene, n_agent, k_pred, n_step, 2]
            trajs_k = (trajs.unsqueeze(3) * assignment[:, :, :, :, None, None]).sum(2)  # sum over n_pred
            trajs_k /= n_members[:, :, :, None, None]
            # update scores_k: [n_scene, n_agent, k_pred]
            scores_k = (scores.unsqueeze(3) * assignment).sum(2) / n_members

        scores_k = scores_k / scores_k.sum(-1, keepdim=True)
        return trajs_k, scores_k
