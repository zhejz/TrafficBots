# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from utils.pose_pe import PosePE
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_dir2local, torch_rad2local


class SceneCentricLatent(nn.Module):
    def __init__(
        self,
        time_step_current: int,
        data_size: DictConfig,
        perturb_input_to_latent: bool,  # random transform input to latent encoder at training time
        dropout_p_history: float,
        pe_dim: int,
        pose_pe: DictConfig,
        max_meter: float = 50.0,
        max_rad: float = 3.14,
    ) -> None:
        super().__init__()
        self.perturb_input_to_latent = perturb_input_to_latent
        self.dropout_p_history = dropout_p_history  # [0, 1], turn off if set to negative
        self.n_step_hist = time_step_current + 1
        self.max_meter = max_meter
        self.max_rad = max_rad

        self.pose_pe_agent = PosePE(pose_pe["agent"], pe_dim=pe_dim)
        self.pose_pe_map = PosePE(pose_pe["map"], pe_dim=pe_dim)
        self.pose_pe_tl = PosePE(pose_pe["tl"], pe_dim=pe_dim)

        self.register_buffer("pl_node_ohe", torch.eye(data_size["map/valid"][-1]))

        self.model_kwargs = {}

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: scene-centric Dict
            # (sc) scene-centric agents states
                "sc/agent_valid": [n_scene, n_step_hist, n_agent]
                "sc/agent_pos": [n_scene, n_step_hist, n_agent, 2]
                "sc/agent_z": [n_scene, n_step_hist, n_agent, 1]
                "sc/agent_vel": [n_scene, n_step_hist, n_agent, 2]
                "sc/agent_spd": [n_scene, n_step_hist, n_agent, 1]
                "sc/agent_acc": [n_scene, n_step_hist, n_agent, 1]
                "sc/agent_yaw_bbox": [n_scene, n_step_hist, n_agent, 1]
                "sc/agent_yaw_rate": [n_scene, n_step_hist, n_agent, 1]
            # agents attributes
                "sc/agent_type": [n_scene, n_agent, 3]
                "sc/agent_role": [n_scene, n_agent, 3]
                "sc/agent_size": [n_scene, n_agent, 3]
            # map polylines
                "sc/map_valid": [n_scene, n_pl, n_pl_node], bool
                "sc/map_type": [n_scene, n_pl, 11], bool one_hot
                "sc/map_pos": [n_scene, n_pl, n_pl_node, 2], float32
                "sc/map_dir": [n_scene, n_pl, n_pl_node, 2], float32
            # traffic lights
                "sc/tl_valid": [n_scene, n_step_hist, n_tl], bool
                "sc/tl_state": [n_scene, n_step_hist, n_tl, 5], bool one_hot
                "sc/tl_pos": [n_scene, n_step_hist, n_tl, 2], x,y
                "sc/tl_dir": [n_scene, n_step_hist, n_tl, 2], x,y
            # input dict
                # agent history
                "input/agent_valid": [n_scene, n_step_hist, n_agent], bool
                "input/agent_attr": [n_scene, n_step_hist, n_agent, agent_attr_dim], for input to MLP
                "input/agent_pe": [n_scene, n_step_hist, n_agent, hidden_dim], for input/cat/add to MLP
                "input/agent_pos": [n_scene, n_step_hist, n_agent, 2], for knn attention
                # map polylines
                "input/map_valid": [n_scene, n_pl, n_pl_node], bool
                "input/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
                "input/map_pe": [n_scene, n_pl, n_pl_node, hidden_dim], for input/cat/add to MLP
                "input/map_pos": [n_scene, n_pl, 2], for knn attention
                # traffic lights: stop point, detections are not tracked, singular node polyline.
                "input/tl_valid": [n_scene, n_step_hist, n_tl], bool
                "input/tl_attr": [n_scene, n_step_hist, n_tl, tl_attr_dim], for input to MLP
                "input/tl_pe": [n_scene, n_step_hist, n_tl, hidden_dim], for input/cat/add to MLP
                "input/tl_pos": [n_scene, n_step_hist, n_tl, 2], for knn attention

        Returns: add following keys to batch Dict
            # latent_prior from "sc/" or "input/"
                # agent
                "latent_prior/agent_valid": [n_scene, n_agent, n_step_hist]
                "latent_prior/agent_attr": [n_scene, n_agent, n_step_hist, agent_attr_dim]
                "latent_prior/agent_pe": [n_scene, n_agent, n_step_hist, hidden_dim]
                "latent_prior/agent_pos": [n_scene, n_agent, n_step_hist, 2]
                # map polylines
                "latent_prior/map_valid": [n_scene, n_pl, n_pl_node]
                "latent_prior/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
                "latent_prior/map_pe": [n_scene, n_pl, n_pl_node, hidden_dim]
                "latent_prior/map_pos": [n_scene, n_pl, 2]
                # traffic lights:
                "latent_prior/tl_valid": [n_scene, n_step_hist, n_tl]
                "latent_prior/tl_attr": [n_scene, n_step_hist, n_tl, tl_attr_dim]
                "latent_prior/tl_pe": [n_scene, n_step_hist, n_tl, hidden_dim]
                "latent_prior/tl_pos": [n_scene, n_step_hist, n_tl, 2]
            # latent_post from original batch gt
                # agent
                "latent_post/agent_valid": [n_scene, n_agent, n_step]
                "latent_post/agent_attr": [n_scene, n_agent, n_step, agent_attr_dim]
                "latent_post/agent_pe": [n_scene, n_agent, n_step, hidden_dim]
                "latent_post/agent_pos": [n_scene, n_agent, n_step, 2]
                # map polylines
                "latent_post/map_valid": [n_scene, n_pl, n_pl_node]
                "latent_post/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
                "latent_post/map_pe": [n_scene, n_pl, n_pl_node, hidden_dim]
                "latent_post/map_pos": [n_scene, n_pl, 2]
                # traffic lights:
                "latent_post/tl_valid": [n_scene, n_step, n_tl]
                "latent_post/tl_attr": [n_scene, n_step, n_tl, tl_attr_dim]
                "latent_post/tl_pe": [n_scene, n_step, n_tl, hidden_dim]
                "latent_post/tl_pos": [n_scene, n_step, n_tl, 2]
        """
        gt_available = "agent/valid" in batch.keys()  # training/validation time, prepare latent_posterior
        if self.training and self.perturb_input_to_latent:
            # "sc/agent_pos": [n_scene, n_agent, n_step_hist, 2]
            n_scene = batch["sc/agent_pos"].shape[0]
            device = batch["sc/agent_pos"].device
            dtype = batch["sc/agent_pos"].dtype
            rand_yaw = torch.rand([n_scene], device=device, dtype=dtype) * 2 * self.max_rad - self.max_rad
            rand_rot = torch_rad2rot(rand_yaw)
            rand_pos = torch.rand([n_scene, 2], device=device, dtype=dtype) * 2 * self.max_meter - self.max_meter
            rand_yaw = rand_yaw[:, None, None]  # [n_scene, 1, 1]
            rand_rot = rand_rot.unsqueeze(1)  # [n_scene, 1, 2, 2]
            rand_pos = rand_pos[:, None, None, :]  # [n_scene, 1, 1, 2]

        # ! map
        if self.training and self.perturb_input_to_latent:
            n_scene, n_pl, n_pl_node = batch["sc/map_valid"].shape
            map_pos = torch_pos2local(batch["sc/map_pos"], rand_pos, rand_rot)
            map_dir = torch_dir2local(batch["sc/map_dir"], rand_rot)
            batch["latent_prior/map_pos"] = map_pos[:, :, 0].contiguous()
            batch["latent_prior/map_attr"] = torch.cat(
                [
                    batch["sc/map_type"].unsqueeze(-2).expand(-1, -1, n_pl_node, -1),
                    self.pl_node_ohe[None, None, :, :].expand(n_scene, n_pl, -1, -1),
                ],
                dim=-1,
            )
            batch["latent_prior/map_pe"] = self.pose_pe_map(map_pos, map_dir)
            batch["latent_prior/map_valid"] = batch["sc/map_valid"]
        else:
            for k in ["valid", "pos", "attr", "pe"]:
                batch[f"latent_prior/map_{k}"] = batch[f"input/map_{k}"]

        if gt_available:
            for k in ["valid", "pos", "attr", "pe"]:
                batch[f"latent_post/map_{k}"] = batch[f"latent_prior/map_{k}"]

        # ! traffic lights
        # prior
        if self.training and self.perturb_input_to_latent:
            batch["latent_prior/tl_valid"] = batch["sc/tl_valid"]
            if self.training and (0 < self.dropout_p_history <= 1.0):
                prob_mask = torch.ones_like(batch["latent_prior/tl_valid"]) * (1 - self.dropout_p_history)
                batch["latent_prior/tl_valid"] &= torch.bernoulli(prob_mask).bool()
            tl_pos = batch["sc/tl_pos"]
            tl_dir = batch["sc/tl_dir"]
            if self.training and self.perturb_input_to_latent:
                tl_pos = torch_pos2local(tl_pos, rand_pos, rand_rot)
                tl_dir = torch_dir2local(tl_dir, rand_rot)
            batch["latent_prior/tl_pos"] = tl_pos
            batch["latent_prior/tl_attr"] = batch["sc/tl_state"].to(batch["latent_prior/tl_pos"].dtype)
            batch["latent_prior/tl_pe"] = self.pose_pe_tl(tl_pos, tl_dir)
        else:
            for k in ["valid", "pos", "attr", "pe"]:
                batch[f"latent_prior/tl_{k}"] = batch[f"input/tl_{k}"]
        # post
        if gt_available:  # train, val
            batch["latent_post/tl_valid"] = batch["tl_stop/valid"]
            if self.training and (0 < self.dropout_p_history <= 1.0):
                prob_mask = torch.ones_like(batch["latent_post/tl_valid"]) * (1 - self.dropout_p_history)
                batch["latent_post/tl_valid"] &= torch.bernoulli(prob_mask).bool()
            tl_pos = batch["tl_stop/pos"]
            tl_dir = batch["tl_stop/dir"]
            if self.training and self.perturb_input_to_latent:
                tl_pos = torch_pos2local(tl_pos, rand_pos, rand_rot)
                tl_dir = torch_dir2local(tl_dir, rand_rot)
            batch["latent_post/tl_pos"] = tl_pos
            batch["latent_post/tl_attr"] = batch["tl_stop/state"].to(batch["latent_post/tl_pos"].dtype)
            batch["latent_post/tl_pe"] = self.pose_pe_tl(tl_pos, tl_dir)

        # ! agents
        if self.training and self.perturb_input_to_latent:
            batch["latent_prior/agent_valid"] = batch["sc/agent_valid"]
            if self.training and (0 < self.dropout_p_history <= 1.0):
                prob_mask = torch.ones_like(batch["latent_prior/agent_valid"][:, :-1, :]) * (1 - self.dropout_p_history)
                batch["latent_prior/agent_valid"][:, :-1, :] &= torch.bernoulli(prob_mask).bool()
            agent_pos = batch["sc/agent_pos"]
            agent_vel = batch["sc/agent_vel"]
            agent_yaw_bbox = batch["sc/agent_yaw_bbox"]
            if self.training and self.perturb_input_to_latent:
                agent_pos = torch_pos2local(agent_pos, rand_pos, rand_rot)
                agent_vel = torch_dir2local(agent_vel, rand_rot)
                agent_yaw_bbox = torch_rad2local(agent_yaw_bbox, rand_yaw, cast=False)
            batch["latent_prior/agent_pos"] = agent_pos
            n_step = batch["latent_prior/agent_valid"].shape[1]
            batch["latent_prior/agent_attr"] = torch.cat(
                [
                    agent_vel,  # vel xy, 2
                    batch["sc/agent_spd"],  # speed, 1
                    batch["sc/agent_yaw_rate"],  # yaw rate, 1
                    batch["sc/agent_acc"],  # acc, 1
                    batch["sc/agent_size"].unsqueeze(1).expand(-1, n_step, -1, -1),  # 3
                    batch["sc/agent_type"].unsqueeze(1).expand(-1, n_step, -1, -1),  # 3
                ],
                dim=-1,
            )
            batch["latent_prior/agent_pe"] = self.pose_pe_agent(agent_pos, agent_yaw_bbox)
        else:
            for k in ["valid", "pos", "attr", "pe"]:
                batch[f"latent_prior/agent_{k}"] = batch[f"input/agent_{k}"]

        if gt_available:  # train, val
            batch["latent_post/agent_valid"] = batch["agent/valid"]
            if self.training and (0 < self.dropout_p_history <= 1.0):
                prob_mask = torch.ones_like(batch["latent_post/agent_valid"]) * (1 - self.dropout_p_history)
                batch["latent_post/agent_valid"] &= torch.bernoulli(prob_mask).bool()
            agent_pos = batch["agent/pos"]
            agent_vel = batch["agent/vel"]
            agent_yaw_bbox = batch["agent/yaw_bbox"]
            if self.training and self.perturb_input_to_latent:
                agent_pos = torch_pos2local(agent_pos, rand_pos, rand_rot)
                agent_vel = torch_dir2local(agent_vel, rand_rot)
                agent_yaw_bbox = torch_rad2local(agent_yaw_bbox, rand_yaw, cast=False)
            batch["latent_post/agent_pos"] = agent_pos
            n_step = batch["latent_post/agent_valid"].shape[1]
            batch["latent_post/agent_attr"] = torch.cat(
                [
                    agent_vel,  # vel xy, 2
                    batch["agent/spd"],  # speed, 1
                    batch["agent/yaw_rate"],  # yaw rate, 1
                    batch["agent/acc"],  # acc, 1
                    batch["agent/size"].unsqueeze(1).expand(-1, n_step, -1, -1),  # 3
                    batch["agent/type"].unsqueeze(1).expand(-1, n_step, -1, -1),  # 3
                ],
                dim=-1,
            )
            batch["latent_post/agent_pe"] = self.pose_pe_agent(agent_pos, agent_yaw_bbox)

        return batch
