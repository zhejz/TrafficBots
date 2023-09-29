# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, Tuple
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from utils.pose_pe import PosePE


class SceneCentricInput(nn.Module):
    def __init__(
        self, time_step_current: int, data_size: DictConfig, dropout_p_history: float, pe_dim: int, pose_pe: DictConfig
    ) -> None:
        super().__init__()
        self.dropout_p_history = dropout_p_history  # [0, 1], turn off if set to negative
        self.n_step_hist = time_step_current + 1

        self.pose_pe_agent = PosePE(pose_pe["agent"], pe_dim=pe_dim)
        self.pose_pe_map = PosePE(pose_pe["map"], pe_dim=pe_dim)
        self.pose_pe_tl = PosePE(pose_pe["tl"], pe_dim=pe_dim)

        agent_attr_dim = (
            data_size["agent/vel"][-1]  # 2
            + data_size["agent/spd"][-1]  # 1
            + data_size["agent/yaw_rate"][-1]  # 1
            + data_size["agent/acc"][-1]  # 1
            + data_size["agent/size"][-1]  # 3
            + data_size["agent/type"][-1]  # 3
        )
        agent_pe_dim = self.pose_pe_agent.out_dim

        n_pl_node = data_size["map/valid"][-1]
        map_attr_dim = data_size["map/type"][-1] + n_pl_node
        map_pe_dim = self.pose_pe_map.out_dim
        self.register_buffer("pl_node_ohe", torch.eye(n_pl_node))

        tl_attr_dim = data_size["tl_stop/state"][-1]
        tl_pe_dim = self.pose_pe_tl.out_dim

        self.model_kwargs = {
            "agent_attr_dim": agent_attr_dim,
            "agent_pe_dim": agent_pe_dim,
            "map_attr_dim": map_attr_dim,
            "map_pe_dim": map_pe_dim,
            "tl_attr_dim": tl_attr_dim,
            "tl_pe_dim": tl_pe_dim,
            "n_step_hist": self.n_step_hist,
            "n_pl_node": n_pl_node,
        }

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

        Returns: add following keys to batch Dict
            # rollout
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
        """
        batch["input/agent_valid"] = batch["sc/agent_valid"]
        batch["input/tl_valid"] = batch["sc/tl_valid"]
        batch["input/map_valid"] = batch["sc/map_valid"]

        # ! randomly mask history agent/tl/map
        if self.training and (0 < self.dropout_p_history <= 1.0):
            prob_mask = torch.ones_like(batch["input/agent_valid"][:, :-1, :]) * (1 - self.dropout_p_history)
            batch["input/agent_valid"][:, :-1, :] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/tl_valid"]) * (1 - self.dropout_p_history)
            batch["input/tl_valid"] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/map_valid"]) * (1 - self.dropout_p_history)
            batch["input/map_valid"] &= torch.bernoulli(prob_mask).bool()

        # ! prepare "input/agent"
        batch["input/agent_pos"] = batch["sc/agent_pos"]
        batch["input/agent_attr"] = torch.cat(
            [
                batch["sc/agent_vel"],  # vel xy, 2
                batch["sc/agent_spd"],  # speed, 1
                batch["sc/agent_yaw_rate"],  # yaw rate, 1
                batch["sc/agent_acc"],  # acc, 1
                batch["sc/agent_size"].unsqueeze(1).expand(-1, self.n_step_hist, -1, -1),  # 3
                batch["sc/agent_type"].unsqueeze(1).expand(-1, self.n_step_hist, -1, -1),  # 3
            ],
            dim=-1,
        )
        # "input/agent_pe": [n_scene, n_agent, n_step_hist, hidden_dim], for input/cat/add to MLP
        batch["input/agent_pe"] = self.pose_pe_agent(batch["sc/agent_pos"], batch["sc/agent_yaw_bbox"])

        # ! prepare "input/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
        n_scene, n_pl, n_pl_node = batch["sc/map_valid"].shape
        batch["input/map_pos"] = batch["sc/map_pos"][:, :, 0].contiguous()
        batch["input/map_attr"] = torch.cat(
            [
                batch["sc/map_type"].unsqueeze(-2).expand(-1, -1, n_pl_node, -1),
                self.pl_node_ohe[None, None, :, :].expand(n_scene, n_pl, -1, -1),
            ],
            dim=-1,
        )
        batch["input/map_pe"] = self.pose_pe_map(batch["sc/map_pos"], batch["sc/map_dir"])

        # ! prepare "input/tl_attr": [n_scene, n_step_hist, n_tl, tl_attr_dim]
        batch["input/tl_pos"] = batch["sc/tl_pos"]
        batch["input/tl_attr"] = batch["sc/tl_state"].to(batch["input/tl_pos"].dtype)
        batch["input/tl_pe"] = self.pose_pe_tl(batch["sc/tl_pos"], batch["sc/tl_dir"])
        return batch

    def get_agent_attr_and_pe(
        self,
        agent_pos: Tensor,
        agent_yaw_bbox: Tensor,
        agent_vel: Tensor,
        agent_spd: Tensor,
        agent_yaw_rate: Tensor,
        agent_acc: Tensor,
        agent_size: Tensor,
        agent_type: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        agent_attr = torch.cat(
            [
                agent_vel,  # vel xy, 2
                agent_spd,  # speed, 1
                agent_yaw_rate,  # yaw rate, 1
                agent_acc,  # acc, 1
                agent_size,  # 3
                agent_type,  # 3
            ],
            dim=-1,
        )  # [n_scene, n_agent, agent_attr_dim]
        agent_pe = self.pose_pe_agent(agent_pos, agent_yaw_bbox)  # for input/cat/add to MLP
        return agent_attr, agent_pe
