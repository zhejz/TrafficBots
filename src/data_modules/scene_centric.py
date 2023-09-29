# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
from torch import nn, Tensor
import torch


class SceneCentricPreProcessing(nn.Module):
    def __init__(self, time_step_current: int, data_size: DictConfig) -> None:
        super().__init__()
        self.n_step_hist = time_step_current + 1
        self.model_kwargs = {}

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: scene-centric Dict
            # agent states
                "agent/valid": [n_scene, n_step, n_agent], bool,
                "agent/pos": [n_scene, n_step, n_agent, 2], float32
                "agent/z": [n_scene, n_step, n_agent, 1], float32
                "agent/vel": [n_scene, n_step, n_agent, 2], float32, v_x, v_y
                "agent/spd": [n_scene, n_step, n_agent, 1], norm of vel, signed using yaw_bbox and vel_xy
                "agent/acc": [n_scene, n_step, n_agent, 1], m/s2, acc[t] = (spd[t]-spd[t-1])/dt
                "agent/yaw_bbox": [n_scene, n_step, n_agent, 1], float32, yaw of the bbox heading
                "agent/yaw_rate": [n_scene, n_step, n_agent, 1], rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            # agent attributes
                "agent/type": [n_scene, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
                "agent/role": [n_scene, n_agent, 3], bool [sdc=0, interest=1, predict=2]
                "agent/size": [n_scene, n_agent, 3], float32: [length, width, height]
            # map polylines
                "map/valid": [n_scene, n_pl, n_pl_node], bool
                "map/type": [n_scene, n_pl, 11], bool one_hot
                "map/pos": [n_scene, n_pl, n_pl_node, 2], float32
                "map/dir": [n_scene, n_pl, n_pl_node, 2], float32
            # traffic lights
                "tl_stop/valid": [n_scene, n_step, n_tl_stop], bool
                "tl_stop/state": [n_scene, n_step, n_tl_stop, 5], bool one_hot
                "tl_stop/pos": [n_scene, n_step, n_tl_stop, 2], x,y
                "tl_stop/dir": [n_scene, n_step, n_tl_stop, 2], x,y

        Returns: scene-centric Dict, masked according to valid
            # (ref) reference information
                "ref/agent_type": [n_scene, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
                "ref/agent_role": [n_scene, n_agent, 3], bool [sdc=0, interest=1, predict=2]
                "ref/agent_state": [n_scene, n_step_hist, n_agent, 4], (x,y,yaw,spd)
                "ref/map_type": [n_scene, n_pl, 11]
            # (gt) ground-truth agent future for training, not available for testing
                "gt/valid": [n_scene, n_step, n_agent], bool
                "gt/spd": [n_scene, n_step, n_agent, 1]
                "gt/pos": [n_scene, n_step, n_agent, 2]
                "gt/vel": [n_scene, n_step, n_agent, 2]
                "gt/yaw_bbox": [n_scene, n_step, n_agent, 1]
                "gt/state": [n_scene, n_step, n_agent, 4], (x,y,yaw,spd)
                "gt/cmd": [n_scene, n_agent, 8]
                "gt/goal": [n_scene, n_agent, 4], (x, y, theta, spd)
                "gt/dest": [n_scene, n_agent], int64: index to map n_pl
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
            # agent_no_sim for validation and testing
                "sc/agent_no_sim_valid": [n_scene, n_step_hist, n_agent_no_sim]
                "sc/agent_no_sim_pos": [n_scene, n_step_hist, n_agent_no_sim, 2]
                "sc/agent_no_sim_z": [n_scene, n_step_hist, n_agent_no_sim, 1]
                "sc/agent_no_sim_vel": [n_scene, n_step_hist, n_agent_no_sim, 2]
                "sc/agent_no_sim_spd": [n_scene, n_step_hist, n_agent_no_sim, 1]
                "sc/agent_no_sim_acc": [n_scene, n_step_hist, n_agent_no_sim, 1]
                "sc/agent_no_sim_yaw_bbox": [n_scene, n_step_hist, n_agent_no_sim, 1]
                "sc/agent_no_sim_yaw_rate": [n_scene, n_step_hist, n_agent_no_sim, 1]
                "sc/agent_no_sim_type": [n_scene, n_agent_no_sim, 3]
                "sc/agent_no_sim_role": [n_scene, n_agent_no_sim, 3]
                "sc/agent_no_sim_size": [n_scene, n_agent_no_sim, 3]
        """
        prefix = "" if self.training else "history/"

        # ! prepare agents states
        # [n_scene, n_step, n_agent, ...] -> [n_scene, n_agent, n_step_hist, ...]
        for k in ("valid", "pos", "z", "vel", "spd", "acc", "yaw_bbox", "yaw_rate"):
            batch[f"sc/agent_{k}"] = batch[f"{prefix}agent/{k}"][:, : self.n_step_hist].contiguous()

        # ! prepare agents attributes
        for k in ("type", "role", "size"):
            batch[f"sc/agent_{k}"] = batch[f"{prefix}agent/{k}"]

        # ! training/validation time, prepare "gt/" for losses
        if "agent/valid" in batch.keys():
            for k in ("cmd", "goal", "dest"):
                batch[f"gt/{k}"] = batch[f"agent/{k}"]
            for k in ("valid", "spd", "pos", "vel", "yaw_bbox"):
                batch[f"gt/{k}"] = batch[f"agent/{k}"]
            batch["gt/state"] = torch.cat([batch["gt/pos"], batch["gt/yaw_bbox"], batch["gt/spd"]], dim=-1)

        # ! prepare map polylines
        for k in ("valid", "type", "pos", "dir"):
            batch[f"sc/map_{k}"] = batch[f"map/{k}"]

        # ! prepare traffic lights
        for k in ("valid", "state", "pos", "dir"):
            batch[f"sc/tl_{k}"] = batch[f"{prefix}tl_stop/{k}"][:, : self.n_step_hist].contiguous()

        # ! prepare agent_no_sim
        if not self.training:
            for k in ("valid", "pos", "z", "vel", "spd", "yaw_bbox"):
                batch[f"sc/agent_no_sim_{k}"] = batch[f"history/agent_no_sim/{k}"][:, : self.n_step_hist].contiguous()
            for k in ("type", "size"):
                batch[f"sc/agent_no_sim_{k}"] = batch[f"history/agent_no_sim/{k}"]

        # ! prepare "ref/"
        batch["ref/agent_type"] = batch[prefix + "agent/type"]
        batch["ref/agent_role"] = batch[prefix + "agent/role"]
        batch["ref/map_type"] = batch["map/type"]
        batch["ref/agent_state"] = torch.cat(
            [batch["sc/agent_pos"], batch["sc/agent_yaw_bbox"], batch["sc/agent_spd"]], dim=-1
        )  # x,y,yaw,spd

        return batch
