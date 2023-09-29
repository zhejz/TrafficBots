# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
from omegaconf import DictConfig
from torch import Tensor, nn
import torch
from .modules.mlp import MLP
from .modules.transformer import TransformerBlock
from .modules.agent_temporal import MultiAgentGRULoop, TemporalAggregate
from .modules.distributions import MyDist, DestCategorical, DiagGaussian
from .modules.attention import Attention
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_pos2global


class GoalManager(nn.Module):
    def __init__(
        self,
        tf_cfg: DictConfig,
        goal_predictor: DictConfig,
        goal_attr_mode: str,
        goal_in_local: bool,
        dest_detach_map_feature: bool,
        disable_if_reached: bool,
    ) -> None:
        super().__init__()
        self.goal_attr_mode = goal_attr_mode
        self.goal_in_local = goal_in_local
        self.dest_detach_map_feature = dest_detach_map_feature
        self.disable_if_reached = disable_if_reached
        hidden_dim = tf_cfg.d_model

        self.update_goal = False
        if self.goal_attr_mode == "dummy":
            self.dummy = True
            self.out_dim = -1
            self.goal_predictor = None
        elif self.goal_attr_mode == "dest":
            self.dummy = False
            self.out_dim = hidden_dim
            self.goal_predictor = DestPredictor(tf_cfg=tf_cfg, **goal_predictor)
        elif self.goal_attr_mode == "goal_xy":
            self.dummy = False
            if goal_in_local:
                self.update_goal = True
            self.out_dim = 2
            self.goal_predictor = GoalPredictor(tf_cfg=tf_cfg, goal_in_local=goal_in_local, **goal_predictor)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_gt_goal(
        self, agent_valid: Tensor, gt_goal: Tensor, gt_dest: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            agent_valid: [n_scene, n_step_hist, n_agent], bool
            gt_goal: [n_scene, n_agent, 4], (x, y, theta, spd)
            gt_dest: [n_scene, n_agent], int64: index to map n_pl

        Returns:
            gt: [n_scene, n_agent, (?)]
            valid: [n_scene, n_agent]
        """
        if self.goal_attr_mode == "dummy":
            gt = None
            valid = None
        elif self.goal_attr_mode == "dest":
            gt = gt_dest
            valid = agent_valid.any(1)
        elif self.goal_attr_mode == "goal_xy":
            gt = gt_goal[..., :2]
            valid = agent_valid.any(1)
        else:
            raise NotImplementedError

        return gt, valid

    def pred_goal(self, *args, **kwargs) -> Optional[MyDist]:
        if self.goal_predictor is None:
            return None
        else:
            return self.goal_predictor(*args, **kwargs)

    def get_goal_feature(self, goal: Tensor, as_state: Tensor, map_feature: Tensor) -> Tensor:
        """
        Args:
            goal: [n_batch, n_agent, 4] (x,y,yaw,spd) in global coordinate
            or goal: [n_batch, n_agent] int64 index to map n_pl
            as_state: [n_batch, (n_step), n_agent, 4] (x,y,yaw,spd) in global coordinate
            map_feature: [n_scene, n_pl, hidden_dim]

        Return:
            goal_feature: [n_batch, (n_step), n_agent, self.out_dim]
        """
        if as_state.dim() == 4:
            n_step = as_state.shape[1]
            if self.goal_attr_mode == "dest":
                # [n_batch, n_step, n_agent]
                goal = goal.unsqueeze(1).expand(-1, n_step, -1)
            else:
                # [n_batch, n_step, n_agent, 4]
                goal = goal.unsqueeze(1).expand(-1, n_step, -1, -1)

        if self.goal_attr_mode == "dest":
            if self.dest_detach_map_feature:
                map_feature = map_feature.detach()
            goal_feature = self._get_dest_feature(goal, map_feature)
        elif self.goal_attr_mode == "goal_xy":
            # [n_batch, (n_step), n_agent, 2]
            goal_feature = goal[..., :2]
            if self.goal_in_local:
                as_state = as_state.detach()
                ref_pos = as_state[..., :2].unsqueeze(-2)  # [n_batch, (n_step), n_agent, 1, 2]
                ref_rot = torch_rad2rot(as_state[..., 2])  # [n_batch, (n_step), n_agent, 2, 2]
                goal_feature = torch_pos2local(goal_feature.unsqueeze(-2), ref_pos, ref_rot).squeeze(-2)
        else:
            raise NotImplementedError

        return goal_feature

    @staticmethod
    def _get_dest_feature(dest: Tensor, map_feature: Tensor) -> Tensor:
        """
        Args:
            dest: [n_scene, (n_step), n_agent] int64(dest)
            map_feature: [n_scene, n_pl, hidden_dim]

        Return:
            dest_feature: [n_scene, (n_step), n_agent, hidden_dim]
        """
        # [n_scene, 1]
        batch_index = torch.arange(dest.shape[0]).unsqueeze(1)
        if dest.dim() == 3:
            dest_feature = []
            for k in range(dest.shape[1]):
                dest_feature.append(map_feature[batch_index, dest[:, k, :]])
            dest_feature = torch.stack(dest_feature, dim=1)
        else:
            dest_feature = map_feature[batch_index, dest]
        return dest_feature

    @torch.no_grad()
    def disable_goal_reached(
        self, goal_valid: Tensor, agent_valid: Tensor, dest_reached: Tensor, goal_reached: Tensor
    ) -> Tensor:
        """
        Args:
            goal_valid: [n_batch, n_agent] at t
            agent_valid: [n_batch, n_agent] at t
            dest_reached: [n_batch, n_agent] at t
            goal_reached: [n_batch, n_agent] at t

        Returns:
            goal_valid: [n_batch, n_agent] at t
        """
        if goal_valid is not None:
            goal_valid = goal_valid & agent_valid
        if self.disable_if_reached:
            if self.goal_attr_mode == "dest":
                goal_valid = goal_valid & (~dest_reached)
            elif self.goal_attr_mode == "goal_xy":
                goal_valid = goal_valid & (~goal_reached)
        return goal_valid


class DestPredictor(nn.Module):
    def __init__(
        self,
        tf_cfg: DictConfig,
        mode: str = "mlp",
        n_layer_gru: int = -1,
        use_layernorm: bool = True,
        res_add_gru: bool = True,
        detach_features: bool = True,
    ) -> None:
        super().__init__()
        assert mode in ["transformer", "transformer_aggr", "mlp", "attn"]
        self.mode = mode
        self.detach_features = detach_features
        self.res_add_gru = res_add_gru
        hidden_dim = tf_cfg.d_model

        if n_layer_gru > 0:
            self.gru_as = MultiAgentGRULoop(hidden_dim, num_layers=n_layer_gru, dropout=tf_cfg.dropout_p)
        else:
            self.gru_as = None
        self.agr_as = TemporalAggregate("last_valid")

        if self.mode == "transformer" or self.mode == "transformer_aggr":
            self.transformer_pl2as = TransformerBlock(hidden_dim, n_layer=1, **tf_cfg)
            self.mlp = MLP([hidden_dim, hidden_dim, 1], end_layer_activation=False, use_layernorm=use_layernorm)
        elif self.mode == "mlp":
            self.mlp = MLP(
                [hidden_dim * 2, hidden_dim, hidden_dim, 1], end_layer_activation=False, use_layernorm=use_layernorm
            )
        elif self.mode == "attn":
            self.attn = Attention(
                d_model=tf_cfg.d_model, n_head=tf_cfg.n_head, dropout_p=tf_cfg.dropout_p, bias=tf_cfg.bias
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        agent_type: Tensor,
        map_type: Tensor,
        agent_state: Tensor,
        agent_feature: Tensor,
        agent_feature_valid: Tensor,
        map_feature: Tensor,
        map_feature_valid: Tensor,
        tl_feature: Optional[Tensor] = None,
        tl_feature_valid: Optional[Tensor] = None,
    ) -> MyDist:
        """
        Args:
            agent_type: [n_scene, n_agent, 3] [Vehicle=0, Pedestrian=1, Cyclist=2] one hot
            map_type: [n_scene, n_pl, 11], one_hot, n_pl_type=11
            agent_state: [n_scene, n_step_hist, n_agent, 4], (x,y,yaw,spd)
            agent_feature: [n_scene, n_step_hist, n_agent, hidden_dim]
            agent_feature_valid: [n_scene, n_step_hist, n_agent] bool
            map_feature: [n_scene, n_pl, hidden_dim]
            map_feature_valid: [n_scene, n_pl] bool
            tl_feature: [n_scene, n_step_hist, n_tl, hidden_dim]
            tl_feature_valid: [n_scene, n_step_hist, n_tl] bool

        Returns:
            dest_dist: DestCategorical
        """
        if self.detach_features:
            agent_feature = agent_feature.detach()
            map_feature = map_feature.detach()

        # [n_scene, n_pl]
        # WOMD: FREEWAY = 0, SURFACE_STREET = 1, STOP_SIGN = 2, BIKE_LANE = 3, TYPE_ROAD_EDGE_BOUNDARY = 4
        map_type_mask = ~(map_feature_valid & (map_type[:, :, :5].any(-1)))

        # exclude (3) for veh(0): [n_scene, n_agent, 1] & [n_scene, 1, n_pl]
        attn_mask_veh = agent_type[:, :, [0]] & map_type[:, :, 3].unsqueeze(1)
        # exclude (0,1,2,3) for ped(1): [n_scene, n_agent, 1] & [n_scene, 1, n_pl]
        attn_mask_ped = agent_type[:, :, [1]] & map_type[:, :, :4].any(-1).unsqueeze(1)
        # exclude (0,1,2) for cyc(2): [n_scene, n_agent, 1] & [n_scene, 1, n_pl]
        attn_mask_cyc = agent_type[:, :, [2]] & map_type[:, :, :3].any(-1).unsqueeze(1)
        # [n_scene, n_agent, n_pl]
        attn_mask = attn_mask_veh | attn_mask_ped | attn_mask_cyc

        n_scene, n_pl, hidden_dim = map_feature.shape
        n_agent = agent_feature_valid.shape[2]
        dist_valid = agent_feature_valid.any(1)
        if self.mode == "transformer_aggr":
            if self.gru_as is None:
                tgt = agent_feature
            else:
                tgt, _ = self.gru_as(agent_feature, agent_feature_valid)
                if self.res_add_gru:
                    tgt = tgt + agent_feature
            # [n_scene, n_agent, hidden_dim]
            tgt, tgt_valid = self.agr_as(tgt, agent_feature_valid)

            map_feature_repeated = map_feature.unsqueeze(1).expand(-1, n_agent, -1, -1).flatten(0, 1)
            map_valid_repeated = map_feature_valid.unsqueeze(1).expand(-1, n_agent, -1).flatten(0, 1)
            # [n_scene*n_agent, n_pl, hidden_dim]
            map_feature_repeated, _ = self.transformer_pl2as(
                src=map_feature_repeated,  # [n_scene*n_agent, n_pl, hidden_dim]
                src_padding_mask=~map_valid_repeated,  # [n_scene*n_agent, n_pl]
                tgt=tgt.flatten(0, 1).unsqueeze(1),  # [n_scene*n_agent, 1, hidden_dim]
                tgt_padding_mask=~tgt_valid.flatten(0, 1).unsqueeze(1),  # [n_scene*n_agent, 1]
            )
            # [n_scene, n_agent, n_pl]
            logits = self.mlp(map_feature_repeated.view(n_scene, n_agent, n_pl, hidden_dim)).squeeze(-1)

        elif self.mode == "transformer":
            # downsample
            k_skip = 2
            # [n_scene, n_step_hist, n_agent, hidden_dim] -> [n_scene*n_agent, n_step_hist/2, hidden_dim]
            tgt = agent_feature.transpose(1, 2)[:, :, 0::k_skip].flatten(0, 1)
            # [n_scene, n_step_hist, n_agent]  -> [n_scene*n_agent, n_step_hist/2]
            tgt_valid = agent_feature_valid.transpose(1, 2)[:, :, 0::k_skip].flatten(0, 1)

            # agent_feature: [n_scene, n_agent, n_step_hist, hidden_dim]
            # agent_feature_valid: [n_scene, n_agent, n_step_hist] bool

            map_feature_repeated = map_feature.unsqueeze(1).expand(-1, n_agent, -1, -1).flatten(0, 1)
            map_valid_repeated = map_feature_valid.unsqueeze(1).expand(-1, n_agent, -1).flatten(0, 1)
            # [n_scene*n_agent, n_pl, hidden_dim]
            map_feature_repeated, _ = self.transformer_pl2as(
                src=map_feature_repeated,  # [n_scene*n_agent, n_pl, hidden_dim]
                src_padding_mask=~map_valid_repeated,  # [n_scene*n_agent, n_pl]
                tgt=tgt,  # [n_scene*n_agent, n_step_hist, hidden_dim]
                tgt_padding_mask=~tgt_valid,  # [n_scene*n_agent, c_step]
            )
            # [n_scene, n_agent, n_pl]
            logits = self.mlp(map_feature_repeated.view(n_scene, n_agent, n_pl, hidden_dim)).squeeze(-1)

        elif self.mode == "mlp":
            if self.gru_as is None:
                tgt = agent_feature
            else:
                tgt, _ = self.gru_as(agent_feature, agent_feature_valid)
                if self.res_add_gru:
                    tgt = tgt + agent_feature
            # [n_scene, n_agent, hidden_dim]
            tgt, tgt_valid = self.agr_as(tgt, agent_feature_valid)
            # [n_scene, n_agent, n_pl, hidden_dim]
            tgt = tgt.unsqueeze(2).expand(-1, -1, n_pl, -1)
            src = map_feature.unsqueeze(1).expand(-1, n_agent, -1, -1)
            # [n_scene, n_agent, n_pl]
            logits = self.mlp(torch.cat([src, tgt], dim=-1)).squeeze(-1)

        elif self.mode == "attn":
            if self.gru_as is None:
                src = agent_feature
            else:
                src, _ = self.gru_as(agent_feature, agent_feature_valid)
                if self.res_add_gru:
                    src = src + agent_feature
            # [n_scene, n_agent, hidden_dim]
            src, _ = self.agr_as(src, agent_feature_valid)
            attn_mask = attn_mask.repeat_interleave(self.n_head, 0)
            # [n_scene, n_agent, n_pl]
            _, probs = self.attn(
                src=src,
                tgt=map_feature,
                tgt_padding_mask=map_feature,
                attn_mask=attn_mask,
                key_padding_mask=map_type_mask,
            )
            logits = torch.log(probs)

        logits = logits.masked_fill(map_type_mask.unsqueeze(1), float("-inf"))
        logits = logits.masked_fill(attn_mask, float("-inf"))
        logits = logits.masked_fill(~dist_valid.unsqueeze(-1), 0)
        logits = logits.masked_fill((logits == float("-inf")).all(-1).unsqueeze(-1), 0)
        return DestCategorical(logits=logits, valid=dist_valid)


class GoalPredictor(nn.Module):
    def __init__(
        self,
        tf_cfg: DictConfig,
        goal_in_local: bool,
        mode: str = "transformer",
        n_layer_gru: int = 3,
        use_layernorm: bool = True,
        res_add_gru: bool = True,
        detach_features: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.detach_features = detach_features
        self.res_add_gru = res_add_gru
        self.goal_in_local = goal_in_local
        hidden_dim = tf_cfg.d_model

        if n_layer_gru > 0:
            self.gru_as = MultiAgentGRULoop(hidden_dim, num_layers=n_layer_gru, dropout=tf_cfg.dropout_p)
        else:
            self.gru_as = None
        self.agr_as = TemporalAggregate("last_valid")

        self.transformer_as2pl = TransformerBlock(n_layer=1, **tf_cfg)

        self.mlp_mean = MLP([hidden_dim, hidden_dim, 2], end_layer_activation=False, use_layernorm=use_layernorm)
        self.log_std = nn.Parameter(2.0 * torch.ones(2), requires_grad=True)

    def forward(
        self,
        agent_type: Tensor,
        map_type: Tensor,
        agent_state: Tensor,
        agent_feature: Tensor,
        agent_feature_valid: Tensor,
        map_feature: Tensor,
        map_feature_valid: Tensor,
        tl_feature: Optional[Tensor] = None,
        tl_feature_valid: Optional[Tensor] = None,
    ) -> MyDist:
        """
        Args:
            agent_type: [n_scene, n_agent, 3] [Vehicle=0, Pedestrian=1, Cyclist=2] one hot
            map_type: [n_scene, n_pl, 11], one_hot, n_pl_type=11
            agent_state: [n_scene, n_step_hist, n_agent, 4], (x,y,yaw,spd)
            agent_feature: [n_scene, n_step_hist, n_agent, hidden_dim]
            agent_feature_valid: [n_scene, n_step_hist, n_agent] bool
            map_feature: [n_scene, n_pl, hidden_dim]
            map_feature_valid: [n_scene, n_pl] bool
            tl_feature: [n_scene, n_step_hist, n_tl, hidden_dim]
            tl_feature_valid: [n_scene, n_step_hist, n_tl] bool

        Returns:
            dest_dist: diag_gaus
        """
        if self.detach_features:
            agent_feature = agent_feature.detach()
            map_feature = map_feature.detach()

        if self.gru_as is None:
            src = agent_feature
        else:
            src, _ = self.gru_as(agent_feature, agent_feature_valid)
            if self.res_add_gru:
                src = src + agent_feature
        # [n_scene, n_agent, hidden_dim]
        src, src_valid = self.agr_as(src, agent_feature_valid)

        # [n_scene, n_agent, hidden_dim]
        goal_feature, _ = self.transformer_as2pl(
            src=src,  # [n_scene, n_agent, hidden_dim]
            src_padding_mask=~src_valid,  # [n_scene, n_agent]
            tgt=map_feature,  # [n_scene, n_pl, hidden_dim]
            tgt_padding_mask=~map_feature_valid,  # [n_scene, n_pl]
            need_weights=False,
        )

        goal_mean = self.mlp_mean(goal_feature)

        if self.goal_in_local:
            ref_pos = agent_state[:, -1, :, :2].unsqueeze(-2)  # [n_scene, n_agent, 1, 2]
            ref_rot = torch_rad2rot(agent_state[:, -1, :, 2])  # [n_scene, n_agent, 2, 2]
            goal_mean = torch_pos2global(goal_mean.unsqueeze(-2), ref_pos, ref_rot).squeeze(-2)

        goal_valid = agent_feature_valid.any(1)
        goal_mean = goal_mean.masked_fill(~goal_valid.unsqueeze(-1), 0)
        return DiagGaussian(mean=goal_mean, log_std=self.log_std, valid=goal_valid)
