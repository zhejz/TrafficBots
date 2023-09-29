# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, Union, Dict
from omegaconf import DictConfig
import hydra
from torch import Tensor, nn
from .modules.distributions import MyDist
from .modules.mlp import MLP
from .modules.transformer import TransformerBlock
from .modules.map_encoder import MapEncoder
from .modules.input_pe_encoder import InputPeEncoder
from .modules.agent_interaction import MultiAgentTF
from .modules.agent_temporal import TemporalAggregate
from .modules.add_latent_goal import AddLatentGoal
from .goal_manager import GoalManager
from .latent_encoder import LatentEncoder


class TrafficBots(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        agent_attr_dim: int,
        map_pe_dim: int,
        tl_pe_dim: int,
        agent_pe_dim: int,
        map_encoder: DictConfig,
        input_pe_encoder: DictConfig,
        goal_manager: DictConfig,
        latent_encoder: DictConfig,
        tf_cfg: DictConfig,
        n_layer_tf_as2pl: int,
        n_layer_tf_as2tl: int,
        n_step_hist: int,
        n_pl_node: int,
        temporal_aggregate: DictConfig,
        agent_temporal: DictConfig,
        agent_interaction: DictConfig,
        add_latent: DictConfig,
        add_goal: DictConfig,
        interaction_first: bool,
        add_goal_latent_first: bool,
        resample_latent: bool,
        n_layer_final_mlp: int,
        final_mlp: DictConfig,
    ):
        super().__init__()
        self.resample_latent = resample_latent
        self.interaction_first = interaction_first
        self.add_goal_latent_first = add_goal_latent_first

        # if self.add_learned_pe:
        #     self.pe_target = nn.Parameter(torch.zeros([1, n_step_hist, hidden_dim]), requires_grad=True)
        #     self.pe_other = nn.Parameter(torch.zeros([1, 1, n_step_hist, hidden_dim]), requires_grad=True)
        #     self.pe_map = nn.Parameter(torch.zeros([1, 1, n_pl_node, hidden_dim]), requires_grad=True)
        #     if self.use_current_tl:
        #         self.pe_tl = nn.Parameter(torch.zeros([1, 1, 1, hidden_dim]), requires_grad=True)
        #     else:
        #         self.pe_tl = nn.Parameter(torch.zeros([1, n_step_hist, 1, hidden_dim]), requires_grad=True)

        # input encoder
        self.map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            attr_dim=map_attr_dim,
            pe_dim=map_pe_dim,
            input_pe_encoder=input_pe_encoder,
            tf_cfg=tf_cfg,
            **map_encoder,
        )
        self.tl_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=tl_attr_dim, pe_dim=tl_pe_dim, **input_pe_encoder
        )
        self.agent_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=agent_attr_dim, pe_dim=agent_pe_dim, **input_pe_encoder
        )
        # cross-attn transformers, could be shared between policy and latent_encoder
        self.transformer_as2pl = TransformerBlock(n_layer=n_layer_tf_as2pl, **tf_cfg)
        self.transformer_as2tl = TransformerBlock(n_layer=n_layer_tf_as2tl, **tf_cfg)
        # goal
        self.goal_manager = GoalManager(tf_cfg=tf_cfg, **goal_manager)
        # latent
        self.latent_encoder = LatentEncoder(
            tf_cfg=tf_cfg,
            interaction_first=interaction_first,
            transformer_as2pl=self.transformer_as2pl,
            transformer_as2tl=self.transformer_as2tl,
            agent_temporal=agent_temporal,
            agent_interaction=agent_interaction,
            temporal_aggregate=temporal_aggregate,
            **latent_encoder,
        )
        # policy
        self.agent_temporal = hydra.utils.instantiate(agent_temporal, hidden_dim=hidden_dim)
        self.agent_interaction = MultiAgentTF(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction)
        self.temporal_aggregate = TemporalAggregate(**temporal_aggregate)
        self.add_goal = AddLatentGoal(
            hidden_dim=hidden_dim, in_dim=self.goal_manager.out_dim, dummy=self.goal_manager.dummy, **add_goal
        )
        self.add_latent = AddLatentGoal(
            hidden_dim=hidden_dim, in_dim=self.latent_encoder.out_dim, dummy=self.latent_encoder.dummy, **add_latent
        )

        if n_layer_final_mlp > 0:
            self.final_mlp = MLP([hidden_dim] * (n_layer_final_mlp + 1), **final_mlp)
        else:
            self.final_mlp = None

    def encode_input_features(
        self,
        agent_valid: Tensor,
        agent_attr: Tensor,
        agent_pe: Tensor,
        agent_pos: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        map_pe: Tensor,
        map_pos: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        tl_pe: Tensor,
        tl_pos: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            agent_valid: [n_scene, n_step(_hist), n_agent], bool
            agent_attr: [n_scene, n_step(_hist), n_agent, agent_attr_dim]
            agent_pe: [n_scene, n_step(_hist), n_agent, hidden_dim]
            agent_pos: [n_scene, n_step(_hist), n_agent, 2]
            map_valid: [n_scene, n_pl, n_pl_node], bool
            map_attr: [n_scene, n_pl, n_pl_node, map_attr_dim]
            map_pe: [n_scene, n_pl, n_pl_node, hidden_dim]
            map_pos: [n_scene, n_pl, 2]
            tl_valid: [n_scene, n_step(_hist), n_tl], bool
            tl_attr: [n_scene, n_step(_hist), n_tl, tl_attr_dim]
            tl_pe: [n_scene, n_step(_hist), n_tl, hidden_dim]
            tl_pos: [n_scene, n_step(_hist), n_tl, 2]

        Returns: for each agent a latent distribution that considers temporal relation and interaction between agents.
            agent_feature: [n_scene, n_step_hist, n_agent, hidden_dim]
            agent_feature_valid: [n_scene, n_step_hist, n_agent]
            map_feature: [n_scene, n_pl, hidden_dim]
            map_feature_valid: [n_scene, n_pl]
            tl_feature: [n_scene, n_step_hist, n_tl, hidden_dim]
            tl_feature_valid: [n_scene, n_step_hist, n_tl]
        """
        feature_dict = {"agent_feature_valid": agent_valid, "tl_feature_valid": tl_valid}
        feature_dict["map_feature"], feature_dict["map_feature_valid"] = self.map_encoder(map_valid, map_attr, map_pe)
        feature_dict["agent_feature"] = self.agent_encoder(agent_valid, agent_attr, agent_pe)
        feature_dict["tl_feature"] = self.tl_encoder(tl_valid, tl_attr, tl_pe)
        return feature_dict

    def init(self, latent: MyDist, deterministic: Union[bool, Tensor]) -> None:
        """
        deterministic: bool or [n_batch, n_agent] for sampling relevant agents and determistic other agents.
        """
        self.latent = latent
        self.deterministic = deterministic
        self.hidden = None
        self.latent_sample = None
        self.latent_logp = None

    def forward(
        self,
        agent_valid: Tensor,
        agent_feature: Tensor,
        map_valid: Tensor,
        map_feature: Tensor,
        tl_valid: Tensor,
        tl_feature: Tensor,
        goal_valid: Optional[Tensor],
        goal_feature: Optional[Tensor],
        need_weights: bool = False,
    ) -> Tuple[
        Tensor, Tensor, Optional[MyDist], Optional[MyDist], Optional[Tensor], Optional[Tensor], Optional[Tensor]
    ]:
        """
        Args:
            agent_valid: [n_batch, n_agent], bool
            agent_feature: [n_batch, n_agent, hidden_dim]
            map_valid: [n_batch, n_pl], bool
            map_feature: [n_batch, n_pl, hidden_dim]
            tl_valid: [n_batch, n_tl], bool
            tl_feature: [n_batch, n_tl, hidden_dim]
            goal_valid: [n_batch, n_agent] or None
            goal_feature: [n_batch, n_agent, hidden_dim] or None

        Returns: for each agent a latent distribution that considers temporal relation and interaction between agents.
            policy_feature: [n_batch, n_agent, hidden_dim]
            latent_logp: [n_batch, n_agent]
            attn_pl: [n_batch, n_agent, n_pl]
            attn_tl: [n_batch, n_agent, n_tl]
            attn_agent: [n_batch, n_agent, n_agent]
        """
        # ! sample latent
        if self.resample_latent or (self.latent_sample is None):
            # sample latent at each time step
            self.latent_sample = self.latent.sample(self.deterministic)
            self.latent_logp = self.latent.log_prob(self.latent_sample.detach())

        # [n_batch, n_step, n_agent, hidden_dim]
        policy_feature = agent_feature

        # ! attention to map polyline
        policy_feature, attn_pl = self.transformer_as2pl(
            src=policy_feature,  # [n_batch, n_agent, hidden_dim]
            src_padding_mask=~agent_valid,  # [n_batch, n_agent]
            tgt=map_feature,  # [n_batch, n_pl, hidden_dim]
            tgt_padding_mask=~map_valid,  # [n_batch, n_pl]
            need_weights=need_weights,
        )
        # ! attention to traffic light
        policy_feature, attn_tl = self.transformer_as2tl(
            src=policy_feature,  # [n_batch, n_agent, hidden_dim]
            src_padding_mask=~agent_valid,  # [n_batch, n_agent]
            tgt=tl_feature,  # [n_batch, n_tl, hidden_dim]
            tgt_padding_mask=~tl_valid,  # [n_batch, n_tl]
            need_weights=need_weights,
        )

        if self.add_goal_latent_first:
            # ! add goal and latent before interaction
            policy_feature = self.add_goal(policy_feature, agent_valid, goal_feature, goal_valid)
            policy_feature = self.add_latent(policy_feature, agent_valid, self.latent_sample, agent_valid)

        # ! interaction and temporal
        if self.interaction_first:
            policy_feature, attn_agent = self.agent_interaction(
                policy_feature, agent_feature, agent_valid, need_weights=need_weights
            )
            policy_feature, self.hidden = self.agent_temporal(policy_feature, agent_valid, self.hidden)
        else:
            policy_feature, self.hidden = self.agent_temporal(policy_feature, agent_valid, self.hidden)
            policy_feature, attn_agent = self.agent_interaction(
                policy_feature, agent_feature, agent_valid, need_weights=need_weights
            )

        if not self.add_goal_latent_first:
            # ! add goal and latent before interaction
            policy_feature = self.add_goal(policy_feature, agent_valid, goal_feature, goal_valid)
            policy_feature = self.add_latent(policy_feature, agent_valid, self.latent_sample, agent_valid)

        # ! final mlp
        if self.final_mlp is not None:
            policy_feature = self.final_mlp(policy_feature, agent_valid)

        return policy_feature, self.latent_logp, attn_pl, attn_tl, attn_agent
