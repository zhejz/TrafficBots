# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from torch import Tensor, nn
import torch
from omegaconf import DictConfig
import hydra
from .modules.mlp import MLP
from .modules.transformer import TransformerBlock
from .modules.distributions import MyDist, DummyLatent, DiagGaussian, MultiCategorical
from .modules.agent_interaction import MultiAgentTF
from .modules.agent_temporal import TemporalAggregate


class LatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        temporal_down_sample_rate: int,
        shared_post_prior_net: bool,
        shared_transformer_as: bool,
        latent_prior: DictConfig,
        latent_post: DictConfig,
        tf_cfg: DictConfig,
        interaction_first: bool,
        transformer_as2pl: TransformerBlock,
        transformer_as2tl: TransformerBlock,
        agent_temporal: DictConfig,
        agent_interaction: DictConfig,
        temporal_aggregate: DictConfig,
    ):
        super().__init__()
        hidden_dim = tf_cfg.d_model
        self.out_dim = latent_dim
        self.dummy = latent_post.dist_type == "dummy"
        self.temporal_down_sample_rate = temporal_down_sample_rate
        self.interaction_first = interaction_first

        # agent state map aware
        if shared_transformer_as:
            self.transformer_as2pl = transformer_as2pl
            self.transformer_as2tl = transformer_as2tl
        else:
            self.transformer_as2pl = TransformerBlock(n_layer=len(transformer_as2pl.layers), **tf_cfg)
            self.transformer_as2tl = TransformerBlock(n_layer=len(transformer_as2tl.layers), **tf_cfg)

        # such that prior and posterior distribution have different standard deviation.
        self.latent_prior_dist = DistEncoder(**latent_prior, hidden_dim=hidden_dim, out_dim=latent_dim)
        self.latent_post_dist = DistEncoder(**latent_post, hidden_dim=hidden_dim, out_dim=latent_dim)

        if self.latent_post_dist.skip_forward:
            self.temporal_aggregate = None
            self.agent_temporal_post = None
            self.agent_interaction_post = None
            self.agent_temporal_prior = None
            self.agent_interaction_prior = None
        else:
            self.temporal_aggregate = TemporalAggregate(**temporal_aggregate)
            self.agent_temporal_post = hydra.utils.instantiate(agent_temporal, hidden_dim=hidden_dim)
            self.agent_interaction_post = MultiAgentTF(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction)
            if self.latent_prior_dist.skip_forward:
                self.agent_temporal_prior = None
                self.agent_interaction_prior = None
            elif shared_post_prior_net:
                self.agent_temporal_prior = self.agent_temporal_post
                self.agent_interaction_prior = self.agent_interaction_post
            else:
                self.agent_temporal_prior = hydra.utils.instantiate(agent_temporal, hidden_dim=hidden_dim)
                self.agent_interaction_prior = MultiAgentTF(hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction)

    def forward(
        self,
        agent_feature: Tensor,
        agent_feature_valid: Tensor,
        map_feature: Tensor,
        map_feature_valid: Tensor,
        tl_feature: Optional[Tensor] = None,
        tl_feature_valid: Optional[Tensor] = None,
        posterior: bool = False,
    ) -> MyDist:
        """
        Args:
            agent_feature: [n_scene, n_step, n_agent, hidden_dim]
            agent_feature_valid: [n_scene, n_step, n_agent] bool
            map_feature: [n_scene, n_pl, hidden_dim]
            map_feature_valid: [n_scene, n_pl] bool
            tl_feature: [n_scene, n_step, n_tl, hidden_dim]
            tl_feature_valid: [n_scene, n_step, n_tl] bool

        Returns: for each agent a latent distribution that considers temporal relation and interaction between agents.
            latent: [n_batch, n_agent, hidden_dim]
        """
        if posterior and self.latent_post_dist.skip_forward:
            return self.latent_post_dist(agent_feature[:, 0], agent_feature_valid.any(1))
        elif (not posterior) and self.latent_prior_dist.skip_forward:
            return self.latent_prior_dist(agent_feature[:, 0], agent_feature_valid.any(1))
        else:
            # ! downsampling
            if self.temporal_down_sample_rate > 1:
                assert (agent_feature_valid.shape[1] - 1) % self.temporal_down_sample_rate == 0
                agent_feature_valid = agent_feature_valid[:, :: self.temporal_down_sample_rate]
                agent_feature = agent_feature[:, :: self.temporal_down_sample_rate]
                tl_feature_valid = tl_feature_valid[:, :: self.temporal_down_sample_rate]
                tl_feature = tl_feature[:, :: self.temporal_down_sample_rate]

            # [n_batch, n_step, n_agent, hidden_dim]
            as_feature_map_aware = agent_feature
            as_shape = as_feature_map_aware.shape
            # ! attention to map polyline
            as_feature_map_aware, _ = self.transformer_as2pl(
                src=as_feature_map_aware.flatten(1, 2),  # [n_batch, n_step * n_agent, hidden_dim]
                src_padding_mask=~agent_feature_valid.flatten(1, 2),  # [n_batch, n_step * n_agent]
                tgt=map_feature,  # [n_batch, n_pl, hidden_dim]
                tgt_padding_mask=~map_feature_valid,  # [n_batch, n_pl]
            )
            as_feature_map_aware = as_feature_map_aware.view(as_shape)
            # ! attention to traffic light
            as_feature_map_aware, _ = self.transformer_as2tl(
                src=as_feature_map_aware.flatten(0, 1),  # [n_batch * n_step, n_agent, hidden_dim]
                src_padding_mask=~agent_feature_valid.flatten(0, 1),  # [n_batch * n_step, n_agent]
                tgt=tl_feature.flatten(0, 1),  # [n_batch * n_step, n_tl, hidden_dim]
                tgt_padding_mask=~tl_feature_valid.flatten(0, 1),  # [n_batch * n_step, n_tl]
            )
            as_feature_map_aware = as_feature_map_aware.view(as_shape)

            # ! interaction and temporal
            if posterior:  # post
                if self.interaction_first:
                    latent_feature, _ = self.agent_interaction_post(
                        as_feature_map_aware, agent_feature, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_temporal_post(latent_feature, agent_feature_valid)
                else:
                    latent_feature, _ = self.agent_temporal_post(as_feature_map_aware, agent_feature_valid)
                    latent_feature, _ = self.agent_interaction_post(latent_feature, agent_feature, agent_feature_valid)
                latent_feature, latent_valid = self.temporal_aggregate(latent_feature, agent_feature_valid)
                return self.latent_post_dist(latent_feature, latent_valid)
            else:  # prior
                if self.interaction_first:
                    latent_feature, _ = self.agent_interaction_prior(
                        as_feature_map_aware, agent_feature, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_temporal_prior(latent_feature, agent_feature_valid)
                else:
                    latent_feature, _ = self.agent_temporal_prior(as_feature_map_aware, agent_feature_valid)
                    latent_feature, _ = self.agent_interaction_prior(latent_feature, agent_feature, agent_feature_valid)
                latent_feature, latent_valid = self.temporal_aggregate(latent_feature, agent_feature_valid)
                return self.latent_prior_dist(latent_feature, latent_valid)


class DistEncoder(nn.Module):
    def __init__(
        self, dist_type: str, hidden_dim: int, out_dim: int, use_layernorm: bool, log_std: float = 0.0, n_cat: int = 1
    ) -> None:
        """
        dist_type in {dummy, std_gaus, diag_gaus, gaus, cat}
        """
        super().__init__()
        self.dist_type = dist_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.skip_forward = False

        if dist_type == "dummy":
            self.skip_forward = True
        elif dist_type == "std_gaus":
            self.log_std = nn.Parameter(log_std * torch.ones(out_dim), requires_grad=False)
            self.skip_forward = True
        elif dist_type == "diag_gaus":
            self.mlp_mean = MLP(
                [hidden_dim, hidden_dim, out_dim], end_layer_activation=False, use_layernorm=use_layernorm
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = MLP(
                    [hidden_dim, hidden_dim, out_dim], end_layer_activation=False, use_layernorm=use_layernorm,
                )
            else:
                self.log_std = nn.Parameter(log_std * torch.ones(out_dim), requires_grad=True)
        elif dist_type == "cat":
            assert out_dim % n_cat == 0
            self.n_cat = n_cat
            self.n_class = out_dim // self.n_cat
            self.mlp_logits = MLP(
                [hidden_dim, hidden_dim, out_dim], end_layer_activation=False, use_layernorm=use_layernorm
            )

    def forward(self, x: Tensor, valid: Tensor) -> MyDist:
        if self.dist_type == "dummy":
            out_dist = DummyLatent(x, valid)
        elif self.dist_type == "std_gaus":
            out_dist = DiagGaussian(
                torch.zeros([*valid.shape, self.out_dim], dtype=x.dtype, device=x.device), self.log_std, valid=valid
            )
        elif self.dist_type == "diag_gaus":
            if self.log_std is None:
                out_dist = DiagGaussian(self.mlp_mean(x, valid), self.mlp_log_std(x, valid), valid=valid)
            else:
                out_dist = DiagGaussian(self.mlp_mean(x, valid), self.log_std, valid=valid)
        elif self.dist_type == "cat":
            logits = self.mlp_logits(x, valid).view(*valid.shape, self.n_cat, self.n_class)
            out_dist = MultiCategorical(nn.functional.softmax(logits, -1), valid=valid)
        return out_dist
