# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from omegaconf import DictConfig
from .transformer import TransformerBlock


class MultiAgentTF(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        attn_to_map_aware_feature: bool,
        mask_self_agent: bool,
        detach_tgt: bool,
        tf_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_self_agent = mask_self_agent
        self.attn_mask = None
        self.detach_tgt = detach_tgt
        self.attn_to_map_aware_feature = attn_to_map_aware_feature
        self.transformer = TransformerBlock(n_layer=n_layer, **tf_cfg)

    def forward(
        self, as_feature_map_aware: Tensor, as_feature: Tensor, as_valid: Tensor, need_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            as_feature_map_aware: [n_batch, (n_step), n_agent, hidden_dim]
            as_feature: [n_batch, (n_step), n_agent, hidden_dim]
            as_valid: [n_batch, (n_step), n_agent]

        Returns:
            x: [n_batch, (n_step), n_agent, out_dim]
            attn_weights: None or [n_batch, n_agent, n_agent]
        """
        valid_dim = as_valid.dim()
        if valid_dim == 3:
            n_batch, n_step, n_agent = as_valid.shape
            # [n_batch*n_step, n_agent, hidden_dim]
            as_feature_map_aware = as_feature_map_aware.flatten(start_dim=0, end_dim=1)
            as_feature = as_feature.flatten(start_dim=0, end_dim=1)
            # [n_batch*n_step, n_agent]
            as_valid = as_valid.flatten(start_dim=0, end_dim=1)
        elif valid_dim == 2:
            n_batch, n_agent = as_valid.shape

        x = as_feature_map_aware
        tgt_x = as_feature_map_aware if self.attn_to_map_aware_feature else as_feature
        if self.detach_tgt:
            tgt_x = tgt_x.detach()

        padding = ~as_valid
        if self.mask_self_agent:
            if self.attn_mask is None:
                self.attn_mask = torch.eye(n_agent, device=as_valid.device, dtype=torch.bool)

            invalid_batch = as_valid.sum(-1) == 1  # only one agent valid
            if invalid_batch.any():
                valid_batch = ~invalid_batch
                x_reduced = x[valid_batch]
                tgt_x_reduced = tgt_x[valid_batch]
                padding_reduced = padding[valid_batch]
                x_reduced, attn_weights_reduced = self.transformer(
                    src=x_reduced,
                    src_padding_mask=padding_reduced,
                    tgt=tgt_x_reduced,
                    tgt_padding_mask=padding_reduced,
                    need_weights=need_weights,
                    attn_mask=self.attn_mask,
                )

                x = x.masked_fill(valid_batch[:, None, None], 0.0)
                x[valid_batch] = x_reduced

                if need_weights:
                    n_batch, n_agent = as_valid.shape
                    attn_weights = torch.zeros([n_batch, n_agent, n_agent], device=x.device, dtype=x.dtype)
                    attn_weights[valid_batch] = attn_weights_reduced
                else:
                    attn_weights = None
            else:
                x, attn_weights = self.transformer(
                    src=x,  # [n_batch(*n_step), n_agent, hidden_dim]
                    src_padding_mask=padding,  # [n_batch(*n_step), n_agent]
                    tgt=tgt_x,
                    tgt_padding_mask=padding,
                    need_weights=need_weights,
                    attn_mask=self.attn_mask,
                )
        else:
            x, attn_weights = self.transformer(
                src=x,  # [n_batch(*n_step), n_agent, hidden_dim]
                src_padding_mask=padding,  # [n_batch(*n_step), n_agent]
                tgt=tgt_x,
                tgt_padding_mask=padding,
                need_weights=need_weights,
                attn_mask=None,
            )

        if valid_dim == 3:
            # [n_step, n_batch * n_agent, hidden_dim] -> [n_batch, n_step, n_agent, hidden_dim]
            x = x.view([n_batch, n_step, n_agent, self.hidden_dim])
        return x, attn_weights
