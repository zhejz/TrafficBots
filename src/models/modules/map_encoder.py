# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple, List, Optional
import torch
from torch import Tensor, nn
from omegaconf import DictConfig
from .mlp import MLP
from .transformer import TransformerBlock
from .input_pe_encoder import InputPeEncoder


class MapEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pe_dim: int,
        input_pe_encoder: DictConfig,
        tf_cfg: DictConfig,
        densetnt_vectornet: bool = False,
        pool_mode: str = "max",  # max, mean, first
        n_layer: int = 3,
        mlp_dropout_p: Optional[float] = 0.1,
        mlp_use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.densetnt_vectornet = densetnt_vectornet
        self.pool_mode = pool_mode

        self.input_pe_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim, pe_dim=pe_dim, **input_pe_encoder
        )

        if self.densetnt_vectornet:
            self.transformer_densetnt = TransformerBlock(n_layer=n_layer, **tf_cfg)
        else:
            mlp_layers: List[nn.Module] = []
            for _ in range(n_layer - 1):
                mlp_layers.append(
                    MLP([hidden_dim, hidden_dim // 2], dropout_p=mlp_dropout_p, use_layernorm=mlp_use_layernorm)
                )

            if tf_cfg.norm_first:
                end_layer_activation = False
            else:
                end_layer_activation = True
            mlp_layers.append(
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=mlp_dropout_p,
                    use_layernorm=mlp_use_layernorm,
                    end_layer_activation=end_layer_activation,
                )
            )
            self.mlp_layers = nn.ModuleList(mlp_layers)

        self.transformer_self_attn = TransformerBlock(n_layer=1, **tf_cfg)

    def forward(self, map_valid: Tensor, map_attr: Tensor, map_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """
        c.f. VectorNet and SceneTransformer
        Aggregate polyline-level feature. n_pl polylines, n_node nodes per polyline.
        Args:
            map_valid: [n_scene, n_pl, n_pl_node], bool
            map_attr: [n_scene, n_pl, n_pl_node, map_attr_dim]
            map_pe: [n_scene, n_pl, n_pl_node, hidden_dim], for input/cat/add to MLP

        Returns:
            map_feature: float32, [n_scene, n_pl, hidden_dim]
            map_valid: bool, [n_scene, n_pl]
        """
        n_scene, n_pl, n_node = map_valid.shape
        pl_feature = self.input_pe_encoder(map_valid, map_attr, map_pe)

        if self.densetnt_vectornet:
            # map_valid: bool, [n_scene, n_pl, n_node]
            # transformer
            pl_feature = pl_feature.flatten(0, 1)
            map_valid = map_valid.flatten(0, 1)
            pl_feature, _ = self.transformer_densetnt(
                src=pl_feature,  # [n_scene*n_pl, n_node, hidden_dim]
                src_padding_mask=~map_valid,  # [n_scene*n_pl, n_node]
                tgt=pl_feature,  # [n_scene*n_pl, n_node, hidden_dim]
                tgt_padding_mask=~map_valid,  # [n_scene*n_pl, n_node]
                need_weights=False,
            )
            hidden_dim = pl_feature.shape[-1]
            pl_feature = pl_feature.view(n_scene, n_pl, n_node, hidden_dim)
            map_valid = map_valid.view(n_scene, n_pl, n_node)
        else:
            for mlp in self.mlp_layers:
                feature_encoded = mlp(pl_feature, map_valid, float("-inf"))
                feature_pooled = feature_encoded.amax(dim=2, keepdim=True)
                pl_feature = torch.cat((feature_encoded, feature_pooled.expand(-1, -1, n_node, -1)), dim=-1)

        if self.pool_mode == "max":
            pl_feature = pl_feature.masked_fill(~map_valid.unsqueeze(-1), float("-inf"))
            pl_feature = pl_feature.amax(dim=2, keepdim=False)  # [batch_size, n_pl, :]
        elif self.pool_mode == "first":
            pl_feature = pl_feature[:, :, 0]
        elif self.pool_mode == "mean":
            pl_feature = pl_feature.masked_fill(~map_valid.unsqueeze(-1), 0)
            pl_feature = pl_feature.sum(dim=2, keepdim=False)  # [batch_size, n_pl, :]
            pl_feature = pl_feature / (map_valid.sum(dim=-1, keepdim=True) + torch.finfo(pl_feature.dtype).eps)

        pl_valid = map_valid.any(-1)  # [batch_size, n_pl]
        pl_feature = pl_feature.masked_fill(~pl_valid.unsqueeze(-1), 0)

        pl_feature, _ = self.transformer_self_attn(
            src=pl_feature,  # [n_scene, n_pl, hidden_dim]
            src_padding_mask=~pl_valid,  # [n_scene, n_pl]
            tgt=pl_feature,  # [n_scene, n_pl, hidden_dim]
            tgt_padding_mask=~pl_valid,  # [n_scene, c_step]
            need_weights=False,
        )
        return pl_feature, pl_valid
