# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
import torch
from torch import Tensor, nn
from .mlp import MLP


class InputPeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pe_dim: int,
        n_layer: int,
        mlp_dropout_p: Optional[float] = 0.1,
        mlp_use_layernorm: bool = True,
        pe_mode: str = "add",
    ) -> None:
        super().__init__()
        self.pe_mode = pe_mode

        if self.pe_mode == "input":
            mlp_in_dim = attr_dim + pe_dim
            mlp_out_dim = hidden_dim
        elif self.pe_mode == "cat":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim - pe_dim
            assert mlp_out_dim >= 32, f"Make sure pe_dim is smaller than {hidden_dim-32}!"
        elif self.pe_mode == "add":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim
            assert pe_dim == hidden_dim, f"Make sure pe_dim equals to hidden_dim={hidden_dim}!"

        self.mlp = MLP(
            [mlp_in_dim] + [mlp_out_dim] * n_layer,
            dropout_p=mlp_dropout_p,
            use_layernorm=mlp_use_layernorm,
            end_layer_activation=False,
        )

    def forward(self, valid: Tensor, attr: Tensor, pe: Tensor) -> Tensor:
        """
        Args:
            valid: [...], bool
            attr: [..., attr_dim], for input to MLP
            pe: [..., hidden_dim], for input/cat/add to MLP
        Returns:
            feature: [..., hidden_dim] float32
        """
        if self.pe_mode == "input":
            x = self.mlp(torch.cat([attr, pe], dim=-1))
        elif self.pe_mode == "cat":
            x = self.mlp(attr)
            x = torch.cat([x, pe], dim=-1)
        elif self.pe_mode == "add":
            x = self.mlp(attr)
            x = x + pe

        feature = x.masked_fill(~valid.unsqueeze(-1), 0)

        return feature
