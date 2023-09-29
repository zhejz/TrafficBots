# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from torch import Tensor, nn
from omegaconf import DictConfig
import torch
from .mlp import MLP


class AddLatentGoal(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        dummy: bool,
        mode: str,
        n_layer_mlp_in: int,
        n_layer_mlp_out: int,
        mlp_in_cfg: DictConfig,
        mlp_out_cfg: DictConfig,
        res_cat: bool = False,
        res_add: bool = False,
    ) -> None:
        super().__init__()
        assert mode in ["add", "mul", "cat"]
        self.mode = mode
        self.dummy = dummy
        self.res_cat = res_cat
        self.res_add = res_add

        if not self.dummy:
            self.mlp_in = MLP([in_dim] + [hidden_dim] * n_layer_mlp_in, **mlp_in_cfg)

            if self.mode == "cat":
                out_dim = hidden_dim * 2
            else:
                out_dim = hidden_dim

            self.mlp_out = MLP([out_dim] + [hidden_dim] * n_layer_mlp_out, **mlp_out_cfg)

            if self.res_cat:
                self.mlp_res_cat = MLP([hidden_dim * 2 + in_dim] + [hidden_dim] * n_layer_mlp_out, **mlp_out_cfg)

    def forward(self, x: Tensor, x_valid: Tensor, z: Optional[Tensor], z_valid: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: [n_batch, n_agent, hidden_dim]
            x_valid: [n_batch, n_agent]
            z: [n_batch, n_agent, hidden_dim], latent or goal
            z_valid: [n_batch, n_agent]

        Returns:
            h: [n_batch, n_agent, hidden_dim], x combined with z
        """
        if self.dummy:
            h = x
        else:
            z = self.mlp_in(z, z_valid)

            if self.mode == "add":
                h = x + z
            elif self.mode == "mul":
                h = x * z
            else:
                h = torch.cat([x, z], dim=-1)

            h = self.mlp_out(h)

            if self.res_cat:
                h = self.mlp_res_cat(torch.cat([x, h, z], dim=-1))

            h = h.masked_fill(~z_valid.unsqueeze(-1), 0)
            if self.res_add:  # h+x if z_valid else x
                h = h + x
            else:  # h if z_valid else x
                h = h + x.masked_fill(z_valid.unsqueeze(-1), 0)

        return h.masked_fill(~x_valid.unsqueeze(-1), 0)
