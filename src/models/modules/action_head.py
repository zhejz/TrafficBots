# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
from torch import Tensor, nn
import torch
from .distributions import DiagGaussian, MyDist
from .mlp import MLP


class ActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        use_layernorm: bool,
        log_std: Optional[float] = None,
        branch_type: bool = False,
    ) -> None:
        super().__init__()
        self.branch_type = branch_type
        self.out_dim = action_dim

        if self.branch_type:
            self.mlp_mean = nn.ModuleList(
                [
                    MLP([hidden_dim, hidden_dim, self.out_dim], end_layer_activation=False, use_layernorm=use_layernorm)
                    for _ in range(3)
                ]
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = nn.ModuleList(
                    [
                        MLP(
                            [hidden_dim, hidden_dim, self.out_dim],
                            end_layer_activation=False,
                            use_layernorm=use_layernorm,
                        )
                        for _ in range(3)
                    ]
                )
            else:
                self.log_std = nn.ParameterList(
                    [nn.Parameter(log_std * torch.ones(self.out_dim), requires_grad=True) for _ in range(3)]
                )

        else:
            self.mlp_mean = MLP(
                [hidden_dim, hidden_dim, self.out_dim], end_layer_activation=False, use_layernorm=use_layernorm
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = MLP(
                    [hidden_dim, hidden_dim, self.out_dim], end_layer_activation=False, use_layernorm=use_layernorm
                )
            else:
                self.log_std = nn.Parameter(log_std * torch.ones(self.out_dim), requires_grad=True)

    def forward(self, x: Tensor, valid: Tensor, agent_type: Tensor) -> Tuple[MyDist]:
        """
        Args:
            x: [n_batch, n_agent, hidden_dim]
            valid: [n_batch, n_agent], bool
            agent_type: [n_batch, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]

        Returns:
            action_dist.mean: [n_batch, n_agent, action_dim]
            action_dist.covariance_matrix: [n_batch, n_agent, action_dim, action_dim]
        """
        n_batch, n_agent, n_type = agent_type.shape
        if self.branch_type:
            mask_type = agent_type & valid.unsqueeze(-1)  # [n_batch, n_agent, 3]
            mean = 0
            for i in range(n_type):
                mean += self.mlp_mean[i](x, mask_type[:, :, i])

            log_std = 0
            if self.log_std is None:
                for i in range(n_type):
                    log_std += self.mlp_log_std[i](x, mask_type[:, :, i])
            else:
                for i in range(n_type):
                    # [n_batch, n_agent, self.out_dim]
                    log_std += (
                        self.log_std[i][None, None, :]
                        .expand(n_batch, n_agent, -1)
                        .masked_fill(~mask_type[:, :, [i]], 0)
                    )
        else:
            mean = self.mlp_mean(x, valid)
            if self.log_std is None:
                # [n_batch, n_agent, self.out_dim]
                log_std = self.mlp_log_std(x, valid)
            else:
                # [self.out_dim] -> [n_batch, n_agent, self.out_dim]
                log_std = self.log_std[None, None, :].expand(n_batch, n_agent, -1)

        action_dist = DiagGaussian(mean, log_std)
        return action_dist
