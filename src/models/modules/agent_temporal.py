# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, Any
import torch
from torch import Tensor, nn


class TemporalAggregate(nn.Module):
    def __init__(self, mode: str) -> None:
        """
        mode in {"max", "last", "max_valid", "last_valid", "mean_valid"}
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [n_batch, n_step, n_agent, hidden_dim], invalid x is padded with 0
            valid: [n_batch, n_step, n_agent]
        Returns:
            x_aggregated: [n_batch, n_agent, hidden_dim]
            valid_aggregated: [n_batch, n_agent]
        """
        if self.mode == "max":
            x_aggregated = x.amax(1)
        elif self.mode == "last":
            x_aggregated = x[:, -1]
        elif self.mode == "max_valid":
            x_aggregated = x.masked_fill(~valid.unsqueeze(-1), -1e3).amax(1)
        elif self.mode == "last_valid":
            n_batch, n_step, n_agent = valid.shape
            idx_last_valid = n_step - 1 - torch.max(valid.flip(1), dim=1)[1]
            x_aggregated = x[torch.arange(n_batch).unsqueeze(1), idx_last_valid, torch.arange(n_agent).unsqueeze(0)]
        elif self.mode == "mean_valid":
            valid_sum = valid.sum(1) + torch.finfo(x.dtype).eps
            x_aggregated = x.sum(1) / valid_sum.unsqueeze(-1)

        valid_aggregated = torch.any(valid, axis=1)
        return x_aggregated.masked_fill(~valid_aggregated.unsqueeze(-1), 0), valid_aggregated


class MultiAgentDummy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor, valid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        return x, h


class MultiAgentGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(hidden_dim, hidden_dim))

    def forward(self, x: Tensor, valid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [n_batch, (n_step), n_agent, hidden_dim]
            valid: [n_batch, (n_step), n_agent]
            h: [num_layers, n_batch * n_agent, hidden_dim] or None
        Returns:
            x_1: [n_batch, (n_step), n_agent, hidden_dim]
            h_1: [num_layers, n_batch * n_agent, hidden_dim]
        """
        n_batch = valid.shape[0]
        n_agent = valid.shape[-1]
        if h is None:
            h = torch.zeros((self.num_layers, n_batch * n_agent, self.hidden_dim), device=x.device)

        if valid.dim() == 3:
            x_1 = []
            for k in range(valid.shape[1]):

                input = x[:, k].flatten(start_dim=0, end_dim=1)  # [n_batch * n_agent, hidden_dim]
                h_1 = []
                for i, layer in enumerate(self.layers):
                    h_1_i = layer(input, h[i])
                    input = h_1_i
                    if i + 1 != self.num_layers:
                        input = self.dropout(input)
                    h_1.append(h_1_i)

                invalid = ~valid[:, k].flatten(start_dim=0, end_dim=1).unsqueeze(-1)  # [n_batch * n_agent, 1]
                h = torch.stack(h_1, dim=0)  # [n_layers, n_batch * n_agent, hidden_dim]
                h.masked_fill_(invalid.unsqueeze(0), 0.0)
                x_1.append(input.masked_fill(invalid, 0.0).view(n_batch, n_agent, self.hidden_dim))
            x_1 = torch.stack(x_1, dim=1)  # [n_batch, n_step,  n_agent, hidden_dim]
            h_1 = None
        elif valid.dim() == 2:
            input = x.flatten(start_dim=0, end_dim=1)  # [n_batch * n_agent, hidden_dim]
            h_1 = []
            for i, layer in enumerate(self.layers):
                h_1_i = layer(input, h[i])
                input = h_1_i
                if i + 1 != self.num_layers:
                    input = self.dropout(input)
                h_1.append(h_1_i)
            invalid = ~valid.flatten(start_dim=0, end_dim=1).unsqueeze(-1)  # [n_batch * n_agent, 1]
            h_1 = torch.stack(h_1, dim=0)  # [n_layers, n_batch * n_agent, hidden_dim]
            h_1.masked_fill_(invalid.unsqueeze(0), 0.0)
            x_1 = input.masked_fill(invalid, 0.0).view(n_batch, n_agent, self.hidden_dim)
        return x_1, h_1


class MultiAgentGRULoop(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout)

    def forward(self, x: Tensor, valid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [n_batch, (n_step), n_agent, hidden_dim]
            valid: [n_batch, (n_step), n_agent]
            h: [num_layers, n_batch * n_agent, hidden_dim] or None
        Returns:
            x_1: [n_batch, (n_step), n_agent, hidden_dim]
            h_1: [num_layers, n_batch * n_agent, hidden_dim]
        """
        n_batch = valid.shape[0]
        n_agent = valid.shape[-1]
        if h is None:
            h = torch.zeros((self.num_layers, n_batch * n_agent, self.hidden_dim), device=x.device)

        if valid.dim() == 3:
            n_step = valid.shape[1]
            x_1 = []
            # [n_step, n_batch * n_agent, hidden_dim]
            x = x.transpose(0, 1).flatten(start_dim=1, end_dim=2)
            # [n_step, n_batch * n_agent, 1]
            invalid = ~valid.transpose(0, 1).flatten(start_dim=1, end_dim=2).unsqueeze(-1)
            for k in range(n_step):
                x_out, h = self.rnn(x[[k]], h)
                h = h.masked_fill(invalid[[k]], 0.0)
                x_1.append(x_out)  # [1, n_batch * n_agent, hidden_dim]
            x_1 = torch.cat(x_1, dim=0)  # [n_step, n_batch*n_agent, hidden_dim]
            x_1 = x_1.masked_fill(invalid, 0.0).view(n_step, n_batch, n_agent, self.hidden_dim).transpose(0, 1)
            h_1 = None
        elif valid.dim() == 2:
            input = x.flatten(start_dim=0, end_dim=1).unsqueeze(0)
            input, h = self.rnn(input, h)
            invalid = ~valid.flatten(start_dim=0, end_dim=1).unsqueeze(-1)  # [n_batch * n_agent, 1]
            h_1 = h.masked_fill(invalid.unsqueeze(0), 0.0)
            x_1 = input[0].masked_fill(invalid, 0.0).view(n_batch, n_agent, self.hidden_dim)
        return x_1, h_1


class MultiAgentGRU(nn.Module):
    # for TrafficSim where interaction_fist=False, temp_aggr_mode="last", forward not masked
    def __init__(self, hidden_dim: int, num_layers: int, dropout: Optional[float]) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout)

    def forward(self, x: Tensor, valid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [n_batch, (n_step), n_agent, hidden_dim]
            valid: [n_batch, (n_step), n_agent]
            h: [num_layers, n_batch * n_agent, hidden_dim] or None
        Returns:
            x: [n_batch, (n_step), n_agent, out_dim]
            h: [num_layers, n_batch * n_agent, hidden_dim]
        """
        if valid.dim() == 3:
            n_batch, n_step, n_agent = valid.shape
        elif valid.dim() == 2:
            n_batch, n_agent = valid.shape
            n_step = 1
            x = x.unsqueeze(dim=1)

        if h is None:
            h = torch.zeros((self.num_layers, n_batch * n_agent, self.hidden_dim), device=x.device)

        x, h = self.rnn((x.transpose(0, 1)).flatten(start_dim=1, end_dim=2), h)

        # [n_step, n_batch * n_agent, hidden_dim] -> [n_batch, n_step, n_agent, hidden_dim]
        x = x.view([n_step, n_batch, n_agent, self.hidden_dim]).transpose(0, 1)

        if valid.dim() == 2:
            x = x.squeeze(axis=1)
            # x = x.masked_fill(~valid.unsqueeze(-1), 0.0)

        return x, h
