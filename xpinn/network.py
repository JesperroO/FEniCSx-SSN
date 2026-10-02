"""
XPINN network: configurable tanh MLP via torch.nn.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class NetworkConfig:
    input_dim:   int = 2        # spatial dimension (2 or 3)
    hidden_dim:  int = 32       # units per hidden layer
    n_layers:    int = 2        # number of hidden layers
    activation:  str = "tanh"  # "tanh" | "sin" | "gelu"
    enforce_bc:  bool = False   # 乘以边界距离函数 ∏ x_i(1-x_i)，硬编码 y=0 on ∂Ω


def _activation(name: str) -> nn.Module:
    return {"tanh": nn.Tanh(), "sin": _Sin(), "gelu": nn.GELU()}[name]


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """
    Fully-connected network Rᵈ → R¹ with `n_layers` hidden tanh layers.
    Input dimension is set by cfg.input_dim (2 for 2D, 3 for 3D).

    Parameters
    ----------
    cfg : NetworkConfig
    """

    def __init__(self, cfg: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.enforce_bc = cfg.enforce_bc
        dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.n_layers + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_activation(cfg.activation))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) → (N,)"""
        raw = self.net(x).squeeze(-1)
        if self.enforce_bc:
            # 硬编码单位超立方体 Dirichlet BC：∏ x_i(1-x_i)，在 ∂Ω 上精确为零
            bc_factor = (x * (1.0 - x)).prod(dim=-1)
            return raw * bc_factor
        return raw
