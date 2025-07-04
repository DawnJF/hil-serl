"""MLP网络的PyTorch实现"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim, hidden_dims, output_dim, activate_final=True):
        super().__init__()
        self.activate_final = activate_final

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if activate_final:
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EnsembleMLP(nn.Module):
    """集成MLP，用于Critic网络"""

    def __init__(self, input_dim, hidden_dims, output_dim, ensemble_size=2):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.networks = nn.ModuleList(
            [
                MLP(input_dim, hidden_dims, output_dim, activate_final=False)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, x):
        # 返回所有ensemble成员的输出
        outputs = []
        for network in self.networks:
            outputs.append(network(x).unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (ensemble_size, batch_size, output_dim)
