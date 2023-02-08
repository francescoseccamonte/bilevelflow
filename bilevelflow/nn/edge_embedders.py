# Copyright 2023 Francesco Seccamonte

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MLP acting on every edge of the inputt graph

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import MessagePassing


class EdgeMLP(MessagePassing):
    """
    Multi-Layer Perceptron to be applied edge-wise
    """
    def __init__(self, n_node_feat, n_edge_feat, dev = torch.device("cpu"), **kwargs):
        super().__init__(**kwargs)
        # Nonlinear function (MLP) transforming node and optional edge features into messages
        self.edge_nonlin = torch.nn.Sequential(
            torch.nn.Linear(2 * n_node_feat + n_edge_feat, 1),
            torch.nn.ReLU()
        )
        self.to(dev)

    def forward(self, x : Tensor, edge_index : Adj, edge_attr : Tensor):
        # Return messages themselves
        return self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

    def edge_update(self, x_i : Tensor, x_j : Tensor, edge_attr : OptTensor = None):
        # Note: x_i corresponds to nodes in edge_index[:,1], x_j to nodes in edge_index[:,0]
        if edge_attr is None:
            tmp = torch.cat((x_i, x_j), dim=1)
        else:
            tmp = torch.cat((x_i, x_j, edge_attr), dim=1)

        return self.edge_nonlin(tmp)
