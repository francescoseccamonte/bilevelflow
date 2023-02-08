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

# Classes performing flow estimation with graph neural nets
# and divergence minimization via
# bilevel optimization and Implicit Differentiation (qpth library)


import torch
import torch_geometric
import qpth

from bilevelflow.utils import *
from .edge_embedders import EdgeMLP


class GCNwithQP(torch.nn.Module):
    '''
        2-layer graph convolutional network,
        with an additional QP layer.
        For GCN, it uses torch_geometric.
        The GCN acts on the line graph
    '''

    def __init__(self, G, torchg, priors, lamb, n_folds, edge_map,
                 n_features, n_hidden, n_iter, lr, lamb_max, early_stop=10, output_activation=torch.relu,
                 nonneg=False, dev=torch.device("cpu")):
        super().__init__()

        self.early_stop = early_stop

        self.G = G
        self.torchg = torchg
        self.nonneg = nonneg
        self.edge_map = edge_map
        self.n_folds = n_folds
        self.priors = priors
        self.lamb = lamb

        self.conv1 = torch_geometric.nn.ChebConv(n_features, n_hidden, 2)
        self.conv2 = torch_geometric.nn.ChebConv(n_hidden, 1, 2)

        self.n_iter = n_iter
        self.lr = lr
        self.early_stop = early_stop
        self.output_activation = output_activation
        self.lamb_max = lamb_max

        self.to(dev)
        self.dev = dev


    def forward(self, g, A, b, feat_ids, lamb, x_prior, save_weights=False):
        # Convolutional layers
        h = torch.relu(self.conv1(x=g.x, edge_index=g.edge_index, lambda_max=self.lamb_max))
        h = lamb*self.output_activation(self.conv2(x=h, edge_index=g.edge_index, lambda_max=self.lamb_max))

        hfold = h[feat_ids] + 1e-5   # ensuring the regularizer is positive definite
        if save_weights:
            self.trained_reg = hfold

        # QP (implicit) layer
        if self.nonneg:
            Gineq = -torch.eye(hfold.shape[0], device=self.dev)
            hineq = torch.zeros((hfold.shape[0],), device=self.dev)
        else:
            # Workaround: qpth seems not to support unconstrained problems
            Gineq = torch.zeros((1, hfold.shape[0]), device=self.dev)
            hineq = torch.tensor([0.1], dtype=torch.float, device=self.dev)

        # empty tensor for equality constraints of QP Layer
        e = torch.nn.Parameter().to(self.dev)

        f = qpth.qp.QPFunction(verbose=-1, check_Q_spd=False)((torch.mm(torch.transpose(A, 0,1), A) + torch.diagflat(hfold)),
                                                              (-torch.mul(x_prior, hfold).squeeze()-torch.mm(torch.transpose(b, 0,1), A).squeeze()),
                                                              Gineq, hineq, e, e)

        return f.T

    def trainloop(self, train_flows, valid_flows, verbose=False):
        '''
            Outer problem
        '''
        super().train()
        int_folds = generate_folds({**train_flows, **valid_flows}, self.n_folds)  # No extra validation

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_losses = []

        X = []

        for (int_train, int_test) in int_folds:
            X.append(torch.zeros((self.G.number_of_edges() - len(int_train), 1), dtype=torch.float, device=self.dev))

        loss_func = torch.nn.MSELoss()

        for epoch in range(self.n_iter):
            optimizer.zero_grad()

            train_loss = torch.zeros(1, device=self.dev)

            f = 0
            for (int_train, int_test) in int_folds:
                A, b, index = lsq_matrix_flow(self.G, int_train)
                A = torch.as_tensor(A, dtype=torch.float, device=self.dev)
                b = torch.as_tensor(b.reshape((b.size, 1)), dtype=torch.float, device=self.dev)
                int_test_flows, mapp = get_fold_flow_data(self.G, int_train, int_test, self.dev)
                feat_ids, prior_flows = get_test_feat_ids_and_priors(self.G, int_train, self.edge_map, self.priors,
                                                                     index, self.dev)

                # GCN + QP forward pass
                x = self.forward(self.torchg, A, b, feat_ids, self.lamb, prior_flows)

                loss = loss_func(torch.mm(mapp, x), int_test_flows)  # validation loss

                X[f] = x.clone().detach()
                train_loss = train_loss + loss
                loss.backward()
                optimizer.step()

                f = f + 1

            train_losses.append(train_loss.item())

            print("epoch: ", epoch, " outer train loss = ", train_loss.item())

            if epoch > self.early_stop and train_losses[-1] > np.mean(train_losses[-(self.early_stop + 1):-1]):
                if verbose is True:
                    print("Early stopping...")
                break


        # Save best estimate for compatibility with other models
        A, b, self.index = lsq_matrix_flow(self.G, {**train_flows, **valid_flows})
        A = torch.as_tensor(A, dtype=torch.float, device=self.dev)
        b = torch.as_tensor(b.reshape((b.size, 1)), dtype=torch.float, device=self.dev)
        feat_ids, prior_flows = get_test_feat_ids_and_priors(self.G, {**train_flows, **valid_flows}, self.edge_map,
                                                             self.priors, self.index, self.dev)

        self.x = self.forward(self.torchg, A, b, feat_ids, self.lamb, prior_flows, save_weights=True)


class GATwithQP(torch.nn.Module):
    '''
        2-layer graph neural network,
        with an additional QP layer.
        It uses GAT layers from torch_geometric.
        The GNN acts on the original graph.
    '''

    def __init__(self, G, torchg, priors, lamb, n_folds, edge_map,
                 n_features, n_hidden, n_node_emb, n_iter, lr, early_stop=10, output_activation=torch.relu,
                 nonneg=False, dev=torch.device("cpu")):
        super().__init__()

        self.early_stop = early_stop

        self.G = G
        self.torchg = torchg
        self.nonneg = nonneg
        self.edge_map = edge_map
        self.n_folds = n_folds
        self.priors = priors
        self.lamb = lamb

        self.l1 = torch_geometric.nn.GATConv(1, n_hidden, edge_dim=n_features)
        self.l2 = torch_geometric.nn.GATConv(n_hidden, n_node_emb, edge_dim=n_features)
        self.l3 = EdgeMLP(n_node_emb, n_features, dev)

        self.n_iter = n_iter
        self.lr = lr
        self.early_stop = early_stop
        self.output_activation = output_activation

        self.to(dev)
        self.dev = dev


    def forward(self, g, A, b, feat_ids, lamb, x_prior, save_weights=False):
        # Convolutional layers
        h = torch.relu(self.l1(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr))
        h = self.output_activation(self.l2(x=h, edge_index=g.edge_index, edge_attr=g.edge_attr))
        h = lamb*self.l3(x=h, edge_index=g.edge_index, edge_attr=g.edge_attr)

        hfold = h[feat_ids] + 1e-5   # ensuring the regularizer is positive definite
        if save_weights:
            self.trained_reg = hfold

        # QP (implicit) layer
        if self.nonneg:
            Gineq = -torch.eye(hfold.shape[0], device=self.dev)
            hineq = torch.zeros((hfold.shape[0],), device=self.dev)
        else:
            # Workaround: qpth seems not to support unconstrained problems
            Gineq = torch.zeros((1, hfold.shape[0]), device=self.dev)
            hineq = torch.tensor([0.1], dtype=torch.float, device=self.dev)

        # empty tensor for equality constraints of QP Layer
        e = torch.nn.Parameter().to(self.dev)

        f = qpth.qp.QPFunction(verbose=-1, check_Q_spd=False)((torch.mm(torch.transpose(A, 0,1), A) + torch.diagflat(hfold)),
                                                              (-torch.mul(x_prior, hfold).squeeze()-torch.mm(torch.transpose(b, 0,1), A).squeeze()),
                                                              Gineq, hineq, e, e)

        return f.T

    def trainloop(self, train_flows, valid_flows, verbose=False):
        '''
            Outer problem
        '''
        super().train()
        int_folds = generate_folds({**train_flows, **valid_flows}, self.n_folds)  # No extra validation

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_losses = []

        X = []

        for (int_train, int_test) in int_folds:
            X.append(torch.zeros((self.G.number_of_edges() - len(int_train), 1), dtype=torch.float, device=self.dev))

        loss_func = torch.nn.MSELoss()

        for epoch in range(self.n_iter):
            optimizer.zero_grad()

            train_loss = torch.zeros(1, device=self.dev)

            f = 0
            for (int_train, int_test) in int_folds:
                A, b, index = lsq_matrix_flow_pyg(self.torchg, int_train, self.edge_map)
                int_test_flows, mapp = get_fold_flow_data_pyg(self.torchg, int_train, int_test, self.edge_map, self.dev)
                feat_ids, prior_flows = get_test_feat_ids_and_priors_pyg(self.torchg, int_train, self.edge_map, self.priors,
                                                                     index, self.dev)

                # GCN + QP forward pass
                x = self.forward(self.torchg, A, b, feat_ids, self.lamb, prior_flows)

                loss = loss_func(torch.mm(mapp, x), int_test_flows)  # validation loss

                X[f] = x.clone().detach()
                train_loss = train_loss + loss
                loss.backward()
                optimizer.step()

                f = f + 1

            train_losses.append(train_loss.item())

            print("epoch: ", epoch, " outer train loss = ", train_loss.item())

            if epoch > self.early_stop and train_losses[-1] > np.mean(train_losses[-(self.early_stop + 1):-1]):
                if verbose is True:
                    print("Early stopping...")
                break


        # Save best estimate for compatibility with other models
        A, b, self.index = lsq_matrix_flow_pyg(self.torchg, {**train_flows, **valid_flows}, self.edge_map)
        b = b.reshape(len(b), 1)
        feat_ids, prior_flows = get_test_feat_ids_and_priors_pyg(self.torchg, {**train_flows, **valid_flows},
                                                                 self.edge_map, self.priors, self.index, self.dev)

        self.x = self.forward(self.torchg, A, b, feat_ids, self.lamb, prior_flows, save_weights=True)
