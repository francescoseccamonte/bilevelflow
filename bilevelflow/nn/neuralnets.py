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

# Simple MLP and GCN
# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material


import torch
import torch.optim as optim
import numpy as np
import torch_geometric


class Net(torch.nn.Module):
    '''
        Simple MLP with ReLU activation in the hidden layer
    '''

    def __init__(self, n_features, n_hidden, n_iter, lr, early_stop=10, output_activation=torch.sigmoid,
                 dev=torch.device("cpu")):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(n_features, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, 1)
        self.n_iter = n_iter
        self.lr = lr
        self.early_stop = early_stop
        self.output_activation = output_activation

        self.to(dev)
        self.dev = dev

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.output_activation(self.layer2(x))

        return x

    def trainloop(self, xs_train, ys_train, xs_valid, ys_valid, verbose=False):
        super().train()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()
        valid_losses = []
        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            outputs_train = (self.forward(xs_train).T)[0]
            train_loss = loss_func(outputs_train, ys_train)
            outputs_valid = (self.forward(xs_valid).T)[0]
            valid_loss = loss_func(outputs_valid, ys_valid)
            train_loss.backward()
            optimizer.step()

            valid_losses.append(valid_loss.item())

            if epoch % 100 == 0 and verbose is True:
                print("epoch: ", epoch, " train loss = ", train_loss.item(), " valid loss = ", valid_loss.item())

            if epoch > self.early_stop and valid_losses[-1] > np.mean(valid_losses[-(self.early_stop + 1):-1]):
                if verbose is True:
                    print("Early stopping...")
                break


class GCN(torch.nn.Module):
    '''
        2-layer graph convolutional network.
    '''
    def __init__(self, n_features, n_hidden, n_iter, lr, lamb_max, early_stop=10, output_activation=torch.sigmoid, dev=torch.device("cpu")):
        super(GCN, self).__init__()

        self.conv1 = torch_geometric.nn.ChebConv(n_features, n_hidden, 2)
        self.conv2 = torch_geometric.nn.ChebConv(n_hidden, 1, 2)
        
        self.lr = lr
        self.n_iter = n_iter
        self.early_stop = early_stop
        self.output_activation = output_activation
        self.lamb_max = lamb_max

        self.to(dev)
        self.dev = dev
      
    def forward(self, g):
        h = torch.relu(self.conv1(x=g.x, edge_index=g.edge_index, lambda_max=self.lamb_max))
        h = self.output_activation(self.conv2(x=h, edge_index=g.edge_index, lambda_max=self.lamb_max))
        
        return h
    
    def trainloop(self, pyg, edge_map, xs_train, ys_train, xs_valid, ys_valid, verbose=False):
        super().train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()
        valid_losses = []
        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            outputs = (self.forward(pyg).T)[0]
            train_loss = loss_func(outputs[xs_train], ys_train)
                       
            valid_loss = loss_func(outputs[xs_valid], ys_valid)
            train_loss.backward()
            optimizer.step()
            
            valid_losses.append(valid_loss.item())
            
            if epoch % 1000 == 0 and verbose is True:
                print("epoch: ", epoch, " train loss = ", train_loss.item(), " valid loss = ", valid_loss.item())
                
            if epoch > self.early_stop and valid_losses[-1] > np.mean(valid_losses[-(self.early_stop+1):-1]):
                if verbose is True:
                    print("Early stopping...")
                break
