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

# Tikhonov regularization problem
# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material


import torch

from bilevelflow.optim import NONNEGSGD
from bilevelflow.utils import *


def fun_opt_lamb(lamb, args, dev=torch.device("cpu")):
    '''
        Auxiliary function for line search for lambda
        in min-div algorithm.
    '''
    G, index, valid_flows, A, b, n_iter, lr, nonneg = args
    reg_vec = lamb * torch.ones((A.shape[1],1), device=dev)
    x_prior = initialize_flows(A.shape[1], zeros=True, dev=dev)
    tk = Tikhonov(A, b, n_iter, lr, nonneg, dev)
    x_init = initialize_flows(A.shape[1], dev=dev)
    tk.trainloop(reg_vec, x_init, x_prior, verbose=False)
    # tk.x = tk.solve(reg_vec, x_prior)
    pred = get_dict_flows_from_tensor(index, tk.x, valid_flows)
    loss = rmse(G, valid_flows, pred, {})

    return loss


class Tikhonov(torch.nn.Module):
    '''
        Solves Tikhonov regularization problem.
        x* = min_x ||Ax-b||_2^2 + lamb^2||x||_q^2
    '''
    def __init__(self, A, b, n_iter, learning_rate, nonneg=False, dev=torch.device("cpu")):
        super(Tikhonov, self).__init__()

        self.to(dev)
        self.dev = dev
        
        self.A = torch.as_tensor(A, dtype=torch.float, device=dev)

        self.b = torch.as_tensor(b.reshape((b.size,1)), dtype=torch.float, device=dev)
        
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.nonneg = nonneg
             
    def forward(self, x, reg_vec, x_prior):
        '''
            returns loss for given x
        '''
        return torch.square(torch.norm(torch.mm(self.A, x)-self.b,2)) + torch.sum(torch.mul(x-x_prior, torch.mul(reg_vec, x-x_prior)))

    def solve(self, reg_vec, x_prior):
        '''
        Solves the Tikhonov regularization problem exactly
        '''

        if self.nonneg:
             raise RuntimeError("Explicit solution not available for non-negative Least-Squares")

        Q = torch.diagflat(reg_vec)

        self.x = torch.linalg.solve((torch.mm(torch.transpose(self.A, 0, 1), self.A) + Q),
                                    (torch.mm(torch.transpose(self.A, 0, 1), self.b) + torch.reshape(torch.mul(reg_vec, x_prior), (self.A.shape[1], 1))))
    
    def trainloop(self, reg_vec, x_init, x_prior, verbose=False):
        '''
            Computes x* given lambda (lamb) parameter
        '''
        super().train()
        losses = []
        self.x =  x_init

        t_reg_vec = reg_vec.detach().clone().to(dtype=torch.float, device=self.dev)

        if self.nonneg:
            optimizer = NONNEGSGD([self.x], lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam([self.x], lr=self.learning_rate)
                
        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            loss = self.forward(self.x, t_reg_vec, x_prior)
            loss.backward()   
            optimizer.step()
            
            if epoch % 1000 == 0 and verbose is True:
                print("epoch: ", epoch, " loss = ", loss.item())
            
            losses.append(loss.item())
