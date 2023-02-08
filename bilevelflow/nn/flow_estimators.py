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

# All flow estimators
# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material

from abc import ABC, abstractmethod
import torch

from bilevelflow.utils import *
from .tikhonov import *
from .neuralnets import *
from .bilevel_implicit import *


class FlowEstimator(ABC):
    '''
        Generic class for flow estimation
    '''
    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        self.G = G
        self.features = features
        self.params = params
        self.verbose = verbose
        self.dev = dev

    @abstractmethod
    def trainloop(self, train_flows, valid_flows):
        pass

    @abstractmethod
    def predict(self, test_flows):
        pass


class MinDiv(FlowEstimator):
    '''
        Divergence minimization

        From https://arxiv.org/abs/1905.07451
    '''
    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        super().__init__(G, features, params, verbose, dev)

    def trainloop(self, train_flows, valid_flows):
        n_iter = self.params['n_iter']
        lr = self.params['lr']
        early_stop = self.params['early_stop']
        nonneg = self.params['nonneg']
        min_lamb = self.params['min_lamb']
        max_lamb = self.params['max_lamb']
        priors = self.params['priors']

        A, b, self.index = lsq_matrix_flow(self.G, train_flows)
        # gss to find optimal regularization parameter
        lamb, loss = gss(fun_opt_lamb, [self.G, self.index, valid_flows, A, b, n_iter, lr, nonneg], min_lamb, max_lamb, dev=self.dev)

        print("lamb = ", lamb)

        reg_vec = lamb * torch.ones((A.shape[1],1), device=self.dev)
        x_prior = get_prior(self.G, priors, train_flows, self.index, self.dev)

        self.tk = Tikhonov(A, b, n_iter, lr, nonneg, self.dev)
        self.tk.solve(reg_vec, x_prior)
        # x_init = initialize_flows(A.shape[1], zeros=True, dev=self.dev)
        # self.tk.trainloop(reg_vec, x_init, x_prior, verbose=self.verbose)

    def predict(self, test_flows):
        return get_dict_flows_from_tensor(self.index, self.tk.x, test_flows)


class MLPFlowPred(FlowEstimator):
    '''
        Flow estimator using MLP

        From https://openreview.net/forum?id=l0V53bErniB
    '''
    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        super().__init__(G, features, params, verbose, dev)

        n_iter = self.params['n_iter']
        lr = self.params['lr']
        early_stop = self.params['early_stop']
        n_hidden = self.params['n_hidden']

        n_features = self.features[list(self.features.keys())[0]].shape[0]

        self.net = Net(n_features, n_hidden, n_iter, lr, early_stop, torch.nn.ReLU(), self.dev)

    def trainloop(self, train_flows, valid_flows):

        train_feat_tensor = get_tensor_features(self.G, self.features, train_flows, self.dev)
        valid_feat_tensor = get_tensor_features(self.G, self.features, valid_flows, self.dev)
        train_flows_tensor = get_tensor_flows(self.G, train_flows, self.dev)
        valid_flows_tensor = get_tensor_flows(self.G, valid_flows, self.dev)

        self.net.trainloop(train_feat_tensor, train_flows_tensor, valid_feat_tensor, valid_flows_tensor, verbose=self.verbose)

    def predict(self, test_flows):
        self.net.eval()
        test_feat_tensor = get_tensor_features(self.G, self.features, test_flows, self.dev)
        pred = self.net.forward(test_feat_tensor)

        return get_dict_flows(self.G, pred, test_flows)


class GCNFlowPred(FlowEstimator):
    '''
        Flow estimator using GCN

        From https://openreview.net/forum?id=l0V53bErniB
    '''
    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        super().__init__(G, features, params, verbose, dev)

        n_iter = self.params['n_iter']
        lr = self.params['lr']
        early_stop = self.params['early_stop']
        n_hidden = self.params['n_hidden']
        nonneg = self.params['nonneg']

        self.torchg, self.edge_map = create_nn_line_graph(self.G, self.features, nonneg, backend='torchg', dev=self.dev)
        lamb_max_comp = torch_geometric.transforms.LaplacianLambdaMax(normalization='sym')
        lamb_max_comp(self.torchg)

        n_features = self.features[list(self.features.keys())[0]].shape[0]

        self.net = GCN(n_features, n_hidden, n_iter, lr, self.torchg.lambda_max, early_stop, torch.nn.ReLU(), self.dev)

    def trainloop(self, train_flows, valid_flows):

        train_feat_ids = get_feat_ids(self.G, train_flows, self.edge_map, self.dev)
        valid_feat_ids = get_feat_ids(self.G, valid_flows, self.edge_map, self.dev)
        train_flows_tensor = get_tensor_flows(self.G, train_flows, self.dev)
        valid_flows_tensor = get_tensor_flows(self.G, valid_flows, self.dev)

        self.net.trainloop(self.torchg, self.edge_map, train_feat_ids, train_flows_tensor, valid_feat_ids, valid_flows_tensor, verbose=self.verbose)
        
    def predict(self, test_flows):
        self.net.eval()
        test_feat_ids = get_feat_ids(self.G, test_flows, self.edge_map, self.dev)
        outputs = self.net.forward(self.torchg)
        
        pred = outputs[test_feat_ids]
        
        return get_dict_flows(self.G, pred, test_flows)


class BilGCNIMP(FlowEstimator):
    '''
        Flow estimator combining GCN and
        divergence minimization via bilevel
        optimization and Implicit Differentiation
    '''

    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        super().__init__(G, features, params, verbose, dev)

        outer_n_iter = self.params['outer_n_iter']
        outer_lr = self.params['outer_lr']
        early_stop = self.params['early_stop']
        nonneg = self.params['nonneg']
        n_hidden = self.params['n_hidden']
        n_folds = self.params['n_folds']
        priors = self.params['priors']
        lamb = self.params['lambda']

        n_features = self.features[list(self.features.keys())[0]].shape[0]

        self.torchg, self.edge_map = create_nn_line_graph(self.G, self.features, nonneg, backend='torchg', dev=self.dev)
        lamb_max_comp = torch_geometric.transforms.LaplacianLambdaMax(normalization='sym')
        lamb_max_comp(self.torchg)

        self.gcn_lf = GCNwithQP(self.G, self.torchg, priors, lamb, n_folds, self.edge_map,
                                n_features, n_hidden, outer_n_iter, outer_lr, self.torchg.lambda_max, early_stop, torch.nn.ReLU(),
                                nonneg, dev=self.dev)

    def trainloop(self, train_flows, valid_flows):
        self.gcn_lf.trainloop(train_flows, valid_flows, verbose=self.verbose)

    def predict(self, test_flows):
        return get_dict_flows_from_tensor(self.gcn_lf.index, self.gcn_lf.x, test_flows)


class BilGATIMP(FlowEstimator):
    '''
        Flow estimator combining GAT and
        divergence minimization via bilevel
        optimization and Implicit Differentiation
    '''

    def __init__(self, G, features, params, verbose=False, dev=torch.device("cpu")):
        super().__init__(G, features, params, verbose, dev)

        outer_n_iter = self.params['outer_n_iter']
        outer_lr = self.params['outer_lr']
        early_stop = self.params['early_stop']
        nonneg = self.params['nonneg']
        n_hidden = self.params['n_hidden']
        n_node_emb = self.params['n_node_emb']
        n_folds = self.params['n_folds']
        priors = self.params['priors']
        lamb = self.params['lambda']
        flows = self.params['flows']

        n_features = self.features[list(self.features.keys())[0]].shape[0]

        self.torchg, self.edge_map = pyg_from_networkx(G, self.features, flows, nonneg, dev=self.dev)

        self.gcn_lf = GATwithQP(self.G, self.torchg, priors, lamb, n_folds, self.edge_map,
                                n_features, n_hidden, n_node_emb, outer_n_iter, outer_lr, early_stop, torch.nn.ReLU(),
                                nonneg, dev=self.dev)

    def trainloop(self, train_flows, valid_flows):
        self.gcn_lf.trainloop(train_flows, valid_flows, verbose=self.verbose)

    def predict(self, test_flows):
        return get_dict_flows_from_tensor(self.gcn_lf.index, self.gcn_lf.x, test_flows)
