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

# Various utilities to setup and run flow estimators
# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material


import networkx as nx
import numpy as np
import torch


def normalize_features(features):
    """Normalizes features using standard scaler"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    n_features = features[list(features.keys())[0]].shape[0]

    feat_matrix = np.zeros((len(features), n_features))

    i = 0
    for e in features:
        feat_matrix[i] = features[e]
        i = i + 1

    scaler.fit(feat_matrix)
    feat_matrix = scaler.transform(feat_matrix)

    norm_features = {}

    i = 0
    for e in features:
        norm_features[e] = feat_matrix[i]
        i = i + 1

    return norm_features


def make_non_neg_norm(G, flows, features):
    '''
        Converts flow estimation instance to a non-negative
        one, i.e. where every flow is non-negative.
    '''
    new_flows = {}
    new_G = nx.DiGraph()
    new_feat = {}

    max_flow = np.max(list(flows.values()))

    for e in G.edges():
        if e in flows:
            if flows[e] < 0:
                new_e = (e[1], e[0])
                new_flows[new_e] = -flows[e] / max_flow
                new_G.add_edge(e[1], e[0])
                new_feat[new_e] = features[e]
            else:
                new_G.add_edge(e[0], e[1])
                new_flows[e] = flows[e] / max_flow
                new_feat[e] = features[e]
        else:
            new_G.add_edge(e[0], e[1])
            new_feat[e] = features[e]

    return new_G, new_flows, new_feat


def zero_prior(G):
    prior = {}
    for e in G.edges():
        prior[e] = 0.

    return prior


def lsq_matrix_flow(G, train_flows):
    '''
        Generates matrix A and vector b to solve flow problem
        as least-squares, also returns index for recovering
        edges from solution.
    '''
    B = nx.incidence_matrix(G, oriented=True)
    f0 = np.zeros(G.number_of_edges())
    sigma = np.zeros((G.number_of_edges(), G.number_of_edges()-len(train_flows)), dtype=float)
    
    index = {}
    
    i = -1
    valid_ctr = -1
    for ei in G.edges():
        i = i + 1
        if ei in train_flows:
            f0[i] = train_flows[ei]
        else:
            valid_ctr = valid_ctr + 1
            sigma[i, valid_ctr] = 1.0
            index[ei] = valid_ctr
    
    return B.dot(sigma), -B.dot(f0), index


def lsq_matrix_flow_pyg(pyg, train_flows, edge_map):
    '''
        Generates matrix A and vector b to solve flow problem
        as least-squares, also returns index for recovering
        edges from solution.
    '''
    dev = pyg.B.device
    B = pyg.B
    n_edges = pyg.B.shape[1]
    f0 = torch.zeros(n_edges, 1, dtype=torch.float, device=dev)
    sigma = torch.zeros(n_edges, (n_edges - len(train_flows)), dtype=torch.float, device=dev)

    index = {}

    valid_ctr = -1
    for i in range(n_edges):
        ei = edge_map[i]
        if ei in train_flows:
            f0[i] = train_flows[ei]
        else:
            valid_ctr = valid_ctr + 1
            sigma[i, valid_ctr] = 1.0
            index[ei] = valid_ctr

    return B @ sigma, -B @ f0, index


def get_dict_flows_from_tensor(index, x, test_flows):
    '''
        Extract x as a dictionary from tensor.
    '''
    flows = {}
    for e in test_flows:
        flows[e] = x[index[e],0].item()
            
    return flows


def get_tensor_features(G, features, flows, dev=torch.device("cpu")):
    '''
    '''

    n_features = features[list(features.keys())[0]].shape[0]
    feat = np.zeros((len(flows), n_features))

    i = 0
    for e in G.edges():
        if e in flows:
            feat[i] = features[e]
            i = i + 1

    return torch.as_tensor(feat, dtype=torch.float, device=dev)


def get_tensor_flows(G, flows, dev=torch.device("cpu")):
    '''
    '''

    f = np.zeros(len(flows))

    i = 0
    for e in G.edges():
        if e in flows:
            f[i] = flows[e]
            i = i + 1

    return torch.as_tensor(f, dtype=torch.float, device=dev)


def get_dict_flows(G, tf, edges):
    '''
    '''
    flows = {}

    i = 0
    for e in G.edges():
        if e in edges:
            flows[e] = tf[i, 0].item()
            i = i + 1

    return flows


def get_feat_ids(G, flows, edge_map, dev=torch.device("cpu")):
    '''
        Extracts feature representation for GNNLearnFlow
    '''
    feat_ids = []

    for e in G.edges():
        if e in flows:
            idx = edge_map[e]
            feat_ids.append(idx)

    return torch.tensor(feat_ids, device=dev)


def get_prior(G, priors, train, index, dev=torch.device("cpu")):
    '''
    '''

    prior_flows = np.zeros((G.number_of_edges()-len(train), 1))

    for e in G.edges():
        if e not in train:
            i = index[e]
            prior_flows[i,0] = priors[e]

    return torch.as_tensor(prior_flows, dtype=torch.float, device=dev)


def get_fold_flow_data(G, train, valid, dev=torch.device("cpu")):
    '''
        Gets tensor representation of validation flows
        and matrix to map from test flows to validation flows
    '''
    flows = torch.zeros(len(valid), 1, dtype=torch.float, device=dev)

    i = 0
    for e in valid:
        flows[i][0] = valid[e]

        i = i + 1

    mapp = torch.zeros(len(valid), (G.number_of_edges() - len(train)), dtype=torch.float, device=dev)

    j = 0
    for ej in G.edges():
        if ej not in train:
            i = 0
            for ei in valid:
                if ei == ej:
                    mapp[i][j] = 1.

                i = i + 1

            j = j + 1

    return flows, mapp


def get_fold_flow_data_pyg(pyg, train, valid, edge_map, dev=torch.device("cpu")):
    '''
        Gets tensor representation of validation flows
        and matrix to map from test flows to validation flows
    '''
    flows = torch.zeros(len(valid), 1, dtype=torch.float, device=dev)

    i = 0
    for e in valid:
        flows[i][0] = valid[e]

        i = i + 1

    n_edges = pyg.B.shape[1]

    mapp = torch.zeros(len(valid), (n_edges - len(train)), dtype=torch.float, device=dev)

    j = 0
    for k in range(n_edges):
        ej = edge_map[k]
        if ej not in train:
            i = 0
            for ei in valid:
                if ei == ej:
                    mapp[i][j] = 1.
                    break

                i = i + 1

            j = j + 1

    return flows, mapp


def get_out_of_train_features_and_prior(G, features, priors, train, index, dev=torch.device("cpu")):
    '''
        Feature representation for test edges in MLPLearnFlow
    '''
    n_features = features[list(features.keys())[0]].shape[0]

    feat = torch.zeros((G.number_of_edges() - len(train)), n_features, dtype=torch.float, device=dev)
    prior_flows = torch.zeros((G.number_of_edges() - len(train)), 1, dtype=torch.float, device=dev)

    for e in G.edges():
        if e not in train:
            i = index[e]
            feat[i][:] = torch.from_numpy(features[e])
            prior_flows[i][0] = priors[e]

    return feat, prior_flows


def get_test_feat_ids_and_priors(G, train, edge_map, priors, index, dev=torch.device("cpu")):
    '''
        Extracts feature representation for GNNLearnFlow
    '''
    feat_ids = torch.zeros((G.number_of_edges() - len(train)), dtype=torch.long, device=dev)
    prior_flows = torch.zeros((G.number_of_edges() - len(train)), 1, dtype=torch.float, device=dev)

    for e in G.edges():
        if e not in train:
            idx = edge_map[e]
            i = index[e]
            feat_ids[i] = idx
            prior_flows[i][0] = priors[e]

    return feat_ids, prior_flows


def get_test_feat_ids_and_priors_pyg(pyg, train, edge_map, priors, index, dev=torch.device("cpu")):
    '''
        Extracts feature representation for GNNLearnFlow
    '''
    n_edges = pyg.B.shape[1]
    feat_ids = torch.zeros((n_edges - len(train)), dtype=torch.long, device=dev)
    prior_flows = torch.zeros((n_edges - len(train)), 1, dtype=torch.float, device=dev)

    for i in range(n_edges):
        e = edge_map[i]
        if e not in train:
            e_idx = index[e]
            feat_ids[e_idx] = i
            prior_flows[e_idx][0] = priors[e]

    return feat_ids, prior_flows

def initialize_flows(n_edges, zeros=False, dev=torch.device("cpu")):
    '''
    '''
        
    if zeros is True:
        flows = np.zeros((n_edges,1))
    else:
        flows = np.random.random((n_edges,1))
        flows = flows / np.max(flows)

    return torch.tensor(flows, requires_grad=True, dtype=torch.float, device=dev)
