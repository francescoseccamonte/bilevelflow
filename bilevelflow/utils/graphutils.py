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

# Various utilities to work on graphs,
# and converting from networkx to torch_geometric

# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material


import networkx as nx
import numpy as np
import torch
import csv


def read_net(road_net_filename, lcc=False):
    '''
        Reads network from csv, optionally gets
        largest connected component.
    '''
    G = nx.DiGraph()

    with open(road_net_filename, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = r[0]
            v = r[1]

            G.add_edge(u,v)

    if lcc:
        LWCC = sorted(nx.weakly_connected_components(G), key = len, reverse=True)[0]
        return G.subgraph(LWCC)
    else:
        return G


def create_nn_line_graph(G, features, directed=False, backend='torchg', dev=torch.device("cpu")):
    '''
        Creates GNN graph representation
        (torch_geometric)
        for line graph of G (edges as nodes).
        Also returns an edge map: G edges to
        GNN backend graph nodes.
    '''
    G_line = nx.line_graph(G)

    edge_map = {}
    for e in G_line.nodes():
        edge_map[e] = len(edge_map)

    if directed:
        source = torch.zeros((G_line.number_of_nodes() + G_line.number_of_edges()), dtype=torch.int)
        dest = torch.zeros((G_line.number_of_nodes() + G_line.number_of_edges()), dtype=torch.int)
    else:
        source = torch.zeros((G_line.number_of_nodes() + 2 * G_line.number_of_edges()), dtype=torch.int)
        dest = torch.zeros((G_line.number_of_nodes() + 2 * G_line.number_of_edges()), dtype=torch.int)

    idx = 0
    for e in G_line.edges():
        m_i = edge_map[e[0]]
        m_j = edge_map[e[1]]

        source[idx] = m_i
        dest[idx] = m_j

        idx = idx + 1

        if directed is False:
            source[idx] = m_j
            dest[idx] = m_i

            idx = idx + 1

    for v in G_line.nodes():
        m_i = edge_map[v]
        source[idx] = m_i
        dest[idx] = m_i

        idx = idx + 1

    if backend == 'torchg':
        return create_torch_geometric_graph(G_line, source, dest, features, edge_map, dev), edge_map
    else:
        raise RuntimeError(f'backend {backend} not supported yet.')


def create_torch_geometric_graph(G, source, dest, features, edge_map, dev):
    '''
        Creates torch geometric graph representation
        for the networkx graph G, with
        source and destination nodes.
    '''
    from torch_geometric.data import Data
    n_features = features[list(features.keys())[0]].shape[0]
    source, dest = source.type(torch.int64), dest.type(torch.int64)
    torchgraph = Data(x=torch.zeros((G.number_of_nodes(), n_features)), edge_index=torch.vstack((source, dest)))

    for e in G.nodes():
        i = edge_map[e]
        torchgraph.x[i,:] = torch.FloatTensor(features[e])

    return torchgraph.to(dev)


def get_incidence_matrix(edge_index, num_nodes, force_undirected=False):
    """
    Create signed incidence matrix.
    Parameters
    ----------
    edge_index : torch.LongTensor
        Edge index tensor.
    num_nodes : int
        Number of nodes in the graph.
    force_undirected : bool, optional, default=False
        If True, B will represent an undirected graph with arbitrary edge orientation.
        If False, B will have precisely one edge corresponding to each column of edge_index,
        in the same order.
    Returns
    -------
    B : torch.Tensor
        Incidence matrix.
    """
    from torch_geometric.utils import coalesce
    # Get unique undirected edges and impose arbitrary edge orientations
    if force_undirected:
        edge_index_oriented, _ = edge_index.sort(dim=0)
        edge_index = coalesce(edge_index_oriented, num_nodes=num_nodes)

    m = edge_index.shape[1]
    idx = torch.stack([
        torch.cat([edge_index[0], edge_index[1]]),
        torch.arange(m, device=edge_index.device).repeat(2)])
    vals = torch.cat([torch.ones(m), -torch.ones(m)])
    B = torch.sparse_coo_tensor(idx, vals, (num_nodes, m), device=edge_index.device)

    return B


def pyg_from_networkx(G, edge_features, flows, directed=False, dev=torch.device("cpu")):
    """
    Custom function to convert a networkx graph
    into a torch geometric one.
    The pyg graph has no added self-loops.
    This function is not optimized for speed/memory,
    being it run only once as preprocessing.

    Args:
        G: networkx graph
        edge_features: dictionary of edge features
        flows: dictionary of flows (target values)
        directed: whether forcing the torch geometric graph
                  to be undirected. Default: False

    Returns:
        torch geometric graph
        edge_map

    """
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce

    N = G.number_of_nodes()
    mapping = dict(zip(G.nodes(), range(0, N)))

    node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    node_attr_dim = 0
    for f in node_attrs:
        node_attr_dim += np.asarray(list(nx.get_node_attributes(G, f).values())[0]).size

    x = None
    if len(node_attrs) > 0:
        x = torch.empty((G.number_of_nodes(), node_attr_dim))
        idx = 0
        for key in node_attrs:
            attr = G.nodes.data(key)
            attrl = np.asarray(list(nx.get_node_attributes(G, key).values())[0]).size
            for n, v in attr:
                x[mapping.get(n, n), idx:(idx + attrl)] = torch.as_tensor(v, dtype=torch.float)
            idx += attrl

    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.int64)
    n_edge_features = edge_features[list(edge_features.keys())[0]].shape[0]
    edge_attr = torch.empty((G.number_of_edges(), n_edge_features))
    # t_flows = torch.empty((G.number_of_edges(),))

    # Dictionary of pyg <-> nx edges
    edge_map = {}

    i = 0
    for (n1, n2) in edge_features:
        edge_index[:, i] = torch.as_tensor([mapping.get(n1, n1), mapping.get(n2, n2)], dtype=torch.int64)
        edge_attr[i, :] = torch.from_numpy(edge_features[n1, n2])
        # t_flows[i] = torch.as_tensor([flows[n1, n2]], dtype=torch.float)
        edge = (n1, n2)
        edge_map[edge] = i
        edge_map[i] = edge
        i = i + 1

    edge_index_orig = edge_index

    if directed == False:
        edge_index_ext = torch.cat([edge_index, torch.tensor(torch.cat([edge_index[1, :].reshape(1,-1), edge_index[0, :].reshape(1,-1)], dim=0))], dim=1)
        edge_attr_ext = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index=edge_index_ext, edge_attr=edge_attr_ext, reduce='max', num_nodes=N)

    pyggraph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Store the original edge_index
    pyggraph.edge_index_orig = edge_index_orig  # Original indices associated with flows

    # pyggraph.flows = t_flows  # Adding target flows, to work only with torch_geometric.data.Data object
    pyggraph.B = get_incidence_matrix(pyggraph.edge_index_orig, N,
                                           force_undirected=False)  # Store incidence matrix of original graph

    return pyggraph.to(dev), edge_map


def custom_networkx_from_pyg(data, source_node_idx: int = -1):
    """
    Function to convert a pyg
    data object into a networkx one,
    plus dictionaries of flows and edge
    features, to make it usable from
    the bilevel flow estimators.

    Args:
        data: torch_geometric data. It assumes
            - flows are stored in the f_true field
            - edge features are stored in the edge_attr field
            - if available, edge weights are stored in the edge_weight field
        source_node_idx: index of "master" node to be added

    Returns:
        G: networkx graph
        flows: edge flows
        features: edge features
    """
    # Step 1: create nx.graph
    G = nx.DiGraph()
    for m in range(data.edge_index.shape[-1]):

        if hasattr(data, 'edge_weight'):
            G.add_edge(data.edge_index[0, m].item(),
                       data.edge_index[1, m].item(),
                       f_true=data.f_true[m].item(),
                       edge_attr=data.edge_attr[m, :].numpy(),
                       edge_weight=data.edge_weight[m].item())
        else:
            G.add_edge(data.edge_index[0, m].item(),
                       data.edge_index[1, m].item(),
                       f_true=data.f_true[m].item(),
                       edge_attr=data.edge_attr[m, :].numpy())

    # Step 2: add source node and make graph divergence-free
    virtual_edges = []
    original_nodes = list(G.nodes)
    for i in original_nodes:
        G.add_edge(source_node_idx,
                   i,
                   f_true=data.x[i],
                   edge_attr=np.zeros(data.edge_attr.shape[-1],))
        virtual_edges.append((source_node_idx, i))

    # Step 3: Rescale flows to [0,1]
    scale_factor = max(map(abs, nx.get_edge_attributes(G, 'f_true').values()))
    original_edges = list(G.edges)
    for i, j in original_edges:
        norm_flow = G.edges[(i, j)]['f_true'] / scale_factor
        if G.edges[(i, j)]['f_true'] < 0:
            G.add_edge(j, i, f_true=-norm_flow, edge_attr=G.edges[(i, j)]['edge_attr'])
            G.remove_edge(i, j)
        else:
            G.edges[(i, j)]['f_true'] = norm_flow

    flows = {edge: G.edges[edge]['f_true'] for edge in G.edges}
    features = nx.get_edge_attributes(G, 'edge_attr')

    return G, flows, features
