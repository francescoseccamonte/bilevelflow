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

# Main script to run experiments. Adapted from Colab notebook

import torch
import pickle5 as pickle
import os
import copy
import argparse

from bilevelflow.utils import *
from bilevelflow.nn import *


def get_est_res(est, est_params, features, lrlist, G, flows, train_test_folds, dev):

    best_res = None
    for lr in lrlist:
        est_params['lr'] = lr

        estimator = est(G, features, est_params, dev)

        res_avg, res_std, pred_flows = k_fold_cross_validation(train_test_folds, G, flows, estimator)

        if best_res is None or res_avg['RMSE'] < best_res[0]['RMSE']:
            best_res = copy.deepcopy([res_avg, res_std, pred_flows])

    print(f'Results for {est.__name__}: {best_res[0]}')
    return best_res


def mainscript(args):

    if args.exp not in ['traffic', 'power']:
        raise RuntimeError("Error: experiment must be one of power or traffic")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {dev}.")

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    CUDA_LAUNCH_BLOCKING=1

    # Reading power data
    data_folder = "../data/"
    G = read_net(data_folder + args.exp + "_net.csv")

    pfile = open(data_folder + "features_" + args.exp + ".pkl", 'rb')
    features = pickle.load(pfile)
    pfile.close()

    pfile = open(data_folder + "flows_" + args.exp + ".pkl", 'rb')
    flows = pickle.load(pfile)
    pfile.close()

    #Preprocessing
    G, flows, features = make_non_neg_norm(G, flows, features)
    features = normalize_features(features)

    #Creating new folds
    train_test_folds = generate_folds(flows, 10, .1)

    # Common parameters
    common_pars = {}
    common_pars['early_stop'] = 10
    common_pars['nonneg'] = (args.exp == 'traffic')
    common_pars['priors'] = zero_prior(G)

    # NN common parameters
    nn_pars = common_pars
    nn_pars['n_hidden'] = 8
    nn_pars['n_folds'] = 10
    nn_pars['lambda'] = 0.1

    # Bilevel common params
    learning_rates = [1e-2]  # Tuned
    bil_params = nn_pars
    bil_params['outer_n_iter'] = 10
    bil_params['outer_lr'] = 1e-2

    # Bil-GCN-IMP
    bil_gcn_imp_params = bil_params

    bil_gcn_imp_res = get_est_res(BilGCNIMP, bil_gcn_imp_params, features, learning_rates, G, flows, train_test_folds, dev)

    # Setting node features
    nx.set_node_attributes(G, dict(G.degree()), 'degree')

    # Bil-GAT-IMP
    bil_gat_imp_params = bil_params
    bil_gat_imp_params['n_hidden'] = 4
    bil_gat_imp_params['n_node_emb'] = 4
    bil_gat_imp_params['flows'] = flows
    bil_gat_imp_params['constrained'] = False
    bil_gat_imp_params['constrained_final'] = False
    bil_gat_imp_params['constrained_perc'] = None
    bil_gat_imp_params['constrained_lb'] = None

    bil_gat_imp_res = get_est_res(BilGATIMP, bil_gat_imp_params, features, learning_rates, G, flows, train_test_folds, dev)

    # Bil-GAT-IMP-C
    bil_gat_imp_c_params = bil_gat_imp_params
    bil_gat_imp_c_params['constrained'] = True
    bil_gat_imp_c_params['constrained_final'] = True
    bil_gat_imp_c_params['constrained_perc'] = 0.25
    bil_gat_imp_c_params['constrained_lb'] = 0.95

    bil_gat_imp_c_res = get_est_res(BilGATIMP, bil_gat_imp_c_params, features, learning_rates, G, flows, train_test_folds, dev)


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Main script to run deepwater.')
    parser.add_argument('--exp', metavar='experiment', type=str, help='Experiment (traffic, power)', default=None)

    args = parser.parse_args()

    mainscript(args)
