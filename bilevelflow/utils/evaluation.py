# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material

import networkx as nx
import numpy as np
import time


def generate_folds(flows, n_folds, perc_valid=0., keep_train=1.):
    '''
        Generating train-test or train-test-validation folds 
        as a list of tuples.
    '''
    folds = []
    labelled_edges = list(flows.keys())
    np.random.shuffle(labelled_edges)
    size_fold = int(len(flows) / n_folds)
        
    for f in range(n_folds):
        folds.append(labelled_edges[f*size_fold:(f+1)*size_fold])
    
    train_test = []
    for f in range(n_folds):
        train_f = {}
        test_f = {}
        
        for fi in range(n_folds):
            if f != fi:
                for e in folds[fi]:
                    train_f[e] = flows[e]
        
        for e in flows:
            if e not in train_f:
                test_f[e] = flows[e]
        
        if keep_train < 1.:
            keep = int(keep_train * len(train_f))
            edges = list(train_f.keys())
            np.random.shuffle(edges)
            edges = edges[0:keep]
            
            new_train_f = {}
            
            for e in train_f:
                if e in edges:
                    new_train_f[e] = train_f[e]
            
            train_f = new_train_f
               
        if perc_valid == 0.:
            train_test.append((train_f, test_f))
        else:
            act_train_f, valid_f = get_train_validation(train_f, perc_valid)
            train_test.append([act_train_f, valid_f, test_f])
            
    return train_test


def get_train_validation(train_flows, perc_valid):
    '''
        Breaks training data into training and validation
    '''
    n_valid_flows = int(len(train_flows) * perc_valid)
    new_train_flows = {}
    new_valid_flows = {}
    
    edges = list(train_flows.keys())
    np.random.shuffle(edges)
    
    for i in range(n_valid_flows):
        e = edges[i]
        new_valid_flows[e] = train_flows[e]
        
    for e in train_flows:
        if e not in new_valid_flows:
            new_train_flows[e] = train_flows[e]
            
    return new_train_flows, new_valid_flows


import math
import scipy.stats


def divergence(G, flows):
    '''
        Returns squared divergence (difference between in and out flows)
        and per node divergences as a dictionary.
    '''
    B = nx.incidence_matrix(G, oriented=True)

    f = np.zeros(G.number_of_edges())
    div = {}

    i = 0
    for e in G.edges():
        if e in flows:
            f[i] = flows[e]
        else:
            f[i] = -flows[(e[1],e[0])]

        i = i + 1

    r = B.dot(f)

    i = 0
    for v in G.nodes():
        div[v] = r[i]
        i = i + 1

    return np.sqrt(np.inner(r, r)), div


def rmse(G, flows, est_flows, labelled_edges):
    '''
        Returns the average squared loss.
    '''
    error = 0.
    for e in flows:
        if e not in labelled_edges:
            error = error + math.pow(est_flows[e]-flows[e], 2)

    return math.sqrt(error / (len(flows)-len(labelled_edges)))

def mae(G, flows, est_flows, labelled_edges):
    '''
        Returns the average absolute error.
    '''
    error = 0.
    for e in flows:
        if e not in labelled_edges:
            error = error + abs(est_flows[e]-flows[e])

    return error / (len(flows)-len(labelled_edges))


def mape(G, flows, est_flows, labelled_edges, tol=1e-6):
    '''
        Returns the average absolute percentage error.
    '''
    error = 0.
    for e in flows:
        if e not in labelled_edges:
            error = error + abs(est_flows[e]-flows[e]) / (flows[e] + tol)

    return error / (len(flows)-len(labelled_edges))


def correlation(flows, est_flows, labelled_edges):
    '''
        Returns the pearson correlation between the true
        and estimated flows and also the flow values for
        (scatter) plotting.
    '''
    x = []
    y = []

    for e in flows:
        if e not in labelled_edges:
            x.append(flows[e])
            y.append(est_flows[e])

    return scipy.stats.pearsonr(x, y)[0], x, y


def evaluate_flow_est(G, flows, est_flows, labelled_edges):
    metrics = {}

    metrics['RMSE'] = rmse(G, flows, est_flows, labelled_edges)
    metrics['MAE'] = mae(G, flows, est_flows, labelled_edges)
    metrics['MAPE'] = mape(G, flows, est_flows, labelled_edges)
    metrics['PEARSON'], x, y = correlation(flows, est_flows, labelled_edges)
    metrics['DIV'], node_div = divergence(G, est_flows)

    return metrics


def sample_labelled_flows(flows, rate, seed=123):
    '''
        Generates a random sample of labelled flows.
    '''
    n_labelled = rate * len(flows)
    ground_truth_edges = list(flows.keys())
    random.Random(seed).shuffle(ground_truth_edges)

    labelled_edges = ground_truth_edges[0:int(rate * n_labelled)]

    labelled_flows = {}

    for e in labelled_edges:
        labelled_flows[e] = flows[e]

    return labelled_flows


def k_fold_cross_validation(folds, G, flows, method):
    '''
        Evaluates flow estimation method using
        ten-fold cross validation
    '''
    divs = []
    maes = []
    mapes = []
    corrs = []
    rmses = []
    train_times = []
    test_times = []

    pred_flows = {}

    for  f in range(len(folds)):
        train_flows = folds[f][0]
        valid_flows = folds[f][1]
        test_flows = folds[f][2]

        start_time = time.time()
        method.trainloop(train_flows, valid_flows)
        train_times.append(time.time() - start_time)

        #pred = method.predict(test_flows)

        train_val_flows = {**train_flows, **valid_flows}
        non_training_edges = {}
        for e in G.edges():
            if e not in train_val_flows:
                non_training_edges[e] = 1

        start_time = time.time()
        pred = method.predict(non_training_edges)
        test_times.append(time.time() - start_time)


        res = evaluate_flow_est(G, flows, {**pred, **train_val_flows}, train_val_flows)
        divs.append(res['DIV'])
        maes.append(res['MAE'])
        mapes.append(res['MAPE'])
        corrs.append(res['PEARSON'])
        rmses.append(res['RMSE'])

        print("fold ", f, " ", res)

        for e in pred:
            if e not in train_val_flows:
                pred_flows[e] = pred[e]

    avg_res_test = {'DIV': np.mean(divs), 'MAE':np.mean(maes), 'MAPE':np.mean(mapes), 'PEARSON':np.mean(corrs), 'RMSE':np.mean(rmses), 'TRAIN-TIME':np.mean(train_times), 'TEST-TIME': np.mean(test_times)}
    std_res_test = {'DIV': np.std(divs), 'MAE':np.std(maes), 'MAPE':np.std(mapes), 'PEARSON':np.std(corrs), 'RMSE':np.std(rmses), 'TRAIN-TIME':np.std(train_times), 'TEST-TIME': np.std(test_times)}

    avg_train_time = np.mean(train_times)
    avg_test_time = np.mean(test_times)

    return avg_res_test, std_res_test, pred_flows
