import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from arg_setting import get_args
from utils import *
from models import Model, ChebModel
from metrics import *
from numpy import random
from torch import autograd
from collections import OrderedDict
import scipy.sparse as sp
import random
from functools import reduce

def propose_add():
    node_start = random.randint(0, _A_obs.shape[0]-1)
    class_start = pre_all_labels[node_start]

    node_candidate = np.where(pre_all_labels != class_start)[0]
    np.random.shuffle(node_candidate)
    find = 0
    for i in range(len(node_candidate)):
        node_end = node_candidate[i]
        
        if (node_start, node_end) in change_edges or (node_end, node_start) in change_edges:
            continue
        elif node_start in onehops_dict[node_end] or node_end in onehops_dict[node_start]:
            continue
        else:
            ori_adj_tensor_cuda[node_start, node_end] = 1
            ori_adj_tensor_cuda[node_end, node_start] = 1

            change_edges.append((node_start, node_end))
            change_edges.append((node_end, node_start))

            find = 1
            print('add edge (%d, %d), class is (%d, %d)' % (node_start, node_end, pre_all_labels[node_start], pre_all_labels[node_end]))
            break
    return find


def propose_del():
    if len(edge_set) == 0:
        return 0
    index_del = random.randint(0, len(edge_set)-1)
    x, y = edge_set[index_del]
    if (x, y) in change_edges or (y, x) in change_edges:
        return 0
    ori_adj_tensor_cuda[x, y] = 0
    ori_adj_tensor_cuda[y, x] = 0

    change_edges.append((x, y))
    change_edges.append((y, x))

    edge_set.remove((x, y))
    print('delete edge (%d, %d), class is (%d, %d)' % (x, y, pre_all_labels[x], pre_all_labels[y]))
    
    return 1

def get_all_labels(train_idx, train_labels, valid_idx, valid_labels, test_idx, test_pre_labels):
    all_idx = reduce(np.union1d, (train_idx, valid_idx, test_idx))
    #print(len(all_idx))
    train_dict = dict(zip(train_idx, train_labels))
    valid_dict = dict(zip(valid_idx, valid_labels))
    test_dict = dict(zip(test_idx, test_pre_labels))
    all_labels = []
    for j in range(len(all_idx)):
        i = all_idx[j]
        if i in train_dict.keys():
            all_labels.append(train_dict[i])
        elif i in valid_dict.keys():
            all_labels.append(valid_dict[i])
        elif i in test_dict.keys():
            all_labels.append(test_dict[i])
        else:
            print('exists an error!')
    all_labels = np.array(all_labels).astype(int)
    return all_labels

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed, args.cuda)
    print(args)

    method_add = propose_add
    method_del = propose_del

    A_obs, _X_obs, _z_obs = load_npz('data/cora_ml.npz')

    if _X_obs is None:
        _X_obs = sp.eye(A_obs.shape[0]).tocsr()

    _A_obs = A_obs + A_obs.T
    _A_obs[_A_obs > 1] = 1
    #lcc = largest_connected_components(_A_obs)
    #_A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    #_X_obs = _X_obs[lcc]
    #_z_obs = _z_obs[lcc]

    # node numbers cora--2485
    _N = _A_obs.shape[0]
    # node classes cora--7
    _K = _z_obs.max() + 1
    # node feature dim cora--1433
    _F = _X_obs.shape[1]
    _Z_obs = np.eye(_K)[_z_obs]
    print("node number: %d; node class: %d; node feature: %d" % (_N, _K, _F))
    # onehops_dict, twohops_dict, threehops_dict = sparse_mx_to_khopsgraph(_A_obs)

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    split_train, split_val, split_test = train_val_test_split_tabular(np.arange(_N),
                                                                      train_size=train_share,
                                                                      val_size=val_share,
                                                                      test_size=unlabeled_share,
                                                                      stratify=_z_obs)

    share_perturbations = args.fake_ratio
    # edges * 2
    mod_adj_number = int(share_perturbations * _A_obs.sum())


    surrogate_model = Model(_F, args.tar_hidden, _K)
    if args.cuda:
        surrogate_model.cuda()


    # define dense adj tensor with autograd
    ori_adj_tensor = torch.tensor(_A_obs.toarray(), dtype=torch.float32, requires_grad=False)
    ori_adj_tensor_cuda = ori_adj_tensor.cuda()
    adj_selfloops = torch.add(ori_adj_tensor_cuda, torch.eye(_N).cuda())
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)

    _X_cuda, _z_cuda, idx_train_cuda, idx_val_cuda, idx_test_cuda = convert_to_Tensor(
        [_X_obs, _Z_obs, split_train, split_val, split_test])
    

    surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=args.tar_lr, weight_decay=args.tar_weight_decay)
    surrogate_model.model_train(surrogate_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda, _z_cuda,
                                idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)
    target_model = Model(_F, args.tar_hidden, _K)
    if args.cuda:
        target_model.cuda()
    target_optimizer = optim.Adam(target_model.parameters(), lr=args.tar_lr, weight_decay=args.tar_weight_decay)
    target_model.model_train(target_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda, _z_cuda, idx_train_cuda,
                             idx_val_cuda, use_relu=True, drop_rate=0)
    target_model.model_test(_X_cuda, adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, use_relu=True)
    print('------------------------------------------------------')

    # results of valid and test data
    surrogate_outputs = surrogate_model(_X_cuda, adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
    pre_labels = surrogate_outputs[idx_test_cuda]
    _, predict_test_labels = torch.max(pre_labels, 1)
    real_train_labels = _z_obs[split_train]
    real_valid_labels = _z_obs[split_val]
    pre_all_labels = get_all_labels(split_train, real_train_labels, split_val, real_valid_labels, split_test, predict_test_labels.cpu().numpy())
    #pre_all_labels_cuda = torch.LongTensor(pre_all_labels).cuda()

    edge_from, edge_to = _A_obs.nonzero()
    edge_set = []
    for i in range(len(edge_from)):
        if (pre_all_labels[edge_from[i]] == pre_all_labels[edge_to[i]]) and (not (edge_to[i], edge_from[i]) in edge_set):
            edge_set.append((edge_from[i], edge_to[i]))
    print('length of edge_set is %d' % (len(edge_set)))

    change_edges = []

    onehops_dict, twohops_dict, threehops_dict, fourhops_dict, degree_distrib = sparse_mx_to_khopsgraph(_A_obs)

    #prob = 1.0
    find = 0
    count = 0
    while (len(change_edges) < mod_adj_number):

        P = random.random()
        if P > 1.0:
            find = method_del()
        else:
            find = method_add()

        if find == 1:
            count = count + 1
        else:
            print('find failure!!!')
            continue


    final_adj_cuda = ori_adj_tensor_cuda.clone()

    #save_adj = final_adj_cuda.cpu().numpy()
    #np.save('DICE_modified_graph/' + args.modified_graph_filename, save_adj)

    adj_selfloops = torch.add(final_adj_cuda, torch.eye(_N).cuda())
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)

    cheb_inv_degrees = torch.pow(torch.sum(final_adj_cuda, dim=0, keepdim=True), -0.5)
    cheb_adj_norm_tensor_cuda = final_adj_cuda * cheb_inv_degrees * cheb_inv_degrees.transpose(0, 1)

    gcn_accu = []
    for i in range(50):
        final_target_model = Model(_F, args.tar_hidden, _K)
        if args.cuda:
            final_target_model.cuda()
        final_target_optimizer = optim.Adam(final_target_model.parameters(), lr=args.tar_lr,
                                            weight_decay=args.tar_weight_decay)
        final_target_model.model_train(final_target_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda, _z_cuda,
                                       idx_train_cuda, idx_val_cuda, use_relu=True, drop_rate=0)
        #print('{:04d} current poisoning attack results:'.format(i + 1))
        final_target_model.model_test(_X_cuda, adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, use_relu=True)
        gcn_accu.append(final_target_model.acc_test.item())
        del final_target_model

    print('-------------------------GCN------------------------------')
    attack_results = np.array(gcn_accu)
    print(attack_results)
    attack_results = np.sort(attack_results)
    print(attack_results)
    attack_results = np.delete(attack_results, np.append(np.arange(5), np.arange(attack_results.size-5, attack_results.size)).tolist())
    print(attack_results)

    print('GCN' + ' (%f of edges modified) results:' % (args.fake_ratio))
    print('mean value: %f' % (attack_results.mean()))
    print('std value: %f' % (attack_results.std()))

    



