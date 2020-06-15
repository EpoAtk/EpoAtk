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

def method_add(x, y, new_adj_tensor_cuda, new_surrogate_model):
    ori_adj_cuda = new_adj_tensor_cuda.clone()
    find = 0
    loss_test = None
    ori_adj_cuda[x, y] = 1
    ori_adj_cuda[y, x] = 1
    adj_selfloops = torch.add(ori_adj_cuda, torch.eye(_N).cuda())
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    print('add edge (%d, %d)' % (x, y))
    find = 1
    new_surrogate_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda, use_relu=False)
    loss_test = -new_surrogate_model.loss_test
    return find, loss_test


def method_del(x, y, new_adj_tensor_cuda, new_surrogate_model):
    ori_adj_cuda = new_adj_tensor_cuda.clone()
    ori_adj_cuda[x, y] = 0
    ori_adj_cuda[y, x] = 0
    print('delete edge (%d, %d)' % (x, y))
    find = 1
    loss_test = None
    adj_selfloops = torch.add(ori_adj_cuda, torch.eye(_N).cuda())
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    new_surrogate_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda, use_relu=False)
    loss_test = -new_surrogate_model.loss_test
    return find, loss_test


def get_greedy_list(ori_adj_cuda, Greedy_edges, change_edges):
    new_adj_tensor_cuda = ori_adj_cuda.clone()
    adj_selfloops = torch.add(new_adj_tensor_cuda, torch.eye(_N).cuda())
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    new_adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    #new_adj_norm_tensor_cuda.requires_grad = True

    new_surrogate_model = Model(_F, args.tar_hidden, _K)
    if args.cuda:
        new_surrogate_model.cuda()
    new_surrogate_optimizer = optim.Adam(new_surrogate_model.parameters(), lr=args.tar_lr,
                                         weight_decay=args.tar_weight_decay)
    new_surrogate_model.model_train(new_surrogate_optimizer, args.tar_epochs, _X_cuda, new_adj_norm_tensor_cuda,
                                    _z_cuda, idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)

    new_surrogate_model.zero_grad()
    new_adj_norm_tensor_cuda.requires_grad = True

    outputs = new_surrogate_model(_X_cuda, new_adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
    loss = F.nll_loss(outputs[idx_train_cuda], _z_cuda[idx_train_cuda])

    loss = -loss
    loss.backward()

    grad = -(new_adj_norm_tensor_cuda.grad.data.cpu().numpy().flatten())
    grad_abs = -(np.abs(grad))

    idxes = np.argsort(grad_abs)
    find = 0
    acc = None

    for p in idxes:
        if (len(Greedy_edges) < args.greedy_edges):
            x = p // _N
            y = p % _N
            if (x, y) in change_edges or (y, x) in change_edges:
                continue

            # add edge
            if grad[p] > 0:
                signal = 1
                if x == y or x in onehops_dict[y] or y in onehops_dict[x]:
                    continue
                else:
                    find, acc = method_add(x, y, new_adj_tensor_cuda, new_surrogate_model)
                    # ori_adj_cuda = new_adj_tensor_cuda.clone()
            # delete edge
            else:
                signal = 0
                if x == y or not x in onehops_dict[y] or not y in onehops_dict[x]:
                    continue
                else:
                    find, acc = method_del(x, y, new_adj_tensor_cuda, new_surrogate_model)
            if find == 1:
                edge_oper = (x, y, signal)
                acc = acc.item()
                Greedy_edges[edge_oper] = acc
                print('Greedy edge number', len(Greedy_edges))
        else:
            break
    Greedy_list = sorted(Greedy_edges.items(), key=lambda x: x[1])

    return Greedy_list

def crossover(fir_edge, sec_edge, adj, changes):
    co_list = []
    fitness_list = []
    co_list.append(fir_edge)
    co_list.append(sec_edge)
    fir_x, fir_y, fir_signal = fir_edge
    sec_x, sec_y, sec_signal = sec_edge
    signal = adj[fir_x, sec_y]
    if signal > 0:
        third_signal = 0
    else:
        third_signal = 1
    third_edge = (fir_x, sec_y, third_signal)
    signal = adj[sec_x, fir_y]
    if signal > 0:
        four_signal = 0
    else:
        four_signal = 1
    four_edge = (sec_x, fir_y, four_signal)
    co_list.append(third_edge)
    co_list.append(four_edge)

    for i in range(len(co_list)):
        x, y, signal = co_list[i]
        new_adj = adj.clone()
        if (x, y) in changes or (y, x) in changes:
            fitness_list.append(sys.maxsize)
            continue
        else:
            if signal == 1:
                new_adj[x, y] = 1.0
                new_adj[y, x] = 1.0
            if signal == 0:
                new_adj[x, y] = 0.0
                new_adj[x, y] = 0.0

            adj_selfloops = torch.add(new_adj, torch.eye(_N).cuda())
            inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
            adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)

            new_model = Model(_F, args.tar_hidden, _K)
            if args.cuda:
                new_model.cuda()
            new_optimizer = optim.Adam(new_model.parameters(), lr=args.tar_lr,
                                              weight_decay=args.tar_weight_decay)
            new_model.model_train(new_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda, _z_cuda,
                                         idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)
            new_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda,
                                        use_relu=False)
            loss_test = -new_model.loss_test
            fitness_list.append(loss_test)

    fitness_idx = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
    index = fitness_idx[0]
    return co_list[index]

if __name__ == '__main__':
    args = get_args()
    print(args)
    # setting random seeds
    set_seed(args.seed, args.cuda)
    A_obs, _X_obs, _z_obs = load_npz('data/cora_ml.npz')

    if _X_obs is None:
        _X_obs = sp.eye(A_obs.shape[0]).tocsr()

    _A_obs = A_obs + A_obs.T
    _A_obs[_A_obs > 1] = 1
    # lcc = largest_connected_components(_A_obs)
    # _A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    # _X_obs = _X_obs[lcc]
    # _z_obs = _z_obs[lcc]

    # node numbers cora--2485
    _N = _A_obs.shape[0]
    # node classes cora--7
    _K = _z_obs.max() + 1
    # node feature dim cora--1433
    _F = _X_obs.shape[1]
    _Z_obs = np.eye(_K)[_z_obs]
    print("node number: %d; node class: %d; node feature: %d" % (_N, _K, _F))
    # onehops_dict, twohops_dict, threehops_dict = sparse_mx_to_khopsgraph(_A_obs)
    # normalized adj sparse matrix

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    split_train, split_val, split_test = train_val_test_split_tabular(np.arange(_N),
                                                                      train_size=train_share,
                                                                      val_size=val_share,
                                                                      test_size=unlabeled_share,
                                                                      stratify=_z_obs)

    split_unlabeled = np.union1d(split_val, split_test)
    share_perturbations = args.fake_ratio

    mod_adj_number = int(share_perturbations * _A_obs.sum() // 2)

    ori_adj_tensor = torch.tensor(_A_obs.toarray(), dtype=torch.float32, requires_grad=False)
    ori_adj_tensor_cuda = ori_adj_tensor.cuda()
    inv_degrees = torch.pow(torch.sum(ori_adj_tensor_cuda, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = ori_adj_tensor_cuda * inv_degrees * inv_degrees.transpose(0, 1)

    adj_selfloops = torch.add(ori_adj_tensor_cuda, torch.eye(_N).cuda())
    target_inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    target_adj_norm_tensor_cuda = adj_selfloops * target_inv_degrees * target_inv_degrees.transpose(0, 1)

    _X_cuda, _z_cuda, idx_train_cuda, idx_val_cuda, idx_test_cuda = convert_to_Tensor(
        [_X_obs, _Z_obs, split_train, split_val, split_test])

    all_idx_cuda = torch.cat((idx_train_cuda, idx_val_cuda, idx_test_cuda))
    extra_idx_cuda = torch.cat((idx_val_cuda, idx_test_cuda))

    surrogate_model = Model(_F, args.tar_hidden, _K)
    if args.cuda:
        surrogate_model.cuda()

    surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=args.tar_lr, weight_decay=args.tar_weight_decay)
    surrogate_model.model_train(surrogate_optimizer, args.tar_epochs, _X_cuda, target_adj_norm_tensor_cuda, _z_cuda,
                                idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)
    surrogate_model.model_test(_X_cuda, target_adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, use_relu=False)

    target_model = Model(_F, args.tar_hidden, _K)
    if args.cuda:
        target_model.cuda()
    target_optimizer = optim.Adam(target_model.parameters(), lr=args.tar_lr, weight_decay=args.tar_weight_decay)
    target_model.model_train(target_optimizer, args.tar_epochs, _X_cuda, target_adj_norm_tensor_cuda, _z_cuda,
                             idx_train_cuda,
                             idx_val_cuda, use_relu=True, drop_rate=0)
    target_model.model_test(_X_cuda, target_adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, use_relu=True)
    print('------------------------------------------------------')

    change_edges_list = [[] for i in range(args.init_alive_numbers)]
    changing_adj_list = [ori_adj_tensor_cuda for i in range(args.init_alive_numbers)]
    Greedy_edges_list = [{} for i in range(args.init_alive_numbers)]

    onehops_dict, twohops_dict, threehops_dict, fourhops_dict, degree_distrib = sparse_mx_to_khopsgraph(_A_obs)

    # results of valid and test data
    surrogate_outputs = surrogate_model(_X_cuda, target_adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
    pre_labels = surrogate_outputs[idx_test_cuda]
    _, predict_test_labels = torch.max(pre_labels, 1)
    predict_test_labels = predict_test_labels.cpu().numpy()
    real_train_labels = _z_obs[split_train]
    real_valid_labels = _z_obs[split_val]
    pre_all_labels = get_all_labels(split_train, real_train_labels, split_val, real_valid_labels, split_test,
                                    predict_test_labels)
    pre_all_labels_cuda = torch.LongTensor(pre_all_labels).cuda()


    begin_mutation_rate = 1.0
    end_mutation_rate = args.re_rate
    mutation_step = (begin_mutation_rate - end_mutation_rate) / mod_adj_number
    mutation_rate = 1.0

    for i in range(mod_adj_number):
        mutation_rate -= mutation_step
        print('current recombination rate %.3f' % mutation_rate)
        for ii in range(len(changing_adj_list)):
            Greedy_list = get_greedy_list(changing_adj_list[ii], Greedy_edges_list[ii], change_edges_list[ii])
            bi_prob = np.random.binomial(1, mutation_rate, 1)[0]
            if bi_prob:
                selected_edge, cur_acc = Greedy_list[0]
                x, y, signal = selected_edge
                change_edges_list[ii].append((x, y))
                change_edges_list[ii].append((y, x))
                if signal > 0:
                    changing_adj_list[ii][x, y] = 1
                    changing_adj_list[ii][y, x] = 1
                else:
                    changing_adj_list[ii][x, y] = 0
                    changing_adj_list[ii][y, x] = 0
                print('selected edge: ', x, y, signal, i, ii)
                Greedy_edges_list[ii].clear()

            else:
                print('recombination!------')
                fir_edge, _ = Greedy_list[0]
                mu_Greedy_list = Greedy_list[1:]
                inverse_ranks = [1 / i for i in range(1, len(mu_Greedy_list)+1)]
                dis_prob = [i[1] for i in mu_Greedy_list]
                #dis_counts = sum(dis_values)
                #dis_prob = [i / dis_counts for i in dis_values]
                #print(dis_prob)
                new_dis_prob = [dis_prob[i] * inverse_ranks[i] for i in range(len(dis_prob))]
                new_dis_prob = torch.FloatTensor(new_dis_prob).cuda()
                new_dis_prob = torch.unsqueeze(new_dis_prob, 0)
                index = F.gumbel_softmax(new_dis_prob, tau=0.5, hard=True).nonzero()[-1][-1].item()
                #index_1, index_2 = np.random.choice(len(Greedy_list), 2, replace=False, p=dis_prob)
                #fir_edge, _ = Greedy_list[index_1]
                sec_edge, _ = mu_Greedy_list[index]
                selected_edge = crossover(fir_edge, sec_edge, changing_adj_list[ii], change_edges_list[ii])
                x, y, signal = selected_edge
                change_edges_list[ii].append((x, y))
                change_edges_list[ii].append((y, x))
                if signal > 0:
                    changing_adj_list[ii][x, y] = 1
                    changing_adj_list[ii][y, x] = 1
                else:
                    changing_adj_list[ii][x, y] = 0
                    changing_adj_list[ii][y, x] = 0
                print('selected edge: ', x, y, signal, i, ii)
                Greedy_edges_list[ii].clear()

    for i in range(len(changing_adj_list)):
        accuracies_atk = []
        final_adj_cuda = changing_adj_list[i].clone()
        save_adj = final_adj_cuda.cpu().numpy()
        np.save('modified_graph/' + args.modified_graph_filename + str(i + 1), save_adj)

        adj_selfloops = torch.add(final_adj_cuda, torch.eye(_N).cuda())
        inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
        lp_adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
        inv_degrees = torch.pow(torch.sum(final_adj_cuda, dim=0, keepdim=True), -0.5)
        nolp_adj_norm_tensor_cuda = final_adj_cuda * inv_degrees * inv_degrees.transpose(0, 1)
        print('The %dth graph results:-----------------------------------------------------------' % (i + 1))

        for j in range(50):
            final_target_model = Model(_F, args.tar_hidden, _K)
            if args.cuda:
                final_target_model.cuda()
            final_target_optimizer = optim.Adam(final_target_model.parameters(), lr=args.tar_lr,
                                                weight_decay=args.tar_weight_decay)
            final_target_model.model_train(final_target_optimizer, args.tar_epochs, _X_cuda, lp_adj_norm_tensor_cuda,
                                           _z_cuda, idx_train_cuda, idx_val_cuda, use_relu=True, drop_rate=0)
            print('{:04d} current poisoning attack results:'.format(j + 1))
            final_target_model.model_test(_X_cuda, lp_adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, use_relu=True)
            accuracies_atk.append(final_target_model.acc_test.cpu().numpy().item())
            del final_target_model

        attack_results = np.array(accuracies_atk)
        attack_results = np.sort(attack_results)
        attack_results = np.delete(attack_results, np.append(np.arange(5), np.arange(attack_results.size - 5,
                                                                                     attack_results.size)).tolist())
        print(attack_results)
        print('mean value: %f' % (attack_results.mean()))
        print('std value: %f' % (attack_results.std()))
