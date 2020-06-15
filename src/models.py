import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import time
from utils import *
from metrics import *


class NewConvolution(nn.Module):
    """
    A Graph Convolution Layer for GraphSage
    """
    def __init__(self, in_features, out_features, bias=True):
        super(NewConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_1 = nn.Linear(in_features, out_features, bias=bias)
        self.W_2 = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W_1.weight.size(1))
        self.W_1.weight.data.uniform_(-stdv, stdv)
        self.W_2.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support_1 = self.W_1(input)
        support_2 = self.W_2(input)
        output = torch.mm(adj, support_2)
        output = output + support_1
        return output

class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.gc1 = NewConvolution(nfeat, nhid)
        self.gc2 = NewConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def model_train(self, optimizer, epochs, features, adj, labels, idx_train, idx_val, use_relu):
        for i in range(epochs):
            t = time.time()
            self.train()
            optimizer.zero_grad()
            # print(features.shape)
            # print(adj.shape)
            outputs = self.forward(features, adj, use_relu)
            loss_train = F.nll_loss(outputs[idx_train], labels[idx_train])
            acc_train = accuracy(outputs[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            #loss_val = F.nll_loss(outputs[idx_val], labels[idx_val])
            #acc_val = accuracy(outputs[idx_val], labels[idx_val])
            if (i + 1) % 50 == 0:
                print('Epoch: {:04d}'.format(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    #'loss_val: {:.4f}'.format(loss_val.item()),
                    #'acc_val: {:.4f}'.format(acc_val.item()),
                    'time: {:.4f}s'.format(time.time() - t))

    def model_test(self, features, adj, labels, idx_test, use_relu, mode=None):
        if mode is None:
            self.eval()
        else:
            self.train()
        output = self.forward(features, adj, use_relu)
        self.loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        self.acc_test = accuracy(output[idx_test], labels[idx_test])
        if use_relu:
            print("Test set results of target GS model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))
        else:
            print("Test set results of surrogate model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))



class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]  #number of nodes

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GatModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GatModel, self).__init__()
        self.dropout = dropout

        self.attentions = [GATConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATConv(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, use_relu):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    def model_train(self, optimizer, epochs, features, adj, labels, idx_train, idx_val, use_relu):
        for i in range(epochs):
            t = time.time()
            self.train()
            optimizer.zero_grad()
            # print(features.shape)
            # print(adj.shape)
            outputs = self.forward(features, adj, use_relu)
            loss_train = F.nll_loss(outputs[idx_train], labels[idx_train])
            acc_train = accuracy(outputs[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            #loss_val = F.nll_loss(outputs[idx_val], labels[idx_val])
            #acc_val = accuracy(outputs[idx_val], labels[idx_val])
            if (i + 1) % 50 == 0:
                print('Epoch: {:04d}'.format(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    #'loss_val: {:.4f}'.format(loss_val.item()),
                    #'acc_val: {:.4f}'.format(acc_val.item()),
                    'time: {:.4f}s'.format(time.time() - t))

    def model_test(self, features, adj, labels, idx_test, use_relu, mode=None):
        if mode is None:
            self.eval()
        else:
            self.train()
        output = self.forward(features, adj, use_relu)
        self.loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        self.acc_test = accuracy(output[idx_test], labels[idx_test])
        if use_relu:
            print("Test set results of target GAT model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))
        else:
            print("Test set results of surrogate model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))

class ChebConv(nn.Module):

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.K = K

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        self.uniform(size, self.weight)
        self.uniform(size, self.bias)

    def forward(self, input, adj):
        Tx_0 = input
        out = torch.mm(Tx_0, self.weight[0])

        if self.K > 1:
            Tx_1 = torch.mm(adj, input)
            out = out + torch.mm(Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2 * torch.mm(adj, Tx_1) - Tx_0
            out = out + torch.mm(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

class ChebModel(nn.Module):
    def __init__(self, nfeature, nhidden, nclass):
        super(ChebModel, self).__init__()
        self.nfeature = nfeature
        self.nclass = nclass
        self.nhidden = nhidden
        self.conv1 = ChebConv(self.nfeature, self.nhidden, K=2)
        self.conv2 = ChebConv(self.nhidden, self.nclass, K=2)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

    def model_train(self, optimizer, epochs, features, adj, labels, idx_train, idx_val, use_relu):
        for i in range(epochs):
            t = time.time()
            self.train()
            optimizer.zero_grad()
            # print(features.shape)
            # print(adj.shape)
            outputs = self.forward(features, adj)
            loss_train = F.nll_loss(outputs[idx_train], labels[idx_train])
            acc_train = accuracy(outputs[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            #loss_val = F.nll_loss(outputs[idx_val], labels[idx_val])
            #acc_val = accuracy(outputs[idx_val], labels[idx_val])
            if (i + 1) % 50 == 0:
                print('Epoch: {:04d}'.format(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    #'loss_val: {:.4f}'.format(loss_val.item()),
                    #'acc_val: {:.4f}'.format(acc_val.item()),
                    'time: {:.4f}s'.format(time.time() - t))

    def model_test(self, features, adj, labels, idx_test, use_relu, mode=None):
        if mode is None:
            self.eval()
        else:
            self.train()
        output = self.forward(features, adj)
        self.loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        self.acc_test = accuracy(output[idx_test], labels[idx_test])
        if use_relu:
            print("Test set results of  target cheb model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))
        else:
            print("Test set results of surrogate model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))

class GraphConvolution(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class Model(nn.Module):

    def __init__(self, nfeature, nhidden, nclass):
        super(Model, self).__init__()
        self.nfeature = nfeature
        self.nclass = nclass
        self.nhidden = nhidden
        self.w_f2h = nn.Linear(self.nfeature, self.nhidden)
        self.w_h2c = nn.Linear(self.nhidden, self.nclass)
        self.weight_initial()

    def init(self):
        stdv = 1. / math.sqrt(self.w_f2h.weight.size(1))
        self.w_f2h.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.w_h2c.weight.size(1))
        self.w_h2c.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, use_relu, drop_rate):
        if use_relu:
            x = self.w_f2h(x)
            x = torch.mm(adj,x)
            x = F.relu(x)
            x = torch.mm(adj, x)
            output = self.w_h2c(x)
        else:
            x = F.dropout(x, drop_rate, training=self.training)
            x = self.w_f2h(x)
            x = torch.mm(adj, x)
            x = torch.mm(adj, x)
            x = F.dropout(x, drop_rate, training=self.training)
            output = self.w_h2c(x)

        return F.log_softmax(output, dim=1)


    def weight_initial(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def model_train(self, optimizer, epochs, features, adj, labels, idx_train, idx_val, use_relu, drop_rate):
        for i in range(epochs):
            t = time.time()
            self.train()
            optimizer.zero_grad()
            # print(features.shape)
            # print(adj.shape)
            outputs = self.forward(features, adj, use_relu, drop_rate)
            loss_train = F.nll_loss(outputs[idx_train], labels[idx_train])
            acc_train = accuracy(outputs[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            #loss_val = F.nll_loss(outputs[idx_val], labels[idx_val])
            #acc_val = accuracy(outputs[idx_val], labels[idx_val])
            if (i + 1) % 50 == 0:
                print('Epoch: {:04d}'.format(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    'time: {:.4f}s'.format(time.time() - t))

    def model_test(self, features, adj, labels, idx_test, use_relu, mode=None):
        if mode is None:
            self.eval()
        else:
            self.train()
        output = self.forward(features, adj, use_relu, 0)
        self.loss_test = F.nll_loss(output[idx_test], labels[idx_test])

        t_pre = labels[idx_test]
        pre = torch.exp(output[idx_test])
        sel = pre[torch.arange(pre.size(0)), t_pre]

        class_eye = torch.eye(self.nclass.item()).cuda()
        _z_one_hot = class_eye[labels][idx_test]
        pre = pre * (1 - _z_one_hot)

        max_pre = torch.min(pre, 1)[0]
        self.margin_loss = torch.sum(max_pre - sel)

        logits = torch.exp(output)
        self.loss_entropy = -torch.sum(torch.mul(logits, torch.log(logits)))

        self.acc_test = accuracy(output[idx_test], labels[idx_test])
        if use_relu:
            print("Test set results of target model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))
        else:
            print("Test set results of surrogate model:",
                  "loss= {:.4f}".format(self.loss_test.item()),
                  "accuracy= {:.4f}".format(self.acc_test.item()))
