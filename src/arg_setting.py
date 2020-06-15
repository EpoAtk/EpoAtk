import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--tar_epochs', type=int, default=200,
                        help='Number of epochs to train target model.')
    parser.add_argument('--tar_lr', type=float, default=0.01,
                        help='learning rate for target model.')
    parser.add_argument('--tar_weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters) for target model.')
    parser.add_argument('--tar_hidden', type=int, default=64,
                        help='Number of hidden units for target model.')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation. In other word，the number of adjcency matrix to multiply.')
    parser.add_argument('--fake_ratio', type=float, default=0.05, help='')
    parser.add_argument('--drop_rate', type=float, default=0, help='')
    parser.add_argument('--greedy_edges', type=int, default=10, help='')
    parser.add_argument('--cheb_lr', type=float, default=5e-3, help='')
    parser.add_argument('--meta_method', type=str, default='', help='')
    parser.add_argument('--modified_graph_filename', type=str, default='', help='')
    parser.add_argument('--sample_num', type=int, default='20')
    parser.add_argument('--re_rate', type=float, default='0.8')
    parser.add_argument('--init_alive_numbers', type=int, default=1, help='the number of initial candidate set')
    #parser.add_argument('--alive_numbers', tpye=int, default=10, help='the number of small candidate set')
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
