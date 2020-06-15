import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from functools import reduce
from normalization import fetch_normalization, row_normalize
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components


def sparse_mx_to_khopsgraph(sp_mx):

    graph = nx.from_scipy_sparse_matrix(sp_mx)
    twohops_graph = nx.power(graph, 2)
    threehops_graph = nx.power(graph, 3)
    fourhops_graph = nx.power(graph, 4)
    assert (len(graph) == len(twohops_graph))
    print('graph has nodes: ', len(graph))
    print('one hop graph has edges: ', len(graph.edges()))
    print('two hop graph has edges: ', len(twohops_graph.edges()))
    assert (len(graph) == len(threehops_graph))
    print('three hop graph has edges:', len(threehops_graph.edges()))
    print('four hop graph has edges:', len(fourhops_graph.edges()))
    nodes_num = len(graph.nodes())
    onehops_dict= {}
    twohops_dict = {}
    threehops_dict = {}
    fourhops_dict = {}
    degree_dict_prob = graph.degree()
    degree_list_prob = []
    all_degrees = sum(degree_dict_prob.values())
    for (k,v) in degree_dict_prob.items():
        val = v / all_degrees
        degree_dict_prob[k] = val
        degree_list_prob.append(val)

    for i in range(len(graph)):
        onehops_dict[i] = graph.neighbors(i)

    for i in range(len(twohops_graph)):
        twohops_dict[i] = twohops_graph.neighbors(i)

    for i in range(len(threehops_graph)):
        threehops_dict[i] = threehops_graph.neighbors(i)
    #with open("{}/ind.{}.{}.{}".format(data_folder, dataset_str, str(cmd_args.n_hops), 'khops_graph'), 'wb') as f:
        #pkl.dump(khops_dict, f)
    for i in range(len(fourhops_graph)):
        fourhops_dict[i] = fourhops_graph.neighbors(i)

    return onehops_dict, twohops_dict, threehops_dict, fourhops_dict, degree_list_prob

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_citation(adj, features, normalization='AugNormAdj'):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def convert_to_Tensor(input_list):
    # porting to pytorch
    features, labels, idx_train, idx_val, idx_test = input_list
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    #adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    return [features, labels, idx_train, idx_val, idx_test]

def load_citation(dataset_str="cora", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    StaticGraph.graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, 'AugNormAdj')

    tensor_inputs = convert_to_Tensor([features, labels, idx_train, idx_val, idx_test, cuda])
    return tensor_inputs, adj


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels



def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def preprocess_graph(adj):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None,
                                 random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;
    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.
    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if set(idx_train_and_val.tolist()).union(set(idx_test.tolist())) != set(idx.tolist()):
        list_inter = list(set(idx.tolist()).difference(set(idx_train_and_val.tolist()).union(set(idx_test.tolist()))))
        #print(list_inter)
        numpy_inter = np.array(list_inter)
        idx_train_and_val = np.concatenate((idx_train_and_val, numpy_inter)).astype('int')

    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    if set(idx_train.tolist()).union(set(idx_val.tolist())) != set(idx_train_and_val.tolist()):
        list_inter = list(set(idx_train_and_val.tolist()).difference(set(idx_train.tolist()).union(set(idx_val.tolist()))))
        numpy_inter = np.array(list_inter)
        idx_train = np.concatenate((idx_train, numpy_inter))

    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result




def get_all_labels(train_idx, train_labels, valid_idx, valid_labels, test_idx, test_pre_labels):
    all_idx = reduce(np.union1d, (train_idx, valid_idx, test_idx))
    train_dict = dict(zip(train_idx, train_labels))
    valid_dict = dict(zip(valid_idx, valid_labels))
    test_dict = dict(zip(test_idx, test_pre_labels))
    all_labels = []
    for j in range(len(all_idx)):
        i = all_idx[j]
        assert i==j
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