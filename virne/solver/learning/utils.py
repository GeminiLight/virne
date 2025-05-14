import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import sort_edge_index
from sklearn.preprocessing import StandardScaler, Normalizer

from virne.network import *

# A constant tensor used to mask values
NEG_TENSER = torch.tensor(-1e8).float()
ZERO_TENSER = torch.tensor(0.).float()


def get_pyg_data(x, edge_index, edge_attr=None, sort_index=False):
    """
    Convert node and edge information into Pytorch Geometric format.

    Args:
        x (ndarray): Node features with shape (num_nodes, num_node_features).
        edge_index (ndarray): Edge connectivity with shape (2, num_edges).
        edge_attr (ndarray, optional): Edge features with shape (num_edges, num_edge_features). Defaults to None.

    Returns:
        PyTorch Geometric Data object: Data object containing node and edge information in Pytorch Geometric format.
    """
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index).long()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
    if sort_index:
        if edge_attr is not None:
            edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
        else:
            edge_index = sort_edge_index(edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def get_pyg_batch(x_batch, edge_index_batch, edge_attr_batch=None):
    """
    Convert a batch of node and edge information into Pytorch Geometric format.

    Args:
        x_batch (list of ndarrays): List of node feature arrays, where each array has shape (num_nodes_i, num_node_features).
        edge_index_batch (list of ndarrays): List of edge connectivity arrays, where each array has shape (2, num_edges_i).
        edge_attr_batch (list of ndarrays, optional): List of edge feature arrays, where each array has shape (num_edges_i, num_edge_features). Defaults to None.

    Returns:
        PyTorch Geometric Batch object: Batch object containing batched node and edge information in Pytorch Geometric format.
    """
    if edge_attr_batch is None: edge_attr_batch = [None] * len(x_batch)
    data_list = []
    for x, edge_index, edge_attr in zip(x_batch, edge_index_batch, edge_attr_batch):
        data = get_pyg_data(x, edge_index, edge_attr)
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch

def get_available_device():
    """
    Get the available device (CPU or GPU).

    Returns:
        device (torch.device): Available device (CPU or GPU).
    """
    # set device to cpu or cuda
    device = torch.device('cpu')

    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to: " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to: cpu")
    return device

def normailize_data(data, method='standardize'):
    """
    Normalize node or edge data.

    Args:
        data (ndarray): Node or edge feature data with shape (num_data_points, num_features).
        method (str, optional): Type of normalization to apply. 'standardize' or 'normalize'. Defaults to 'standardize'.

    Returns:
        ndarray: Normalized node or edge feature data with shape (num_data_points, num_features).
    """
    if method == 'standardize':
        norm_data = StandardScaler().fit_transform(data).astype('float32')
    else:
        norm_data = Normalizer().fit_transform(data).astype('float32')
    return norm_data

def load_pyg_data_from_network(network, attr_types=['resource'], normailize_method='standardize',
                        normailize_nodes_data=False, normailize_edges_data=False, ):
    """
    Load data from network

    Args:
        network (obj): the input network object
        attr_types (list, optional): list of attribute types to load. Defaults to ['resource'].
        normailize_method (str, optional): method of normalization for the data. Defaults to 'standardize'.
        normailize_nodes_data (bool, optional): flag to normalize the node data. Defaults to False.
        normailize_edges_data (bool, optional): flag to normalize the edge data. Defaults to False.

    Returns:
        data (obj): the PyG data object
    """
    # edge index
    edge_index = np.array(list(network.links),dtype=np.int64).T
    edge_index = torch.LongTensor(edge_index)
    # node data
    n_attrs = network.get_node_attrs(attr_types)
    node_data = np.array(network.get_node_attrs_data(n_attrs), dtype=np.float32).T
    if normailize_nodes_data:
        node_data = normailize_data(node_data, method=normailize_method)
    node_data = torch.tensor(node_data)
    # edge data
    e_attrs = network.get_link_attrs(attr_types)
    link_data = np.array(network.get_link_attrs_data(e_attrs), dtype=np.float32).T
    if normailize_edges_data:
        link_data = normailize_data(link_data, method=normailize_method)
    link_data = torch.tensor(link_data)
    # pyg data
    data = Data(x=node_data, edge_index=edge_index, edge_attr=link_data)
    return data

def load_pyg_batch_from_network_list(network_list):
    """
    Load batch data from a list of networks

    Args:
        network_list (list): list of input network objects

    Returns:
        batch (obj): the PyG batch object
    """
    data_list = []
    for network in network_list:
        data = load_pyg_data_from_network(network)
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch

def apply_mask_to_logit(logit, mask=None):
    """
    Apply a mask to a given logits tensor.

    Args:
        logit (tensor): input logits tensor
        mask (tensor, optional): input mask tensor. Defaults to None.

    Returns:
        masked_logit (tensor): the masked logits tensor
    """
    if mask is None:
        return logit
    # mask = torch.IntTensor(mask).to(logit.device).expand_as(logit)
    # masked_logit = logit + mask.log()
    if not isinstance(mask, torch.Tensor):
        mask = torch.BoolTensor(mask)
    
    # flag = ~torch.any(mask, dim=1, keepdim=True).repeat(1, mask.shape[-1])
    # mask = torch.where(flag, True, mask)
    
    mask = mask.bool().to(logit.device).reshape(logit.size())
    mask_value_tensor = NEG_TENSER.type_as(logit).to(logit.device)
    masked_logit = torch.where(mask, logit, mask_value_tensor)
    return masked_logit

def apply_mask_to_prob(prob, mask=None):
    """
    Apply a mask to a given logits tensor.

    Args:
        logit (tensor): input logits tensor
        mask (tensor, optional): input mask tensor. Defaults to None.

    Returns:
        masked_logit (tensor): the masked logits tensor
    """
    if mask is None:
        return prob
    # mask = torch.IntTensor(mask).to(logit.device).expand_as(logit)
    # masked_logit = logit + mask.log()
    if not isinstance(mask, torch.Tensor):
        mask = torch.BoolTensor(mask)
    
    # flag = ~torch.any(mask, dim=1, keepdim=True).repeat(1, mask.shape[-1])
    # mask = torch.where(flag, True, mask)
    
    mask = mask.bool().to(prob.device).reshape(prob.size())
    mask_value_tensor = ZERO_TENSER.type_as(prob).to(prob.device)
    masked_logit = torch.where(mask, prob, mask_value_tensor)
    return masked_logit

def get_observations_sample(obs_batch, indices, device='cpu'):
    """
    Get a sample from an input observation batch given the indices.

    Args:
        obs_batch (obj): the input observation batch object
        indices (tensor): the indices to use for sampling

    Returns:
        sample (obj): the sample object
    """
    if isinstance(obs_batch, Batch):
        return obs_batch.index_select(indices).to(device)
    elif isinstance(obs_batch, dict):
        sample = {}
        for key, value in obs_batch.items():
            if isinstance(value, Batch):
                value_sample = Batch.from_data_list(value.index_select(indices)).to(device)
            else:
                value_sample = value[indices].to(device)
            sample[key] = value_sample
        return sample
    else:
        return obs_batch[indices].to(device)


class RunningMeanStd(object):
    """
    Calculate running mean and standard deviation for a given data.

    The update function is called every time the data is added.

    References: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Constructor of RunningMeanStd.

        Args:
            epsilon: A small number to avoid divide-by-zero errors.
            shape: A tuple containing the shape of the data.
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        """
        Update the running mean and standard deviation with the new data.

        Args:
            x: An array-like object containing the new data to be added.
        """
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Update the running mean and standard deviation with the new moments.

        Args:
            batch_mean: An array-like object containing the batch mean.
            batch_var: An array-like object containing the batch variance.
            batch_count: An integer specifying the number of data points in the batch.
        """
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """
    Update mean, variance, and count from the batch mean, variance, and count.

    Args:
        mean: An array-like object containing the current mean.
        var: An array-like object containing the current variance.
        count: An integer specifying the current count of data points.
        batch_mean: An array-like object containing the batch mean.
        batch_var: An array-like object containing the batch variance.
        batch_count: An integer specifying the number of data points in the batch.

    Returns:
        new_mean: An array-like object containing the new mean.
        new_var: An array-like object containing the new variance.
        new_count: An integer specifying the new count of data points.
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def sort_edge_index(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    sort_by_row=True
):
    """
    Row-wise sorts 'edge_index' using NumPy.

    Args:
        edge_index (np.ndarray): The edge indices.
        edge_attr (np.ndarray or List[np.ndarray], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: None)
        num_nodes (int, optional): The number of nodes, i.e.,
            'max_val + 1' of 'edge_index'. (default: None)
        sort_by_row (bool, optional): If set to False, will sort
            'edge_index' column-wise/by destination node.
            (default: True)

    Returns:
        np.ndarray if 'edge_attr' is not passed, else
        (np.ndarray, Optional[np.ndarray] or List[np.ndarray])
    """

    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    perm = np.argsort(idx)

    edge_index_sorted = edge_index[:, perm]

    if edge_attr is None:
        return edge_index_sorted, None
    elif isinstance(edge_attr, np.ndarray):
        return edge_index_sorted, edge_attr[perm]
    elif isinstance(edge_attr, (list, tuple)):
        return edge_index_sorted, [e[perm] for e in edge_attr]

    return edge_index_sorted


def get_all_possible_link_pairs(G):
    rows, cols = np.triu_indices(G.number_of_nodes(), k=1)
    all_possible_link_pairs = np.array([rows, cols]).T
    return all_possible_link_pairs

def get_unexistent_link_pairs(G, existent_link_pairs=None):
    if existent_link_pairs is None:
        existent_link_pairs = np.array(list(G.edges()))
    all_possible_link_pairs = get_all_possible_link_pairs(G)
    unexistent_link_pairs = set(map(tuple, all_possible_link_pairs)) - set(map(tuple, existent_link_pairs))
    unexistent_link_pairs = np.array(list(unexistent_link_pairs))
    return unexistent_link_pairs

def get_useless_link_pairs(G, min_bw_resource=0):
    network_bw_adj_array = nx.adjacency_matrix(G, weight='bw').toarray()
    useless_link_indices = np.where(network_bw_adj_array < min_bw_resource)
    useless_link_pairs = np.array([useless_link_indices[0], useless_link_indices[1]]).T
    useless_link_pairs = [x for x in useless_link_pairs if x[0] <= x[1]]
    useless_link_pairs = np.array(useless_link_pairs)
    return useless_link_pairs

def get_random_unexistent_links(unexistent_link_pairs, num_added_links):
    num_unexistent_links = len(unexistent_link_pairs)
    num_added_links = min(num_added_links, num_unexistent_links)
    if num_added_links == 0:
        return np.array([[], []]).T

    if num_unexistent_links < num_added_links:
        raise Warning(f'Number of unexistent links {num_unexistent_links} is less than number of links {num_added_links}')
        # raise Exception(f'Number of unexistent links {num_unexistent_links} is less than number of links {unexistent_link_pairs}')
    newly_edge_pairs = np.random.permutation(unexistent_link_pairs)
    num_added_links = min(num_added_links, len(newly_edge_pairs))
    newly_edge_pairs = newly_edge_pairs[:num_added_links]
    return newly_edge_pairs