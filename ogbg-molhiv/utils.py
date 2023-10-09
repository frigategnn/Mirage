r"""List of utilities used by this code. Though this code started as one
file, so many of the utilities reside inside mine_trees.py"""
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def disjointed_union(tree_list, class_=None, device=None):
    r"""Computes disjointed union of trees inside tree_list. trees are in
    torch_geometric.data.Data format. Returns a torch_geometric.data.Data
    graph without the roots_to_embed information."""
    tree_list = [
        tree for tree in tree_list if tree.x.shape[0] > 1
    ]  # skipping single node trees for now because it causes issues
    total_nodes = sum(tree.x.shape[0] for tree in tree_list)
    total_edges = sum(tree.edge_index.shape[1] for tree in tree_list)
    if total_nodes <= 1:
        return None  # if the whole graph has just one node, then we return none
    if device is None:
        device = tree_list[0].x.device
    assert all(
        device == tree.x.device for tree in tree_list
    ), "Trees must be on same device."
    assert all(
        tree.x.shape[1] == tree_list[0].x.shape[1] for tree in tree_list
    ), "Number of node features are different."
    assert all(
        tree.y == tree_list[0].y for tree in tree_list
    ), "Tree must be in same class."
    if hasattr(tree_list[0], "edge_attr"):
        assert all(
            hasattr(tree, "edge_attr") for tree in tree_list
        ), "Not all trees have edge features."
        assert all(
            tree.edge_attr.shape[1] == tree_list[0].edge_attr.shape[1]
            for tree in tree_list
        ), "Number of edge features are different."
        hasedgeattr = True
    else:
        hasedgeattr = False
    x = torch.zeros(
        (total_nodes, tree_list[0].x.shape[1]),
        device=device,
        dtype=tree_list[0].x.dtype,
    )
    if hasedgeattr:
        edge_attr = torch.zeros(
            (total_edges, tree_list[0].edge_attr.shape[1]),
            device=device,
            dtype=tree_list[0].edge_attr.dtype,
        )
    edge_index = torch.zeros((2, total_edges), device=device, dtype=torch.long)
    node_start = 0
    edge_start = 0
    for tree in tree_list:
        num_nodes = tree.x.shape[0]
        idxs = torch.arange(num_nodes) + node_start
        x[idxs, :] = tree.x
        num_edges = tree.edge_index.shape[1]
        idxs = torch.arange(num_edges) + edge_start
        if hasedgeattr:
            edge_attr[idxs, :] = tree.edge_attr
        edge_index_extract = tree.edge_index
        edge_index_extract = edge_index_extract + node_start
        edge_index[:, idxs] = edge_index_extract
        node_start += num_nodes
        edge_start += num_edges
    if hasedgeattr:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=class_)
    else:
        data = Data(x=x, edge_index=edge_index, y=class_)
    return data


def roots_to_embed(data):
    nn = data.x.shape[0]
    root = torch.zeros((nn,), device=data.x.device)
    root_indices = torch.tensor(
        list(
            set(data.edge_index[1].tolist()).difference(
                set(data.edge_index[0].tolist())
            )
        )
    ).to(data.x.device)
    root[root_indices] = 1
    data.roots_to_embed = root
    return data


class myDataset(Dataset):
    def __init__(this, dataset_list, dataset_len=0, **kwargs):
        super().__init__()
        this.dataset_list = [list(d.values())[0] for d in dataset_list]
        this.dataset_len = max(len(this.dataset_list), dataset_len)
        this.rng = np.random.RandomState(seed=kwargs.get('seed', 0))
        freq, data = list(zip(*((d['freq'],d['data']) for d in this.dataset_list)))
        probs = np.asarray(freq)/np.sum(freq)
        this.data = [data[this.rng.choice(np.arange(len(data)), p=probs)] for _ in range(this.dataset_len)]
        this.dataset_len = len(this.dataset_list)
        this.data = data

    def len(this):
        return this.dataset_len

    def get(this, idx):
        return this.data[idx]


def get_dataloader(dataset_classwise, **kwargs):
    if isinstance(dataset_classwise, dict):
        dataset = (
            dataset_classwise[0] + dataset_classwise[1]
        )  # for now assumes only two classes named 0 and 1 exist
        mydataset = myDataset(dataset, **kwargs)
        if 'dataset_len' in kwargs:
            del kwargs['dataset_len']
        dataloader = DataLoader(mydataset, **kwargs)
    else:
        dataloader = DataLoader(dataset_classwise, **kwargs)
    return dataloader
