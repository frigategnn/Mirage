r"""This code loads and runs mine trees for DD dataset. DD dataset was
processed differently from the other ogbg datasets. So this doesn't make
use of the dataset preprocessing utilities available in data_utils.py,
it directly loads from pickle file.
There is also no default split available, so I'll just make one split.
"""
import os
import time
import pickle
import argparse
from io import StringIO
from pprint import pprint
from datetime import datetime
from importlib import import_module

import torch
import pygcanl
import numpy as np
from torch_geometric.data import Data

from utils import disjointed_union, roots_to_embed
from utils import get_dataloader

from data_utils import prettify_canonical_label, parse_canonical_label
from data_utils import canonical_label_to_naturals, tree_class_ctr, get_invalid_trees
from data_utils import pyfpgrowth_wrapper, get_raw_dataset

from model import GCN_DD as GCN
from optimization import train_and_test as train

# import networkx as nx
# import matplotlib.pyplot as plt

def get_data(
    node_label_map,
    edge_label_map,
    node_label_map_original,
    node_label_map_orig,
    edge_label_map_orig,
    node_label_map_full,
    class_,
):
    i2n = {v: k for k, v in node_label_map_orig.items()}
    i2n_o = {v: k for k, v in node_label_map_full.items()}
    i2e = {v: k for k, v in edge_label_map_orig.items()}
    # features = [list(i2n_o[int(n)]) for idx, n in node_label_map_original.items()]
    # node_labels = [i2n[int(n)] for idx, n in node_label_map.items()]
    features, node_labels = [], []
    for idx, n in node_label_map_original.items():
        feature = i2n_o[int(n)]
        features.append(feature)
        #
        node_label = i2n[int(node_label_map[idx])]
        node_labels.append(node_label)
    #
    edge_index_row = []
    edge_index_col = []
    edge_attr = []
    for k, v in edge_label_map.items():
        edge_index_row.append(k[0])  # k[0] is child (source)
        edge_index_col.append(k[1])  # k[1] is parent (destination)
        edge_attr.append(i2e[int(v)])
    edge_index = [edge_index_row, edge_index_col]
    data = Data(
        x=torch.tensor(features).reshape(-1,1),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_attr).reshape(-1,1),
        node_labels=torch.tensor(node_labels),
    )
    return data


def append_to_file(message, filename):
    r"""This functions is used to log strings and general python objects
    in a pretty format to the log file specified by param filename."""
    if not isinstance(message, str):
        with StringIO() as stream:
            pprint(message, stream=stream)
            message = stream.getvalue()
    with open(filename, "a", encoding="utf-8") as output_file:
        output_file.write("-" * 20 + "\n")
        output_file.write(datetime.now().strftime("%d-%m-%Y %H:%M:%S\n"))
        output_file.write(message)
        output_file.write("\n")


def main(seed):
    r"""The main function. Consider this the algorithm."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="number of epochs to run the training step for",
    )
    parser.add_argument(
        "--run_num",
        required=True,
        type=int,
        help="experiment run number for bookkeeping",
    )
    parser.add_argument(
        "--dataset_name",
        default="DD",
        type=str,
        help="dataset name. right now this is fine tuned for"
        "DD dataset only.",
    )
    parser.add_argument(
        "--dataset_path",
        default="dataset/pyg_DD.pkl",
        type=str,
        help="path to the pyg version of the processed DD dataset."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_dd",
        type=str,
        help="directory where experiment outputs will be saved",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        required=True,
        help="Path to params file containing a dataclass specifying parameters",
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help="Number of runs of the algorithm",
    )
    parser.add_argument(
        "--params_obj_name",
        type=str,
        required=True,
        help="Name of the params obj within the params file",
    )
    args = parser.parse_args()
    extra_params = getattr(import_module(args.params_file), args.params_obj_name)
    dataset_name = args.dataset_name
    output_filename = os.path.join(
        args.output_dir,
        f"{dataset_name}_{args.run_num}_{extra_params.model_gnn_type}_log.txt",
    )
    pprint(args)
    pprint(extra_params)
    pprint(f"Outputting logs to {output_filename}")
    append_to_file(extra_params, output_filename)
    append_to_file(f"Seed is {seed}", output_filename)
    append_to_file('Turned off frequency proportional sampling', output_filename)
    dd_dataset = get_raw_dataset(dataset_name)
    rng = np.random.RandomState(seed=0)
    perm = rng.permutation(len(dd_dataset))
    train_len = int(0.8 * len(perm))
    val_len = int(0.1 * len(perm))
    train_idx = perm[:train_len]
    val_idx = perm[train_len:train_len+val_len]
    train_dataset = [dd_dataset[idx] for idx in train_idx]
    val_dataset = [dd_dataset[idx] for idx in val_idx]
    test_dataset = [dd_dataset[idx] for idx in perm[train_len+val_len:]]
    proc_data = train_dataset
    # the following are identity maps
    # because we didn't transform the data at all
    node_label_map_orig = {1:1}
    edge_label_map_orig = {1:1}
    node_label_map_full = {}
    for d in dd_dataset:
        for n in d.node_attr:
            nitem = n.item()
            node_label_map_full[nitem] = nitem
    hops = extra_params.n_hops
    append_to_file(f"n_hops (used for MPTree construction): {hops}", output_filename)
    time_0 = time.perf_counter()
    ret = pygcanl.canonical(proc_data, hops)
    time_1 = time.perf_counter()
    print(f"It took {time_1-time_0:.4f}s to compute MPTree representations")
    append_to_file(
        f"It took {time_1-time_0:.4f}s to compute MPTree representations",
        output_filename,
    )
    classes = [d.y.item() for d in proc_data]
    unique_classes = set(classes)
    prettified = [[prettify_canonical_label(tree) for tree in graph] for graph in ret]
    mapping = canonical_label_to_naturals(prettified)
    mapped_dataset = [[mapping[tree] for tree in graph] for graph in prettified]
    classwise_mapped_graphs = {
        class__: [
            graph for graph, class_ in zip(mapped_dataset, classes) if class_ == class__
        ]
        for class__ in unique_classes
    }
    classwise_trees = {0: set(), 1: set()}
    for class_ in (0, 1):
        for graph in classwise_mapped_graphs[class_]:
            for tree in graph:
                classwise_trees[class_].add(tree)
    # tree_class_count = tree_class_ctr(classes, mapped_dataset)
    # invalid_tree_idx = get_invalid_trees(tree_class_count)
    invalid_tree_idx = {}
    print(f"invalid_tree_idx = {invalid_tree_idx}")
    selected_dataset = [
        list({tree for tree in graph if tree not in invalid_tree_idx})
        for graph in mapped_dataset
    ]
    classwise = {
        class__: [
            graph
            for graph, class_ in zip(selected_dataset, classes)
            if class_ == class__ and len(graph) > 0
        ]
        for class__ in unique_classes
    }
    threshs = extra_params.threshs
    append_to_file(threshs, output_filename)
    patterns = pyfpgrowth_wrapper(classwise, threshs)
    print(
        f"Unique trees in class 0: {len(classwise_trees[0])},"
        f" in class 1: {len(classwise_trees[1])}"
    )
    print(
        f"Patterns mined in class 0: {len(patterns[0])} (thresh: {threshs[0]}),"
        f" in class 1: {len(patterns[1])} (thresh: {threshs[1]})"
    )
    append_to_file(
        f"Unique trees in class 0: {len(classwise_trees[0])},"
        f" in class 1: {len(classwise_trees[1])}",
        output_filename,
    )
    append_to_file(
        f"Patterns mined in class 0: {len(patterns[0])} (thresh: {threshs[0]}),"
        f" in class 1: {len(patterns[1])} (thresh: {threshs[1]})",
        output_filename,
    )
    inv_mapping = {v: k for k, v in mapping.items()}
    reconstructed = {}
    for class_ in unique_classes:
        class_reconstructed = []
        for idx, (pattern, freq) in enumerate(patterns[class_].items()):
            datas = []
            frq = {}
            for tree in pattern:
                data = inv_mapping[tree]
                (
                    node_label_map,
                    edge_label_map,
                    node_label_map_original,
                ) = parse_canonical_label(data)
                data = get_data(
                    node_label_map,
                    edge_label_map,
                    node_label_map_original,
                    node_label_map_orig,
                    edge_label_map_orig,
                    node_label_map_full,
                    class_,
                )
                datas.append(data)
            frq[idx] = {'freq':freq, 'data':datas}
            class_reconstructed.append(frq)
        reconstructed[class_] = class_reconstructed
    dataset = {}
    for class_, recon in reconstructed.items():
        class_dataset = []
        for rec in recon:
            idx = list(rec.keys())[0]
            data = disjointed_union(rec[idx]['data'], class_)
            if data is None:
                # skipping when graph has only one node
                continue
            data = roots_to_embed(data)
            rec[idx]['data'] = data
            class_dataset.append(rec)
        dataset[class_] = class_dataset
    # add random samples
    #n = 3
    #c = 0
    #for class_ in unique_classes:
    #    while c < n:
    #        p = rng.randint(0, len(train_dataset))
    #        data = train_dataset[p]
    #        if data.y.item() != class_:
    #            continue
    #        data2 = Data(x=data.node_attr.reshape(-1,1), edge_index=data.edge_index, edge_attr=data.edge_attr.reshape(-1,1), y=data.y, roots_to_embed=data.x.float())
    #        dataset[class_].append(data2)
    #        c += 1
    #
    # ---------------------------------------------------
    if "gon" not in args.params_obj_name.lower():
        test_dataset_reconstructed = test_dataset
    else:
        test_dataset_canonicalized = pygcanl.canonical(test_dataset, hops)
        test_dataset_prettified = [[prettify_canonical_label(tree) for tree in graph] for graph in test_dataset_canonicalized]
        prettified_collapsed = set()
        [[[prettified_collapsed.add(inv_mapping[tree]) for tree in graph] for graph in dset.keys()] for dset in patterns.values()]
        test_dataset_prettified = [[tree for tree in graph if tree in prettified_collapsed] for graph in test_dataset_prettified]
        test_dataset_reconstructed = []
        assert len(test_dataset_prettified) == len(test_dataset)
        for trees, data in zip(test_dataset_prettified, test_dataset):
            cls = data.y.item()
            if len(trees) == 0:
                datum = Data(x=data.node_attr.reshape(-1,1), edge_index=data.edge_index, edge_attr=data.edge_attr.reshape(-1,1), y=data.y, roots_to_embed=data.x.float())
            else:
                datum = []
                for tree in trees:
                    node_label_map, edge_label_map, node_label_map_original = parse_canonical_label(tree)
                    tree_ = get_data(node_label_map, edge_label_map, node_label_map_original, node_label_map_orig, edge_label_map_orig, node_label_map_full, cls)
                    datum.append(tree_)
                datum = disjointed_union(datum, cls)
            test_dataset_reconstructed.append(datum)
        assert not any(hasattr(d,'node_attr') for d in test_dataset_reconstructed)
    # ---------------------------------------------------
    with open(f"{args.output_dir}/saved_dataset_{args.run_num}.pkl", "wb") as data_file:
        pickle.dump(dataset, data_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_params = {
        "reduce_type": extra_params.model_reduce_type,
        "JK": extra_params.model_JK,
        "net_norm": extra_params.model_net_norm,
        "hidden_channels": extra_params.model_hidden_channels,
        "num_layers": extra_params.model_num_layers,
        "gnn_type": extra_params.model_gnn_type,
        "drop_ratio": extra_params.model_drop_ratio,
    }
    append_to_file(device, output_filename)
    append_to_file(model_params, output_filename)
    model = GCN(**model_params).to(device)

    kwargs = {"batch_size": extra_params.train_batch_size, "shuffle": False}
    append_to_file(kwargs, output_filename)
    train_loader = get_dataloader(dataset, **kwargs)

    kwargs = {"batch_size": extra_params.val_batch_size, "shuffle": False}
    val_loader = get_dataloader(val_dataset, **kwargs)
    append_to_file(kwargs, output_filename)

    test_loader = get_dataloader(test_dataset_reconstructed, batch_size=8192, shuffle=False)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    criterion = torch.nn.CrossEntropyLoss()
    append_to_file(criterion, output_filename)
    train(model, dataloaders, criterion, device, args.epochs, args.runs, output_filename, args.output_dir)


if __name__ == "__main__":
    s = np.random.randint(2000)
    torch.manual_seed(s)
    main(s)
