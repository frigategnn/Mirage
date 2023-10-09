r"""This code generates MPTrees for ogbg datasets. The code right now is
specific for molbace (it deals with edge attribute issues specific to it).
It can be adapted to others.
"""
import os
import time
import pickle
import argparse
from io import StringIO
from pprint import pprint
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
import pygcanl

from torch_geometric.data import Data

from utils import disjointed_union, roots_to_embed
from utils import get_dataloader

from data_utils import preprocess_dataset, get_raw_dataset, get_data
from data_utils import prettify_canonical_label, parse_canonical_label
from data_utils import canonical_label_to_naturals, tree_class_ctr, get_invalid_trees
from data_utils import pyfpgrowth_wrapper
from data_utils import preprocess_dataset_test

from model import GCN
from optimization import train_and_test as train

# import networkx as nx
# import matplotlib.pyplot as plt


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
        default="ogbg-molhiv",
        type=str,
        help="dataset name. right now this is fine tuned for"
        "ogbg-molhiv dataset only.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
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
        "--params_obj_name",
        type=str,
        required=True,
        help="Name of the params obj within the params file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help="Number of train runs to perform (GradientDescent with random restarts)",
    )
    args = parser.parse_args()
    extra_params = getattr(import_module(args.params_file), args.params_obj_name)
    dataset_name = args.dataset_name
    output_filename = os.path.join(
        args.output_dir, f"{dataset_name}_{args.run_num}_log.txt"
    )
    pprint(args)
    pprint(extra_params)
    pprint(f"Outputting logs to {output_filename}")
    append_to_file({"seed": seed}, output_filename)
    append_to_file(extra_params, output_filename)
    dataset = get_raw_dataset(dataset_name)
    split = dataset.get_idx_split()
    train_dataset = dataset[split["train"]]
    val_dataset = dataset[split["valid"]]
    test_dataset = dataset[split["test"]]
    (
        proc_data,
        node_label_map_orig,
        edge_label_map_orig,
        node_label_map_full,
    ) = preprocess_dataset(train_dataset, dataset)
    # nf = node_feature[0]  # is this correct?
    # ef = set(edge_feature tuples)
    # nff = set(node_feature tuples)
    # nf = nff[0]
    # 1. node_label_map_orig: nf ->  {0, 1, ..., |nf| } (:= nfctr)
    # 2. edge_label_map_orig: ef ->  {0, 1, ..., |ef| } (:= efctr)
    # 3. node_label_map_full: nff -> {0, 1, ..., |nff|} (:=nffctr)
    # 4. node_label_map: tree_node -> nfctr
    # 5. edge_label_map: tree_node x tree_node -> efctr
    # 6. node_label_map_original: tree_node -> nffctr
    # the keys() of node_label_map and node_label_map_original defined
    # set of nodes and keys() of edge_label_map define edges of the graph
    #
    # get_data() uses this information to create correct pyg graph (?)
    # it takes args in this order: 4,5,6,1,2,3
    # constructs i2n as inverse of 1, i2n_o as inverse of 3 and i2e as
    # inverse of 2.
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
    tree_class_count = tree_class_ctr(classes, mapped_dataset)
    invalid_tree_idx = get_invalid_trees(tree_class_count)
    invalid_tree_idx = {}
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
    # threshs = {0: 120, 1: 18}
    # threshs = {0: 144, 1: 18}
    # threshs = {0: 150, 1: 18}
    # threshs = {0: 60, 1: 10}
    threshs = extra_params.threshs
    append_to_file(threshs, output_filename)
    patterns = pyfpgrowth_wrapper(classwise, threshs)
    #freq, pats = list(zip(*((v,k) for k,v in patterns[0].items())))
    #probs = np.asarray(freq)/np.sum(freq)
    #rng = np.random.RandomState(seed=vars(args).get('seed',0))
    #sampled_0 = rng.choice(np.arange(len(pats)), p=probs, size=len(patterns[1]), replace=False)
    #patterns[0] = {pats[k]:freq[k] for k in sampled_0}
    print(
        f"Unique trees in class 0: {len(classwise_trees[0])},"
        f" in class 1: {len(classwise_trees[1])}"
    )
    print(
        f"Patterns mined in class 0: {len(patterns[0])} (thresh: {threshs[0]}),"
        f" in class 1: {len(patterns[1])} (thresh: {threshs[1]})"
    )
    print(f"#(class 0)/#(class 1) = {len(patterns[0])/len(patterns[1]):.2f}")
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
    append_to_file(
        f"#(class 0)/#(class 1) = {len(patterns[0])/len(patterns[1]):.2f}",
        output_filename,
    )
    inv_mapping = {v: k for k, v in mapping.items()}
    reconstructed = {}
    # frequency is stored in patterns.
    # I need frequency with reconstructed sample mapping and
    # I need a dataset size to be generated.
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
            frq[idx] = {'freq':freq, 'data': datas}
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
    #
    test_dataset_proc = preprocess_dataset_test(test_dataset, node_label_map_orig, edge_label_map_orig, node_label_map_full)
    test_dataset_canonicalized = pygcanl.canonical(test_dataset_proc, hops)
    test_dataset_prettified = [[prettify_canonical_label(tree) for tree in graph] for graph in test_dataset_canonicalized]
    prettified_collapsed = set()
    [[[prettified_collapsed.add(inv_mapping[tree]) for tree in graph] for graph in dset.keys()] for dset in patterns.values()]
    test_dataset_prettified = [[tree for tree in graph if tree in prettified_collapsed] for graph in test_dataset_prettified]
    test_dataset_reconstructed = []
    assert len(test_dataset_prettified) == len(test_dataset)
    for trees, data in zip(test_dataset_prettified, test_dataset):
        cls = data.y.item()
        if len(trees) == 0:
            datum = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, roots_to_embed=torch.ones(data.x.shape[0]))
        else:
            datum = []
            for tree in trees:
                node_label_map, edge_label_map, node_label_map_original = parse_canonical_label(tree)
                tree_ = get_data(node_label_map, edge_label_map, node_label_map_original, node_label_map_orig, edge_label_map_orig, node_label_map_full, cls)
                datum.append(tree_)
            datum = disjointed_union(datum, cls)
        test_dataset_reconstructed.append(datum)
    assert not any(hasattr(d,'node_attr') for d in test_dataset_reconstructed)
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
    if hasattr(extra_params, 'lr'):
        opt_args = {
                "lr": extra_params.lr,
                "weight_decay": extra_params.wd,
        }
        append_to_file(opt_args, output_filename)
    else:
        opt_args = None
    append_to_file(device, output_filename)
    append_to_file(model_params, output_filename)
    model = GCN(**model_params).to(device)

    kwargs = {"batch_size": extra_params.train_batch_size, "shuffle": True}
    append_to_file(kwargs, output_filename)
    # dataset length parameter needs to be added to config
    train_loader = get_dataloader(dataset, **kwargs, dataset_len=1000)

    kwargs = {"batch_size": extra_params.val_batch_size, "shuffle": False}
    val_loader = get_dataloader(val_dataset, **kwargs)
    append_to_file(kwargs, output_filename)

    kwargs = {"batch_size": extra_params.val_batch_size, "shuffle": False}
    test_loader = get_dataloader(test_dataset, **kwargs)
    append_to_file(kwargs, output_filename)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    criterion = torch.nn.CrossEntropyLoss()
    append_to_file(criterion, output_filename)
    # append_to_file("Turning off frequency based sampling for comparison", output_filename)
    train(
        model, dataloaders, criterion, device, args.epochs, args.runs, output_filename,
        args.output_dir, opt_args
    )


if __name__ == "__main__":
    import numpy as np

    s = np.random.randint(2000)
    print(f"seed = {s}")
    torch.manual_seed(s)
    main(s)
