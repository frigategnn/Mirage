import torch
import pyfpgrowth
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data


# ---------------  PROCESS DATASET (LABELS)  ------------------
def get_label_maps(dataset):
    edge_attrs = set()
    for d in dataset:
        edge_attr = d.edge_attr
        [
            edge_attrs.add(tuple(edge_attr[ea, :].tolist()))
            for ea in range(edge_attr.shape[0])
        ]
    node_labels = set()
    node_labels_full = set()
    for d in dataset:
        node_label = d.x[:, 0]
        node_label_full = d.x
        [node_labels.add(n.item()) for n in node_label]
        [
            node_labels_full.add(tuple(node_label_full[n, :].tolist()))
            for n in range(len(node_label_full))
        ]
    node_label_map = {n: idx for idx, n in enumerate(node_labels)}
    node_label_map_full = {n: idx for idx, n in enumerate(node_labels_full)}
    edge_label_map = {edge_config: idx for idx, edge_config in enumerate(edge_attrs)}
    return node_label_map, edge_label_map, node_label_map_full


def process_labels(dataset, node_label_map, edge_label_map, node_label_map_full):
    dataset_ = []
    for d in dataset:
        data = Data(
            x=torch.tensor(
                [node_label_map[n.item()] for n in d.x[:, 0]]
            ),  # since this goes into C++ and out comes just a string, the node features are lost
            original_x=d.x,
            edge_index=d.edge_index,
            y=d.y,
            original_edge_attr=d.edge_attr,
            edge_attr=torch.tensor(
                [
                    edge_label_map[tuple(d.edge_attr[ea, :].tolist())]
                    for ea in range(d.edge_attr.shape[0])
                ],
                dtype=torch.long,
            ),
            node_attr=torch.tensor(
                [
                    node_label_map_full[tuple(d.x[n, :].tolist())]
                    for n in range(d.x.shape[0])
                ]
            ),
        )
        dataset_.append(data)
    return dataset_


def preprocess_dataset(dataset, full_dataset):
    node_label_map, edge_label_map, node_label_map_full = get_label_maps(full_dataset)
    label_processed_dataset = process_labels(
        dataset,
        node_label_map,
        edge_label_map,
        node_label_map_full,
    )
    return label_processed_dataset, node_label_map, edge_label_map, node_label_map_full


def preprocess_dataset_test(dataset, node_label_map, edge_label_map, node_label_map_full):
    label_processed_dataset = process_labels(dataset, node_label_map, edge_label_map, node_label_map_full)
    return label_processed_dataset


# ------------  </PREPROCESS DATASET (LABELS)>  ---------------


# -------------------  GET RAW DATASET  -----------------------
def get_raw_dataset(dataset_name):
    if dataset_name in ["ogbg-molhiv", "ogbg-molbbbp", "ogbg-molbace"]:
        dataset = PygGraphPropPredDataset(name=dataset_name)
    return dataset


# ------------------  </GET RAW DATASET>  ---------------------


# ----------------  PROCESS CANONICAL LABELS  -----------------
def prettify_canonical_label(label):
    assert isinstance(label, str)
    tokens = label.split()
    label2 = f"<{tokens[0]},{tokens[1]}>"
    ptr = 2
    while ptr < len(tokens):
        if tokens[ptr] == "$":
            label2 += tokens[ptr]
            ptr += 1
        else:
            node_label = tokens[ptr]
            node_original_label = tokens[ptr + 1]
            edge_label = tokens[ptr + 2]
            ptr += 3
            label2 += f"<{node_label},{node_original_label},{edge_label}>"
    return label2


def canonical_label_to_naturals(dataset_labels):
    mapping = {}
    ctr = 0
    for graph_labels in dataset_labels:
        for label in graph_labels:
            if label not in mapping:
                mapping[label] = ctr
                ctr += 1
    return mapping


# --------------  </PROCESS CANONICAL LABELS>  ----------------


# ---------------  COMPUTE FREQUENT PATTERNS  -----------------
def tree_class_ctr(classes, dataset):
    tree_class_count = {}
    for class_, graph in zip(classes, dataset):
        for tree in graph:
            if tree not in tree_class_count:
                tree_class_count[tree] = {}
            if class_ not in tree_class_count[tree]:
                tree_class_count[tree][class_] = 0
            tree_class_count[tree][class_] += 1
    return tree_class_count


def get_invalid_trees(tree_class_count):
    trees = []
    for tree, ctr in tree_class_count.items():
        if len(ctr) != 1:
            trees.append(tree)
    return trees


def pyfpgrowth_wrapper(classwise, freq_thresholds):
    patterns = {}
    for class_, dataset in classwise.items():
        thresh = freq_thresholds[class_]
        class_patterns = pyfpgrowth.find_frequent_patterns(dataset, thresh)
        patterns[class_] = class_patterns
    return patterns


# -------------  </COMPUTE FREQUENT PATTERNS>  ----------------


def parse_canonical_label_bak(label):
    q = 0
    parsed = []
    idx = 0
    tmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == "<":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                nid += 1
                parsed.append({"node_label": tmp})
                q = 1
                tmp = ""
            elif label[idx] == "$":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                break
        if q == 1:
            if label[idx] == "$":
                # this will be used to pop parent for creating actual edges later
                stack.pop()
                pass
            if label[idx] == "<":
                q = 2
        if q == 2:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 3
        if q == 3:
            if label[idx].isnumeric():
                etmp += label[idx]
            if label[idx] == ">":
                parsed.append({"node_label": tmp, "edge_label": etmp})
                # create node here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                # create edges here
                # (to, from)
                edge_label_map[(nid, par)] = etmp
                nid += 1
                tmp = ""
                etmp = ""
                q = 1
    assert q == 1, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map


def parse_canonical_label(label):
    q = 0
    idx = 0
    tmp = ""
    otmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    node_label_map_original = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx] == "<":
                q = 1
        elif q == 1:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 2
        elif q == 2:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                tmp = ""
                otmp = ""
                nid += 1
                q = 3
        elif q == 3:
            if label[idx] == "<":
                q = 4
            elif label[idx] == "$":
                # pop stack here
                stack.pop()
        elif q == 4:
            if label[idx].isnumeric():
                etmp += label[idx]
            elif label[idx] == ",":
                q = 5
        elif q == 5:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 6
        elif q == 6:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create new node and edge here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                edge_label_map[(nid, par)] = etmp
                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html#from_networkx
                # in the source, edge_index[0,:] is sources and edge_index[1,:] is destinations
                nid += 1
                tmp = ""
                otmp = ""
                etmp = ""
                q = 3
    assert q == 3, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map, node_label_map_original


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
        feature = list(i2n_o[int(n)])
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
        x=torch.tensor(features),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_attr),
        node_labels=torch.tensor(node_labels),
    )
    return data
