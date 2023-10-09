import os
import tqdm
import zipfile
import requests
import pickle
import networkx as nx

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
class DDExistsError(BaseException):
    def __init__(this, thepath, thefile):
        this.thepath = thepath
        this.thefile = thefile
    def __repr__(this):
        return f"{this.thefile} already exists in {this.thepath}. Please remove it for a clean download."
class DDNotExistsError(BaseException):
    def __init__(this, thepath, thefile):
        this.thepath = thepath
        this.thefile = thefile
    def __repr__(this):
        return f"{this.thefile} doesn't exist in {this.thepath}."
class DDError(BaseException):
    def __init__(this, thepath):
        this.thepath = thepath
    def __repr__(this):
        return f"Something failed while trying to extract DD.zip in {this.thepath}"


def download_DD(root):
    if os.path.isfile(os.path.join(root, "DD.zip")):
        raise DDExistsError(root, "DD.zip")
    r = requests.get("https://www.chrsmrrs.com/graphkerneldatasets/DD.zip", stream=True)
    with open(os.path.join(root,"DD.zip"),"wb") as dd_file_handle:
        loop = tqdm.tqdm(r.iter_content(chunk_size=256), ascii=True)
        loop.set_description("Downloading DD Dataset")
        for chunk in loop:
            dd_file_handle.write(chunk)


def read_adj(adj_file: str):
    r"""Utility to read adj file"""
    edges = []
    with open(adj_file, "r") as adj_handle:
        for line in adj_handle:
            edge_pair = [token.strip() for token in line.strip().split(",")]
            assert len(edge_pair) == 2, f"Edge pair length is not correct: {edge_pair}"
            edges.append((int(edge_pair[0]), int(edge_pair[1])))
    return edges


def read_dict(filename: str):
    r"""Utility to read dictionary format files (each line contains
    labels for object at index = line number."""
    dictionary = {}
    with open(filename, "r") as file_handle:
        for linum, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            assert (
                len(stripped.split()) == 1
            ), f"Wrong format of line, see README.txt: {line}"
            try:
                int(float(stripped))
            except ValueError as exc:
                raise AssertionError(
                    f"Invalid label format, see README.txt: {stripped}"
                ) from exc
            dictionary[linum] = int(float(stripped))
    return dictionary


def read_dd_data(root):
    r"""Read the full raw dataset"""
    adj_file = os.path.join(root,"DD_A.txt")
    graph_indicator_file = os.path.join(root,"DD_graph_indicator.txt")
    graph_labels_file = os.path.join(root,"DD_graph_labels.txt")
    node_labels_file = os.path.join(root,"DD_node_labels.txt")
    adj_contents = read_adj(adj_file)
    graph_indicators = read_dict(graph_indicator_file)
    graph_labels = read_dict(graph_labels_file)
    node_labels = read_dict(node_labels_file)
    graphs = create_dd_graphs(adj_contents, graph_indicators, graph_labels, node_labels)
    return graphs


def create_dd_graphs(adj_contents, graph_indicators, graph_labels, node_labels):
    r"""Create networkx graphs from the read data"""
    graphs = [None] + [
        nx.Graph() for _ in graph_labels
    ]  # work around for 1-based indexing
    for n_idx in node_labels.keys():
        graph_indicator = graph_indicators[n_idx]
        if node_labels[n_idx] not in [19,6,14,3,15,2,17,10,16,4,12,7,20,8,1,11]:
            # these have been hand-picked
            node_label=0
        else:
            node_label = node_labels[n_idx]
        graphs[graph_indicator].add_nodes_from([(n_idx, {"label": node_label})])
    for edge in adj_contents:
        graph_indicator1 = graph_indicators[edge[0]]
        graph_indicator2 = graph_indicators[edge[1]]
        assert (
            graph_indicator1 == graph_indicator2
        ), f"Nodes do not belong to same graph - edge: {edge}, graph_ids: {(graph_indicator1, graph_indicator2)}"
        graphs[graph_indicator1].add_edges_from([edge])
    for gid, glab in graph_labels.items():
        graphs[gid].graph["label"] = glab
    return graphs[1:]  # work around for 1-based indexing (contd.)


def convert_to_pyg_DD(renumbered_nx_graphs):
    r"""Convert networkx graphs to pytorch geometric graph with the for-
    mat required by pygcanl utility."""
    datas = []
    for graph in renumbered_nx_graphs:
        x = []
        node_attr = []
        edge_index_row = []
        edge_index_col = []
        edge_attr = []
        for node in graph.nodes:
            x.append(1)
            node_attr.append(graph.nodes[node]["label"])
        for edge in graph.edges():
            edge_index_row.append(edge[0])
            edge_index_col.append(edge[1])
            edge_index_row.append(edge[1])
            edge_index_col.append(edge[0])
            edge_attr.extend([1, 1])
        x = torch.tensor(x)
        edge_index = torch.tensor([edge_index_row, edge_index_col])
        node_attr = torch.tensor(node_attr)
        edge_attr = torch.tensor(edge_attr)
        cls = torch.tensor(graph.graph["label"]-1)
        data = Data(
            x=x,
            edge_index=edge_index,
            original_x=x.clone(),
            edge_attr=edge_attr,
            node_attr=node_attr,
            original_edge_attr=edge_attr.clone(),
            y=cls,
        )  # weird format required by my weird code
        datas.append(data)
    return datas


def preprocess_DD(root):
    # check if already preprocessed
    if os.path.isfile(os.path.join(root, "pyg_DD.pkl")):
        raise DDExistsError(root, "pyg_DD.pkl")
    # check if DD.zip exists
    if not os.path.isfile(os.path.join(root, "DD.zip")):
        raise DDNotExistsError(root, "DD.zip")
    # extract DD.zip
    try:
        with zipfile.ZipFile(os.path.join(root, "DD.zip")) as DD_zipfile:
            DD_zipfile.extractall(path=root)
    except BaseException as e:
        raise DDError(root) from e
    # preprocess the labels
    networkx_graphs = read_dd_data(os.path.join(root, "DD"))
    renumbered_nx_graphs = [
        nx.convert_node_labels_to_integers(g) for g in networkx_graphs
    ]
    pyg_datas = convert_to_pyg_DD(renumbered_nx_graphs)
    with open(os.path.join(root, "pyg_DD.pkl"), "wb") as f:
        pickle.dump(pyg_datas, f)


def get_DD_dataset(root):
    # check if already downloaded
    if os.path.isfile(os.path.join(root,"DD.zip")):
        # check if extracted and alread processed
        if not os.path.isfile(os.path.join(root,"pyg_DD.pkl")):
            preprocess_DD(root)
    else:
        download_DD(root)
        preprocess_DD(root)
    with open(os.path.join(root,"pyg_DD.pkl"), "rb") as dd_file:
        dd_list_dataset = pickle.load(dd_file)
    return dd_list_dataset


# -------------------  GET RAW DATASET  -----------------------
def get_raw_dataset(dataset_name):
    if dataset_name in ["ogbg-molhiv", "ogbg-molbbbp", "ogbg-molbace"]:
        dataset = PygGraphPropPredDataset(name=dataset_name)
    elif dataset_name in ["DD"]:
        dataset = get_DD_dataset(root="dataset")
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
