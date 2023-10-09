import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.utils import degree


class GINConvEdge(MessagePassing):
    def __init__(this, emb_dim, net_norm="none"):
        super(GINConvEdge, this).__init__(aggr="add")
        print(f"Net-Norm: {net_norm}")
        if net_norm == "batchnorm":
            this.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim),
            )
        elif net_norm == "none":
            this.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim),
            )
        this.eps = nn.Parameter(torch.Tensor([0]))
        this.bond_encoder = BondEncoder(emb_dim=emb_dim)
        this.mlp_edge_attr = nn.Linear(2 * emb_dim, emb_dim)

    def forward(this, x, edge_index, edge_attr):
        edge_embedding = this.bond_encoder(edge_attr)
        out = this.mlp(
            (1 + this.eps) * x
            + this.propagate(
                edge_index,
                x=x,
                edge_attr=edge_embedding,
            )
        )
        return out

    def message(this, x_j, edge_attr):
        concat_j_edge = torch.cat((x_j, edge_attr), 1)
        concat_j_edge = this.mlp_edge_attr(concat_j_edge)
        return F.relu(x_j + edge_attr)

    def update(this, aggr_out):
        return aggr_out

    def reset_parameters(this):
        super(GINConvEdge, this).reset_parameters()
        [
            layer.reset_parameters()
            for layer in this.mlp.children()
            if hasattr(layer, "reset_parameters")
        ]
        this.mlp_edge_attr.reset_parameters()
        [
            [
                layer.reset_parameters()
                for layer in module.children()
                if hasattr(layer, "reset_parameters")
            ]
            for module in this.bond_encoder.children()
        ]


class GCNConvEdge(MessagePassing):
    def __init__(this, emb_dim):
        super(GCNConvEdge, this).__init__(aggr="add")
        this.linear = nn.Linear(emb_dim, emb_dim)
        this.root_emb = nn.Embedding(1, emb_dim)
        this.bond_encoder = BondEncoder(emb_dim=emb_dim)
        this.mlp_edge_attr = nn.Linear(2 * emb_dim, emb_dim)

    def forward(this, x, edge_index, edge_attr):
        edge_embedding = this.bond_encoder(edge_attr)
        x = this.linear(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = this.propagate(
            edge_index,
            x=x,
            edge_attr=edge_embedding,
            norm=norm,
        ) + F.relu(x + this.root_emb.weight) * 1.0 / deg.view(-1, 1)
        return out

    def message(this, x_j, edge_attr, norm):
        concat_j_edge = torch.cat((x_j, edge_attr), 1)
        concat_j_edge = this.mlp_edge_attr(concat_j_edge)
        return F.relu(concat_j_edge) + norm.view(-1, 1)

    def update(this, aggr_out):
        return aggr_out

    def reset_parameters(this):
        super(GCNConvEdge, this).reset_parameters()
        this.linear.reset_parameters()
        this.root_emb.reset_parameters()
        this.mlp_edge_attr.reset_parameters()
        [
            [
                layer.reset_parameters()
                for layer in module.children()
                if hasattr(layer, "reset_parameters")
            ]
            for module in this.bond_encoder.children()
        ]


class GNN_node(nn.Module):
    def __init__(
        this,
        num_layer,
        emb_dim,
        drop_ratio=0,
        JK="last",
        residual=False,
        gnn_type="gin",
        net_norm="none",
    ):
        super(GNN_node, this).__init__()
        this.num_layer = num_layer
        this.drop_ratio = drop_ratio
        this.JK = JK
        if this.JK == "cat":
            this.catnn = nn.Linear((num_layer + 1) * emb_dim, emb_dim)
        this.residual = residual
        this.net_norm = net_norm
        if this.num_layer < 1:
            raise ValueError("Number of GNN layers must be greater than 0")
        this.atom_encoder = AtomEncoder(emb_dim)
        this.convs = nn.ModuleList()
        this.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "GINConv":
                this.convs.append(GINConvEdge(emb_dim, this.net_norm))
            elif gnn_type == "GCNConv":
                this.convs.append(GCNConvEdge(emb_dim))
            elif gnn_type == "GATConv":
                this.convs.append(GATConv(emb_dim, emb_dim))
            elif gnn_type == "SAGEConv":
                this.convs.append(SAGEConv(emb_dim, emb_dim))
            else:
                raise ValueError(f"Undefined GNN type: {gnn_type}")
            this.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(this, x, edge_index, edge_attr, batch):
        x = x.long()
        h_list = [this.atom_encoder(x)]
        for layer in range(this.num_layer):
            h = this.convs[layer](h_list[layer], edge_index, edge_attr=edge_attr)
            if this.net_norm == "batchnorm":
                h = this.batch_norms[layer](h)
            if layer == this.num_layer - 1:
                h = F.dropout(h, this.drop_ratio, training=this.training)
            else:
                h = F.dropout(F.relu(h), this.drop_ratio, training=this.training)
            if this.residual:
                h += h_list[layer]
            h_list.append(h)
        node_representation = h
        if this.JK == "last":
            node_representation = h_list[-1]
        elif this.JK == "sum":
            node_representation = 0
            for layer in range(this.num_layer + 1):
                node_representation += h_list[layer]
        elif this.JK == "cat":
            node_representation = torch.cat(h_list, dim=-1)
            node_representation = this.catnn(node_representation)
        return node_representation

    def reset_parameters(this):
        if hasattr(this, "catnn"):
            this.catnn.reset_parameters()
        [
            [
                layer.reset_parameters()
                for layer in module.children()
                if hasattr(layer, "reset_parameters")
            ]
            for module in this.atom_encoder.children()
        ]
        [
            layer.reset_parameters()
            for layer in this.convs.children()
            if hasattr(layer, "reset_parameters")
        ]
        [
            layer.reset_parameters()
            for layer in this.batch_norms.children()
            if hasattr(layer, "reset_parameters")
        ]


class GNNEdgeBased(nn.Module):
    def __init__(
        this,
        num_layer=5,
        emb_dim=300,
        gnn_type="GCN",
        virtual_node=False,
        residual=False,
        drop_ratio=0.5,
        JK="last",
        dataset="None",
        net_norm="none",
    ):
        super().__init__()
        this.num_layer = num_layer
        this.drop_ratio = drop_ratio
        this.JK = JK
        this.emb_dim = emb_dim
        if this.num_layer < 1:
            raise ValueError(f"Num layers received ({this.num_layer}) is less than 1")
        this.gnn_node = GNN_node(
            num_layer,
            emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            net_norm=net_norm,
        )

    def forward(this, x, edge_index, edge_attr, batch):
        return this.gnn_node(x, edge_index, edge_attr, batch)

    def reset_parameters(this):
        this.gnn_node.reset_parameters()
