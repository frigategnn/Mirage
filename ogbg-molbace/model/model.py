from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv

from .gnns_edge2 import GNNEdgeBased as GCNEdge
from .custom_pool import global_pool_custom


class GCN(Module):
    def __init__(
        this,
        reduce_type="sum",
        JK="last",
        net_norm="none",
        hidden_channels=128,
        num_layers=3,
        gnn_type="GATConv",
        drop_ratio=0.6,
        num_classes=2,
    ):
        super(GCN, this).__init__()
        this.drop_ratio = drop_ratio
        this.reduce_type = reduce_type
        print(f"Reduce-type: {this.reduce_type}")
        this.node_embed = GCNEdge(
            JK=JK,
            net_norm=net_norm,
            emb_dim=hidden_channels,
            num_layer=num_layers,
            gnn_type=gnn_type,
            drop_ratio=drop_ratio,
        )
        this.lin = Linear(hidden_channels, num_classes)

    def forward(this, x, edge_index, edge_attr, batch, roots_to_embed=None):
        x = this.node_embed(x, edge_index, edge_attr, batch)
        x = global_pool_custom(
            x, batch, roots_to_embed=roots_to_embed, reduce_type=this.reduce_type
        )
        x = F.dropout(x, p=this.drop_ratio, training=this.training)
        x = this.lin(x)
        return x

    def reset_parameters(this):
        this.node_embed.reset_parameters()
        this.lin.reset_parameters()
