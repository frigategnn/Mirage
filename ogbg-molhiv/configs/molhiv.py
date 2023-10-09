r"""This file contains extra configuration parameters that pertain to a
the ogbg-molhiv dataset."""
from pprint import pprint
from io import StringIO


class MolhivGCNConfig:
    r"""This class defines the parameters for GCN model."""

    def __init__(this):
        this.n_hops = 3
        # this.threshs = {0: 120, 1: 18}
        # this.threshs = {0: 144, 1: 18}
        this.threshs = {0: 1800, 1: 115}
        # this.threshs = {0: 120, 1:17}
        # this.threshs = {0: 90, 1: 17}
        this.model_reduce_type = "sum"
        this.model_JK = "last"
        this.model_net_norm = "batchnorm"
        this.model_hidden_channels = 128
        this.model_num_layers = 3
        this.model_gnn_type = "GCNConv"
        this.model_drop_ratio = 0.1
        this.train_batch_size = 64
        this.val_batch_size = 8192

    def __repr__(this):
        rep = {
            "n_hops": this.n_hops,
            "threshs": this.threshs,
            "model_reduce_type": this.model_reduce_type,
            "model_JK": this.model_JK,
            "model_net_norm": this.model_net_norm,
            "model_hidden_channels": this.model_hidden_channels,
            "model_num_layers": this.model_num_layers,
            "model_gnn_type": this.model_gnn_type,
            "model_drop_ratio": this.model_drop_ratio,
        }
        with StringIO() as s:
            pprint(rep, indent=2, stream=s)
            repr_ = s.getvalue()
        return repr_


class MolhivGATConfig:
    r"""This class defines the parameters for GCN model."""

    def __init__(this):
        this.n_hops = 3
        # this.threshs = {0: 120, 1: 18}
        # this.threshs = {0: 144, 1: 18}
        this.threshs = {0: 1800, 1: 115} # with n_hops=3
        # this.threshs = {0: 2750, 1: 150} # with n_hops=2
        this.model_reduce_type = "sum"
        this.model_JK = "sum"
        this.model_net_norm = "none"
        this.model_hidden_channels = 128
        this.model_num_layers = 2
        this.lr = 1e-3
        this.wd = 1e-6
        this.model_gnn_type = "GATConv"
        this.model_drop_ratio = 0.1
        this.train_batch_size = 32
        this.val_batch_size = 8192

    def __repr__(this):
        rep = {
            "n_hops": this.n_hops,
            "threshs": this.threshs,
            "model_reduce_type": this.model_reduce_type,
            "model_JK": this.model_JK,
            "model_net_norm": this.model_net_norm,
            "model_hidden_channels": this.model_hidden_channels,
            "model_num_layers": this.model_num_layers,
            "model_gnn_type": this.model_gnn_type,
            "model_drop_ratio": this.model_drop_ratio,
        }
        with StringIO() as s:
            pprint(rep, indent=2, stream=s)
            repr_ = s.getvalue()
        return repr_


class MolhivGINConfig:
    r"""This class defines the parameters for GCN model."""

    def __init__(this):
        this.n_hops = 3
        # this.threshs = {0: 120, 1: 18}
        # this.threshs = {0: 144, 1: 18}
        this.threshs = {0: 1800, 1: 115}
        this.model_reduce_type = "sum"
        this.model_JK = "sum"
        this.model_net_norm = "none"
        this.model_hidden_channels = 128
        this.model_num_layers = 2
        this.model_gnn_type = "GINConv"
        this.model_drop_ratio = 0.2
        this.train_batch_size = 32
        this.val_batch_size = 8192

    def __repr__(this):
        rep = {
            "n_hops": this.n_hops,
            "threshs": this.threshs,
            "model_reduce_type": this.model_reduce_type,
            "model_JK": this.model_JK,
            "model_net_norm": this.model_net_norm,
            "model_hidden_channels": this.model_hidden_channels,
            "model_num_layers": this.model_num_layers,
            "model_gnn_type": this.model_gnn_type,
            "model_drop_ratio": this.model_drop_ratio,
        }
        with StringIO() as s:
            pprint(rep, indent=2, stream=s)
            repr_ = s.getvalue()
        return repr_


molhiv_gcn_config = MolhivGCNConfig()
molhiv_gat_config = MolhivGATConfig()
molhiv_gin_config = MolhivGINConfig()
