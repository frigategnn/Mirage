from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils import scatter


def global_pool_custom(
    x: Tensor,
    batch: Optional[Tensor],
    size: Optional[int] = None,
    roots_to_embed: Optional[Tensor] = None,
    reduce_type="sum",
) -> Tensor:
    dim = -1 if x.dim() == 1 else -2
    if batch is None:
        return x.sum(dim=dim, keepdim=x.dim() <= 2)
    size = int(batch.max().item() + 1) if size is None else size
    if roots_to_embed is not None:
        ind_nonroot_long = roots_to_embed.to(torch.long)
        cond = torch.where(ind_nonroot_long == 1, True, False)
        x = x[cond]
        batch = batch[cond]
    return scatter(x, batch, dim=dim, dim_size=size, reduce=reduce_type)
