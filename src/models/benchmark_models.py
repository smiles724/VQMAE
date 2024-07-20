from typing import Optional

import torch
import torch.nn as nn
from pykeops.torch import LazyTensor
from torch.nn import (Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d as BN, )
from torch_cluster import knn
from torch_geometric.nn import DynamicEdgeConv, PointConv, radius, global_max_pool, knn_interpolate
from torch_geometric.nn import EdgeConv

from src.utils.geometry import dMaSIFConv


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device))
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


@torch.jit.ignore
def keops_knn(x: torch.Tensor, y: torch.Tensor, k: int, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, cosine: bool = False, ) -> torch.Tensor:
    r"""Straightforward modification of PyTorch_geometric's knn method."""

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    y_i = LazyTensor(y[:, None, :])
    x_j = LazyTensor(x[None, :, :])

    if cosine:
        D_ij = -(y_i | x_j)
    else:
        D_ij = ((y_i - x_j) ** 2).sum(-1)

    D_ij.ranges = diagonal_ranges(batch_y, batch_x)
    idy = D_ij.argKmin(k, dim=1)  # (N, K)

    rows = torch.arange(k * len(y), device=idy.device) // k

    return torch.stack([rows, idy.view(-1)], dim=0)


knns = {"torch": knn, "keops": keops_knn}


@torch.jit.ignore
def knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None, loop: bool = False, flow: str = "source_to_target", cosine: bool = False,
        target: Optional[torch.Tensor] = None, batch_target: Optional[torch.Tensor] = None, backend: str = "torch", ) -> torch.Tensor:
    r"""Straightforward modification of PyTorch_geometric's knn_graph method to allow for source/targets."""

    assert flow in ["source_to_target", "target_to_source"]
    if target is None:
        target = x
    if batch_target is None:
        batch_target = batch

    row, col = knns[backend](x, target, k if loop else k + 1, batch, batch_target, cosine=cosine)
    row, col = (col, row) if flow == "source_to_target" else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


class MyDynamicEdgeConv(EdgeConv):
    r"""Straightforward modification of PyTorch_geometric's DynamicEdgeConv layer."""

    def __init__(self, nn, k, aggr="max", **kwargs):
        super(MyDynamicEdgeConv, self).__init__(nn=nn, aggr=aggr, **kwargs)
        self.k = k

    def forward(self, x, batch=None):
        """"""
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow, backend="keops")
        return super(MyDynamicEdgeConv, self).forward(x, edge_index)

    def __repr__(self):
        return "{}(nn={}, k={})".format(self.__class__.__name__, self.nn, self.k)


DEConv = {"torch": DynamicEdgeConv, "keops": MyDynamicEdgeConv}


# Dynamic Graph CNNs ===========================================================
# Adapted from the PyTorch_geometric gallery to get a close fit to the original paper.


def MLP(channels, batch_norm=True):
    """Multi-layer perceptron, with ReLU non-linearities and batch normalization."""
    return Seq(*[Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]) if batch_norm else nn.Identity(), LeakyReLU(negative_slope=0.2), ) for i in range(1, len(channels))])


class DGCNN_seg(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, k=40, aggr="max", backend="keops"):
        super(DGCNN_seg, self).__init__()

        self.name = "DGCNN_seg_" + backend
        self.I, self.O = (in_channels + 3, out_channels,)  # Add coordinates to input channels
        self.n_layers = n_layers

        self.transform_1 = DEConv[backend](MLP([2 * 3, 64, 128]), k, aggr)
        self.transform_2 = MLP([128, 1024])
        self.transform_3 = MLP([1024, 512, 256], batch_norm=False)
        self.transform_4 = Lin(256, 3 * 3)

        self.conv_layers = nn.ModuleList(
            [DEConv[backend](MLP([2 * self.I, self.O, self.O]), k, aggr)] + [DEConv[backend](MLP([2 * self.O, self.O, self.O]), k, aggr) for i in range(n_layers - 1)])

        self.linear_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)) for i in range(n_layers)])

        self.linear_transform = nn.ModuleList([nn.Linear(self.I, self.O)] + [nn.Linear(self.O, self.O) for i in range(n_layers - 1)])

    def forward(self, positions, features, batch_indices):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        pos, feat, batch = positions, features, batch_indices

        # TransformNet:
        x = pos  # Don't use the normals!

        x = self.transform_1(x, batch)  # (N, 3) -> (N, 128)
        x = self.transform_2(x)  # (N, 128) -> (N, 1024)
        x = global_max_pool(x, batch)  # (B, 1024)

        x = self.transform_3(x)  # (B, 256)
        x = self.transform_4(x)  # (B, 3*3)
        x = x[batch]  # (N, 3*3)
        x = x.view(-1, 3, 3)  # (N, 3, 3)

        # Apply the transform:
        x0 = torch.einsum("ni,nij->nj", pos, x)  # (N, 3)

        # Add features to coordinates
        x = torch.cat([x0, feat], dim=-1).contiguous()
        for i in range(self.n_layers):
            x_i = self.conv_layers[i](x, batch)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i
        return x


# Reference PointNet models, from the PyTorch_geometric gallery =========================


class SAModule(torch.nn.Module):
    """Set abstraction module."""

    def __init__(self, ratio, r, nn, max_num_neighbors=64):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x, pos, batch):
        # Subsample with Farthest Point Sampling:
        idx = torch.arange(0, len(pos), device=pos.device)

        # For each "cluster", get the list of (up to 64) neighbors in a ball of radius r:
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_num_neighbors, )

        # Applies the PointNet++ Conv:
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)

        # Return the features and sub-sampled point clouds:
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2_seg(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(PointNet2_seg, self).__init__()
        self.name = "PointNet2"
        self.I, self.O = in_channels, out_channels
        self.radius = args.radius
        self.k = 10000  # We don't restrict the number of points in a patch
        self.n_layers = args.n_layers

        self.layers = nn.ModuleList(
            [SAModule(1.0, self.radius, MLP([self.I + 3, self.O, self.O]), self.k)] + [SAModule(1.0, self.radius, MLP([self.O + 3, self.O, self.O]), self.k) for _ in
                                                                                       range(self.n_layers - 1)])
        self.linear_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)) for _ in range(self.n_layers)])
        self.linear_transform = nn.ModuleList([nn.Linear(self.I, self.O)] + [nn.Linear(self.O, self.O) for _ in range(self.n_layers - 1)])

    def forward(self, positions, features, batch_indices):
        x = (features, positions, batch_indices)
        for i, layer in enumerate(self.layers):
            x_i, pos, b_ind = layer(*x)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x[0])
            x = x + x_i
            x = (x, pos, b_ind)
        return x[0]


## TangentConv benchmark segmentation
class dMaSIFConv_seg(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, radius=9.0):
        super(dMaSIFConv_seg, self).__init__()
        self.radius = radius
        self.I, self.O = in_channels, out_channels

        self.layers = nn.ModuleList([dMaSIFConv(self.I, self.O, radius, self.O)] + [dMaSIFConv(self.O, self.O, radius, self.O) for _ in range(n_layers - 1)])
        self.linear_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)) for _ in range(n_layers)])
        self.linear_transform = nn.ModuleList([nn.Linear(self.I, self.O)] + [nn.Linear(self.O, self.O) for _ in range(n_layers - 1)])

    def forward(self, features, points, normals, ranges):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        x = features
        for i, layer in enumerate(self.layers):
            x_i = layer(points, normals, x, ranges)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i
        return x
