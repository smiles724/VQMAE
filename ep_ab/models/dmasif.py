import torch
import torch.nn as nn
from pykeops.torch import LazyTensor

from ep_ab.models.benchmark_models import dMaSIFConv_seg
from ep_ab.utils.geometry import (curvatures, )
from ep_ab.utils.helper import soft_dimension, diagonal_ranges


def knn_nodes(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)
    return idx, dists

# -----------------------------
# Atom-level


class Atom_embedding_MP(nn.Module):
    def __init__(self, cfg):
        super(Atom_embedding_MP, self).__init__()
        self.D = cfg.atom_dims
        self.k = 16   # KNN
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * self.D + 1, 2 * self.D + 1), nn.LeakyReLU(negative_slope=0.2), nn.Linear(2 * self.D + 1, self.D), ) for i in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(2, self.D) for _ in range(self.n_layers)])  # umber of channels in input to be divisible by num_groups (e.g., 2)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_nodes(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat([point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1)  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb


class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, cfg):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.k = 17
        self.D = cfg.atom_dims
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * self.D + 1, 2 * self.D + 1), nn.LeakyReLU(negative_slope=0.2), nn.Linear(2 * self.D + 1, self.D), ) for _ in range(self.n_layers)])

        self.norm = nn.ModuleList([nn.GroupNorm(2, self.D) for i in range(self.n_layers)])  # default num_groups: 2
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomfeat, x_batch, y_batch):
        idx, dists = knn_nodes(x, y, x_batch, y_batch, k=self.k)  # (N, 9, 7)
        idx = idx[:, 1:]  # Remove xxx
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomfeat.shape[0]

        out = y_atomfeat
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat([out[:, None, :].repeat(1, k, 1), features], dim=-1)  # (N, 8, 13)

            messages = self.mlp[i](features)  # (N, 8, 6)
            messages = messages.sum(1)  # (N, 6)
            out = out + self.relu(self.norm[i](messages))
        return out


class AtomNet_MP(nn.Module):
    def __init__(self, cfg, max_aa_types=22):
        super(AtomNet_MP, self).__init__()
        self.cfg = cfg
        self.transform_types = nn.Sequential(nn.Linear(6, cfg.atom_dims), nn.LeakyReLU(negative_slope=0.2), nn.Linear(cfg.atom_dims, cfg.atom_dims), )
        self.res_embed = nn.Embedding(max_aa_types, cfg.atom_dims)
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=cfg.atom_dims, padding_idx=0, )
        self.embed = Atom_embedding_MP(cfg)
        self.atom_atom = Atom_Atom_embedding_MP(cfg)

    def forward(self, xyz, atom_xyz, atomtypes, batch, batch_atom, xyz_=None, batch_=None, atom_restypes=None):
        """
        Args:
            xyz: coordinates of a surface point cloud.
            atom_xyz: atomic coordinates
            atomtypes: atom types.
            atom_restypes: residue types of each atom.
            batch: batch indicator of xyz.
            batch_atom: batch indicator of atom_xyz.
            xyz_: coordinates of another surface point cloud.
            batch_: batch indicator of xyz_.
        """
        atom_feat = self.transform_types(atomtypes)
        if atom_restypes is not None:
            atom_feat += self.res_embed(atom_restypes)

        atom_feat = self.atom_atom(atom_xyz, atom_xyz, atom_feat, batch_atom, batch_atom)   # interactions between atoms
        point_feat = self.embed(xyz, atom_xyz, atom_feat, batch, batch_atom)                # interactions between points and atoms
        if xyz_ is not None:
            point_feat_ = self.embed(xyz_, atom_xyz, atom_feat, batch_, batch_atom)         # interactions between another points and atoms
            return point_feat, point_feat_
        return point_feat, None

# -----------------------------
# Residue-level


class Res_point_embedding_MP(nn.Module):
    def __init__(self, cfg):
        super(Res_point_embedding_MP, self).__init__()
        self.D = cfg.res_dims
        self.k = cfg.nearest_neighbors.res_point  # KNN
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * self.D + 1, 2 * self.D + 1), nn.LeakyReLU(negative_slope=0.2), nn.Linear(2 * self.D + 1, self.D), ) for _ in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(2, self.D) for _ in range(self.n_layers)])  # umber of channels in input to be divisible by num_groups (e.g., 2)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_restypes, x_batch, y_batch):
        idx, dists = knn_nodes(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_restypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_restypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat([point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1)  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))
        return point_emb


class Res_Res_embedding_MP(nn.Module):
    def __init__(self, cfg):
        super(Res_Res_embedding_MP, self).__init__()
        self.k = cfg.nearest_neighbors.res_res + 1    # nearest_neighbors + 1
        self.D = cfg.res_dims
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * self.D + 1, 2 * self.D + 1), nn.LeakyReLU(negative_slope=0.2), nn.Linear(2 * self.D + 1, self.D), ) for _ in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(2, self.D) for _ in range(self.n_layers)])  # default num_groups: 2
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_restypes, x_batch, y_batch):
        idx, dists = knn_nodes(x, y, x_batch, y_batch, k=self.k)  # (N, 9, 7)
        idx = idx[:, 1:]  # Remove xxx
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_restypes.shape[0]

        out = y_restypes
        _, num_dims = out.size()
        for i in range(self.n_layers):
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat([out[:, None, :].repeat(1, k, 1), features], dim=-1)  # (N, 8, 2D + 1)

            messages = self.mlp[i](features)  # (N, 8, D)
            messages = messages.sum(1)  # (N, D)
            out = out + self.relu(self.norm[i](messages))
        return out


class ResNet_MP(nn.Module):
    def __init__(self, cfg, max_aa_types=22):
        super(ResNet_MP, self).__init__()
        self.cfg = cfg
        self.res_embed = nn.Embedding(max_aa_types, cfg.res_dims)
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=cfg.res_dims, padding_idx=0, )
        self.res_res = Res_Res_embedding_MP(cfg)
        self.embed = Res_point_embedding_MP(cfg)

    def forward(self, xyz, resxyz, restypes, batch, batch_res, xyz_=None, batch_=None):
        """
        Args:
            xyz: coordinates of a surface point cloud.
            resxyz: residue coordinates.
            restypes: residue types of each residue.
            batch: batch indicator of xyz.
            batch_res: batch indicator of resxyz.
            xyz_: coordinates of another surface point cloud.
            batch_: batch indicator of xyz_.
        """
        res_feat = self.res_embed(restypes)
        res_feat = self.res_res(resxyz, resxyz, res_feat, batch_res, batch_res)  # interactions between residues
        pts_embed = self.embed(xyz, resxyz, res_feat, batch, batch_res)  # interactions between points and residues
        if xyz_ is not None:
            pts_embed_ = self.embed(xyz_, resxyz, res_feat, batch_, batch_res)
            return pts_embed, pts_embed_
        return pts_embed, None


def project_iface_labels(P, threshold=2.0):
    queries, batch_queries = P["xyz"], P["batch"]
    source, batch_source = P["mesh_xyz"], P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i, y_j = LazyTensor(queries[:, None, :]), LazyTensor(source[None, :, :])  # (N, 1, D), (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (D_ij.min(dim=1).view(-1, 1) < threshold).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    query_labels = labels[nn_i] * nn_dist_i
    P["labels"] = query_labels


class dMaSIF(nn.Module):
    def __init__(self, cfg):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = [1.0, 2.0, 3.0, 5.0, 10.0]  # cfg.curvature_scales, scales to compute the geometric features (mean and Gauss curvatures)
        self.cfg = cfg

        I = cfg.res_dims + 2 * len(self.curvature_scales)
        O = cfg.ori_dim
        E = cfg.hidden_dims

        # Computes chemical features
        if cfg.resolution == 'atom':
            self.nodenet = AtomNet_MP(cfg)
        else:
            self.nodenet = ResNet_MP(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

        self.orientation_scores = nn.Sequential(nn.Linear(I, O), nn.LeakyReLU(negative_slope=0.2), nn.Linear(O, 1), )  # Post-processing, without batch norm
        self.conv = dMaSIFConv_seg(in_channels=I, out_channels=E, n_layers=cfg.n_layers, radius=cfg.radius, )          # Segmentation network

        if cfg.site:     # Post-processing, without batch norm:
            self.net_out = nn.Sequential(nn.Linear(E, E), nn.LeakyReLU(negative_slope=0.2), nn.Linear(E, E), nn.LeakyReLU(negative_slope=0.2), nn.Linear(E, 1), )

    def forward(self, P, return_r_value=False):
        """Embeds all points of a protein in a high-dimensional vector space."""
        if "xyz_" not in P:
            P["xyz_"], P["batch_"] = None, None
        P_curv = curvatures(P["xyz"], normals=P["normals"], scales=self.curvature_scales, batch=P["batch"], )    # Estimate the curvatures using the estimated normals
        if self.cfg.resolution == 'atom':
            chemfeats, chemfeats_ = self.nodenet(P["xyz"], P["atomxyz"], P["atomtypes"], P["batch"], P["batch_atom"], P['xyz_'], P['batch_'])   # Compute chemical features
        else:
            chemfeats, chemfeats_ = self.nodenet(P["xyz"], P["resxyz"], P["restypes"], P["batch"], P["batch_res"], P['xyz_'], P['batch_'])   # Compute chemical features
        features = self.dropout(torch.cat([P_curv, chemfeats], dim=1).contiguous())     # Concatenate our features
        if P["xyz_"] is not None:
            P_curv_ = curvatures(P["xyz_"], normals=P["normals_"], scales=self.curvature_scales, batch=P["batch_"], )
            features_ = self.dropout(torch.cat([P_curv_, chemfeats_], dim=1).contiguous())

        torch.cuda.synchronize(device=features[0].device)   # wait for GPU

        # TODO: change dMaSIF to PointNet ++
        ranges = diagonal_ranges(P["batch"])
        nuv = self.conv.load_mesh(P["xyz"], normals=P["normals"], weights=self.orientation_scores(features), ranges=ranges, )
        output = self.conv(features, P["xyz"], nuv, ranges)
        if P["xyz_"] is not None:  # two surfaces: ligand and receptor surfaces
            ranges_ = diagonal_ranges(P["batch_"])  # KeOps support for heterogeneous batch processing
            nuv_ = self.conv.load_mesh(P["xyz_"], normals=P["normals_"], weights=self.orientation_scores(features_), ranges=ranges_, )
            output_ = self.conv(features_, P["xyz_"], nuv_, ranges_)

        torch.cuda.synchronize(device=features[0].device)
        if self.cfg.site:
            P["iface_preds"] = self.net_out(output)
        if return_r_value:                 # Monitor the approximate rank of our representations
            R_values = {"input": soft_dimension(features), "conv": soft_dimension(output)}
            return {"output": output, "R_values": R_values}
        if P["xyz_"] is not None:
            return output, output_
        return output
