import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from ep_ab.models.dmasif import dMaSIF
from ep_ab.utils.train import focal_loss
from ._base import register_model


####################################################
## Helper Functions
####################################################

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, feature, coordinates, or label, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


####################################################
## PointGPT
## https://github.com/CGuangyan-BIT/PointGPT/blob/9c56432a7b7dc4a071ef681f3ca6cd0f7babba0f/segmentation/models/gpt2_seg.py
####################################################


def simplied_morton_sorting(patch, num_group):
    """ https://github.com/CGuangyan-BIT/PointGPT/blob/9c56432a7b7dc4a071ef681f3ca6cd0f7babba0f/models/PointGPT.py#L98
    Simplifying the Morton code sorting to iterate and set the nearest patch to the last patch as the next patch, we found this to be more efficient.  """
    batch_size = len(patch)
    dist = torch.cdist(patch, patch)  # (B, K, K)
    dist[:, torch.eye(num_group).bool()] = float("inf")
    idx_base = torch.arange(0, batch_size, device=patch.device) * num_group
    sorted_indices_list = [idx_base]
    dist = dist.view(batch_size, num_group, num_group).view(batch_size * num_group, num_group)
    dist[idx_base] = float("inf")
    dist = dist.view(batch_size, num_group, num_group).transpose(1, 2).contiguous()
    for i in range(num_group - 1):
        dist = dist.view(batch_size * num_group, num_group)
        distances_to_last_batch = dist[sorted_indices_list[-1]]
        closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
        closest_point_idx = closest_point_idx + idx_base
        sorted_indices_list.append(closest_point_idx)
        dist = dist.view(batch_size, num_group, num_group).transpose(1, 2).contiguous().view(batch_size * num_group, num_group)
        dist[closest_point_idx] = float("inf")
        dist = dist.view(batch_size, num_group, num_group).transpose(1, 2).contiguous()

    sorted_indices_list = [idx - idx_base for idx in sorted_indices_list]  # transfer from batch index to sample index
    sorted_indices = torch.stack(sorted_indices_list, dim=-1)  # (B, K)
    return sorted_indices


####################################################
## Vanilla Transformer
## https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
## https://github.com/hyunwoongko/transformer
####################################################

def PositionalEncoding(x, pos_idx):
    B, L, D = x.shape
    pe = torch.zeros_like(x)
    div_term = torch.exp(torch.arange(0, D, 2).float() * -(math.log(10000.0) / D)).to(x.device)

    pe[..., 0::2] = torch.sin(pos_idx.unsqueeze(-1) * div_term)
    pe[..., 1::2] = torch.cos(pos_idx.unsqueeze(-1) * div_term)
    return x + pe[:, :x.size(1)]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_length=None, partner=None):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value
        self.attn = None

        # Linear layers for transforming inputs
        self.partner = partner
        assert partner in [None, 'surface', 'esm'], ValueError('Choose correct partner input format!')
        if partner == 'esm':
            self.W_k, self.W_v = [nn.Linear(d_model, d_model * seq_length) for _ in range(2)]
        else:
            self.W_k, self.W_v = [nn.Linear(d_model, d_model) for _ in range(2)]      # Key/Value transformation
        self.W_q, self.W_o = [nn.Linear(d_model, d_model) for _ in range(2)]          # Query/Output transformation
        self.W_rpe = nn.Sequential(nn.Conv2d(1, num_heads * 2, kernel_size=1, stride=1), nn.ReLU(), nn.Conv2d(num_heads * 2, num_heads, kernel_size=1, stride=1))

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, rel_pos=None, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))                                    # (B, L, D) -> (B, H, L, D/H)
        if self.partner == 'esm':
            batch_size, d_model = K.size()
            K = self.split_heads(self.W_k(Q).view(batch_size, -1, d_model))  # (B, D * L) -> (B, L, D) -> (B, H, L, D/H)
            V = self.split_heads(self.W_v(Q).view(batch_size, -1, d_model))  # (B, D * L) -> (B, L, D) -> (B, H, L, D/H)
        else:
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention, calculate attention scores
        rpe = self.W_rpe(rel_pos.unsqueeze(1)) if rel_pos is not None else 0         # (B, 1, L, L)  -> (B, H, L, L)
        attn_scores = (torch.matmul(Q, K.transpose(-2, -1)) + rpe) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        self.attn = attn_scores   # for visualization

        # Obtain attention probabilities by softmax and get the final output
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        # Combine heads and apply output transformation
        batch_size, _, seq_length, d_k = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.W_o(attn_output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, seq_length=None, partner='surface'):
        super(Encoder, self).__init__()
        " decoder_only https://medium.com/@ManishChablani/gpt-and-other-llms-are-they-decoder-only-or-encoder-decoder-models-1bdaf23a256a "
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, seq_length, partner)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_=None, rel_pos=None, mask=None, mask_=None):
        attn_output = self.self_attn(x, x, x, rel_pos, mask)
        x = self.norm1(x + self.dropout(attn_output))
        if x_ is None:
            x_ = x
        attn_output = self.cross_attn(x, x_, x_, None, mask_)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


@register_model('surfformerv1')
class SurfaceTransformerV1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_setup = cfg.loss_setup
        self.loss_type = cfg.loss_setup.get('type', 'binary_cross_entropy')

        self.n_patches = cfg.patch_setup.n_patches
        self.n_pts_per_patch = cfg.patch_setup.n_pts_per_patch  # max sample number in local region (patch)

        dim = cfg.masif.hidden_dims
        self.ape = cfg.transformer.ape
        self.surface_encoder = dMaSIF(cfg.masif)
        self.dropout = nn.Dropout(cfg.transformer.dropout)
        self.blocks = nn.ModuleList([Encoder(dim, dim, cfg.transformer.n_heads, cfg.transformer.dropout, partner=cfg.partner.input) for _ in range(cfg.transformer.n_layers)])
        self.classifier = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1), nn.Sigmoid())

    def point_forward(self, P, ):
        feat = self.surface_encoder(P)                                              # feat: (N, D)
        xyz_dense, mask = to_dense_batch(P['xyz'], P['batch'], fill_value=0.0)      # xyz_dense: (B, L, 3), mask: (B, L)
        feat_dense, _ = to_dense_batch(feat, P['batch'], fill_value=0.0)            # feat_dense: (B, L, D)
        patch_idx = farthest_point_sample(xyz_dense, self.n_patches, mask)          # patch_idx: (B, npatch)
        patch_xyz = index_points(xyz_dense, patch_idx)                              # patch_xyz: (B, npatch, 3)
        dists = square_distance(xyz_dense, patch_xyz)                               # B x L x npatch
        dists[~mask] = 1e10
        _, group_idx = dists.transpose(1, 2).contiguous().topk(self.n_pts_per_patch, largest=False)  # B x npatch x n_pts
        grouped_feat = index_points(feat_dense, group_idx)                                    # grouped_feat: (B, npatch, n_pts, D)
        patch_feat = grouped_feat.max(-2)[0]                                                  # patch_feat: (B, npatch, D)

        if self.loss_setup.fine > 0:
            ref_idx = dists.topk(1, largest=False)[1].squeeze()                                                 # B x L
            ref_feat = index_points(feat_dense, ref_idx)[mask]                       # (B, L, D)
            feat = torch.max(ref_feat, feat)
        return feat, patch_feat, patch_idx, patch_xyz, group_idx

    def patch_forward(self, patch_feat, patch_xyz, patch_feat_=None, patch_pos=None, patch_pos_=None):
        if patch_pos is not None:
            patch_feat = PositionalEncoding(patch_feat, patch_pos)
        if patch_pos_ is not None:
            patch_feat_ = PositionalEncoding(patch_feat_, patch_pos_)
        patch_feat = self.dropout(patch_feat)
        patch_feat_ = self.dropout(patch_feat_) if patch_feat_ is not None else None
        patch_dist = torch.cdist(patch_xyz, patch_xyz)                                        # patch_dist: (B, npatch, npatch)

        for block in self.blocks:
            patch_feat = block(patch_feat, patch_feat_, rel_pos=patch_dist)
        return patch_feat

    def forward(self, batch):     # TODO: use PLM as initial features
        # receptor
        assert batch['min_num_nodes'] >= self.n_patches, f'Surface has {batch["min_num_nodes"]} points than {self.n_patches}, please decrease the patch number.'
        if self.cfg.masif.resolution == 'atom':
            P_receptor = {'atomxyz': batch['atomxyz_receptor'], 'atomtypes': batch['atomtypes_receptor'], 'batch_atom': batch['atomxyz_receptor_batch'],
                          'xyz': batch['xyz_receptor'], 'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
        else:
            P_receptor = {'resxyz': batch['resxyz_receptor'], 'restypes': batch['restypes_receptor'], 'batch_res': batch['resxyz_receptor_batch'], 'xyz': batch['xyz_receptor'],
                          'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
        feat_receptor, patch_feat_receptor, patch_idx_receptor, patch_xyz_receptor, group_idx_receptor = self.point_forward(P_receptor)
        patch_pos_receptor = simplied_morton_sorting(patch_xyz_receptor, num_group=self.n_patches) if self.ape else None               # pos_idx_r: (B, npatch)

        # ligand
        if self.cfg.masif.resolution == 'atom':
            P_ligand = {'atomxyz': batch['atomxyz_ligand'], 'atomtypes': batch['atomtypes_ligand'], 'batch_atom': batch['atomxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}
        else:
            P_ligand = {'resxyz': batch['resxyz_ligand'], 'restypes': batch['restypes_ligand'], 'batch_res': batch['resxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}
        _, patch_feat_ligand, _, patch_xyz_ligand, _ = self.point_forward(P_ligand)
        patch_pos_ligand = simplied_morton_sorting(patch_xyz_ligand, num_group=self.n_patches) if self.ape else None

        # Transformer
        patch_feat_receptor = self.patch_forward(patch_feat_receptor, patch_xyz_receptor, patch_feat_ligand, patch_pos=patch_pos_receptor, patch_pos_=patch_pos_ligand)
        pred_coarse = self.classifier(patch_feat_receptor).squeeze(-1)

        # loss
        intf_receptor_dense, _ = to_dense_batch(batch['intf_receptor'], P_receptor['batch'], fill_value=0)  # intf_receptor_dense: (B, L)
        label_coarse = index_points(intf_receptor_dense.unsqueeze(-1), patch_idx_receptor).squeeze()
        if self.loss_type == 'binary_cross_entropy':
            if self.loss_setup.soft:
                label_coarse_soft = index_points(intf_receptor_dense.unsqueeze(-1), group_idx_receptor).squeeze().float().mean(-1)  # soft label: (B, npatch, n_pts)
                loss_ep_receptor = F.binary_cross_entropy(pred_coarse, label_coarse_soft.float()) * self.loss_setup.coarse
            else:
                loss_ep_receptor = F.binary_cross_entropy(pred_coarse, label_coarse.float()) * self.loss_setup.coarse

            if self.loss_setup.fine > 0:
                label_fine = batch['intf_receptor']  # (N, )
                pred_fine = self.classifier(feat_receptor).squeeze(-1)
                loss_ep_receptor += F.binary_cross_entropy(pred_fine, label_fine.float())
        elif self.loss_type == 'focal_loss':
            loss_ep_receptor = focal_loss(pred_coarse, label_coarse.float(), alpha=0.25, gamma=2.0, reduction='mean')
        loss_dict, out_dict = {'ep': loss_ep_receptor}, {'ep_pred': None, 'ep_pred_coarse': pred_coarse, 'ep_true_coarse': label_coarse}
        return loss_dict, out_dict
