import chamfer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps
from torch_geometric.utils import to_dense_batch

from src.models.dmasif import dMaSIF
from src.utils.train import focal_loss
from ._base import register_model
from .surfformer_v1 import square_distance, index_points, Encoder


####################################################
## Discrete VAE from PointBERT
## https://github.com/lulutang0608/Point-BERT/blob/master/models/dvae.py
####################################################


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistance(nn.Module):
    """ Chamder Distance """
    def __init__(self, norm='L2', ignore_zeros=False):
        super().__init__()
        self.norm = norm
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        if self.norm == 'L2':
            return torch.mean(dist1) + torch.mean(dist2)
        elif self.norm == 'L1':
            return (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2


class FoldingNet(nn.Module):
    """ FoldingNet: https://arxiv.org/abs/1712.07262 """
    def __init__(self, d_model, d_ff, num_fine, grid_size=2):
        super().__init__()
        assert num_fine % (grid_size ** 2) == 0
        self.num_fine = num_fine
        self.grid_size = grid_size
        self.num_coarse = self.num_fine // (self.grid_size ** 2)    # num_fine = num_coarse * grid_size^2

        self.xyz_mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(inplace=True), nn.Linear(d_ff, d_ff), nn.ReLU(inplace=True), nn.Linear(d_ff, 3 * self.num_coarse))
        self.final_conv = nn.Sequential(nn.Conv1d(d_model + 5, d_ff, 1), nn.BatchNorm1d(d_ff), nn.ReLU(inplace=True), nn.Conv1d(d_ff, d_ff, 1), nn.BatchNorm1d(d_ff),
                                        nn.ReLU(inplace=True), nn.Conv1d(d_ff, 3, 1))
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # (1, 2, grid_size^2)

    def forward(self, patch_feat):
        B, n_patches, D = patch_feat.shape
        patch_feat = patch_feat.reshape(B * n_patches, D)
        coarse = self.xyz_mlp(patch_feat).reshape(B * n_patches, self.num_coarse, 3)   # coarse: (B * n_patches, num_coarse, 3)

        pts_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)           # pts_feat: (B * n_patches, num_coarse, grid_size^2, 3)
        pts_feat = pts_feat.reshape(B * n_patches, self.num_fine, 3).transpose(2, 1)     # pts_feat: (B * n_patches, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B * n_patches, -1, self.num_coarse, -1)  # seed: (B * n_patches, 2, num_coarse, grid_size^2)
        seed = seed.reshape(B * n_patches, -1, self.num_fine).to(patch_feat.device)          # seed: (B * n_patches, 2, num_fine)

        patch_feat = patch_feat.unsqueeze(2).expand(-1, -1, self.num_fine)  # (B * n_patches, D, num_fine)
        feat = torch.cat([patch_feat, seed, pts_feat], dim=1)  # (B * n_patches, D + 5, num_fine)

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)
        center = center.reshape(B * n_patches, self.num_fine, 3).transpose(2, 1)  # (B * n_patches, 3, num_fine)

        fine = self.final_conv(feat) + center   # (B * n_patches, 3, num_fine)
        fine = fine.reshape(B, n_patches, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(B, n_patches, self.num_coarse, 3)
        return coarse, fine


@register_model('surfformerv2')
class SurfaceTransformerV2(nn.Module):

    def __init__(self, cfg, ):
        super().__init__()
        self.cfg = cfg
        self.patch_ratio = cfg.patch_setup.patch_ratio
        self.n_patches = cfg.patch_setup.n_patches
        self.n_pts_per_patch = cfg.patch_setup.n_pts_per_patch  # max sample number in local region (patch)

        dim = cfg.masif.hidden_dims
        self.surface_encoder = dMaSIF(cfg.masif)    # TODO: use a more efficient point cloud encoder
        self.dropout = nn.Dropout(cfg.transformer.dropout)
        self.blocks = nn.ModuleList([Encoder(dim, dim, cfg.transformer.n_heads, cfg.transformer.dropout,
                                             partner=cfg.partner.input if 'partner' in cfg else 'surface') for _ in range(cfg.transformer.n_layers)])

        # Epitope classification
        self.classifier = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1), nn.Sigmoid())
        if 'partner' in cfg:
            self.loss_type = cfg.loss_setup.type
            self.classifier = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1), nn.Sigmoid())
        else:
            # dVAE
            self.vocab_size = cfg.get('vocab_size', 1000)
            self.mask_ratio = cfg.get('mask_ratio', 0.5)
            self.codebook = nn.Parameter(torch.randn(self.vocab_size, dim))
            self.tokenizer_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, self.vocab_size))
            self.foldingnet = cfg.decoder.foldingnet if 'decoder' in cfg else None
            if self.foldingnet:
                self.decoder = FoldingNet(dim, d_ff=1024, num_fine=self.n_pts_per_patch)
            else:
                self.decoder = nn.Sequential(nn.Conv1d(dim, 3 * self.n_pts_per_patch, 1))
            self.hbond_mlp, self.hphobicity_mlp = [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)) for _ in range(2)]
            self.cdl2 = ChamferDistance().cuda()

    def point_forward(self, P, ):
        feat = self.surface_encoder(P)                                              # feat: (N, D)
        xyz_dense, mask = to_dense_batch(P['xyz'], P['batch'], fill_value=0.0)      # xyz_dense: (B, L, 3), mask: (B, L)
        feat_dense, _ = to_dense_batch(feat, P['batch'], fill_value=0.0)            # feat_dense: (B, L, D)

        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.fps.html
        patch_idx = fps(P['xyz'], P['batch'], ratio=self.patch_ratio)                                         # patch_idx: (B x L)
        patch_xyz, patch_mask = to_dense_batch(P['xyz'][patch_idx], P['batch'][patch_idx], fill_value=0.0)    # patch_xyz: (B, npatch, 3), patch_mask: (B, npatch)

        dists = square_distance(xyz_dense, patch_xyz)                                                             # dists: (B, L, npatch)
        dists[~mask] = 1e10
        _, group_idx = dists.transpose(1, 2).contiguous().topk(self.n_pts_per_patch, largest=False)               # group_idx: (B, npatch, n_pts)
        grouped_feat = index_points(feat_dense, group_idx)                                                        # grouped_feat: (B, npatch, n_pts, D)
        patch_feat = grouped_feat.max(-2)[0]                                                                      # patch_feat: (B, npatch, D)
        return patch_feat, patch_idx, patch_xyz, patch_mask, xyz_dense, group_idx

    def patch_forward(self, patch_feat, patch_xyz, patch_feat_=None, patch_mask=None, patch_mask_=None):
        patch_dist = torch.cdist(patch_xyz, patch_xyz)                                        # patch_dist: (B, npatch, npatch)
        patch_feat = self.dropout(patch_feat)
        patch_feat_ = self.dropout(patch_feat_) if patch_feat_ is not None else None
        for block in self.blocks:
            patch_feat = block(patch_feat, patch_feat_, rel_pos=patch_dist, mask=patch_mask, mask_=patch_mask_)
        return patch_feat

    def _mask_center_rand(self, patch_mask):
        patch_mae_mask = torch.zeros_like(patch_mask)
        for i in range(len(patch_mask)):
            n_mask = int(self.mask_ratio * patch_mask[i].sum())
            if n_mask > 0:
                mask_idx = torch.multinomial(patch_mask[i].float(), num_samples=n_mask, )  # sample from the multinomial distribution
                patch_mae_mask[i, mask_idx] = 1
        return patch_mae_mask.bool()

    def dvae_forward(self, batch, temperature=1., hard=False, return_chem=False):
        P = {'resxyz': batch['resxyz'], 'restypes': batch['restypes'], 'batch_res': batch['resxyz_batch'], 'xyz': batch['xyz'], 'normals': batch['normals'],
             'batch': batch['xyz_batch']}
        patch_feat, patch_idx, patch_xyz, patch_mask, xyz_dense, group_idx = self.point_forward(P)
        logits = self.tokenizer_mlp(patch_feat)                                                                     # logits: (B, npatch, ntoken)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)                                  # soft_one_hot: (B, npatch, ntoken)
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)                                  # sampled: (B, npatch, D)
        patch_mae_mask = self._mask_center_rand(patch_mask)
        sampled[~patch_mae_mask] = patch_feat[~patch_mae_mask]

        # reconstruction loss (MAE)
        feat_decoder = self.patch_forward(sampled, patch_xyz, patch_mask=patch_mask.unsqueeze(1).unsqueeze(2), patch_mask_=patch_mask.unsqueeze(1).unsqueeze(2))
        B, npatch, _ = feat_decoder.shape
        grouped_xyz = index_points(xyz_dense, group_idx)  # grouped_xyz: (B, npatch, n_pts, 3)
        grouped_xyz_norm = grouped_xyz - patch_xyz.unsqueeze(-2)  # normalization to the patch center
        if self.foldingnet:
            coarse, fine = self.decoder(feat_decoder)                                                                   # coarse: (B, npatch, C, 3), fine: (B, npatch, n_pts, 3)
            with torch.no_grad():
                whole_fine = fine + patch_xyz.unsqueeze(-2)
            coarse, fine, grouped_xyz_norm = coarse[patch_mae_mask].contiguous(), fine[patch_mae_mask].contiguous(), grouped_xyz_norm[patch_mae_mask].contiguous()
            loss_recon = self.cdl2(coarse, grouped_xyz_norm) + self.cdl2(fine, grouped_xyz_norm)   # (B x npatch, C, 3) https://www.fwilliams.info/point-cloud-utils/sections/shape_metrics/
        else:
            fine = self.decoder(feat_decoder.transpose(1, 2)).transpose(1, 2).reshape(B, npatch, -1, 3)
            with torch.no_grad():
                whole_fine = fine + patch_xyz.unsqueeze(-2)
            fine, grouped_xyz_norm = fine[patch_mae_mask].contiguous(), grouped_xyz_norm[patch_mae_mask].contiguous()
            loss_recon = self.cdl2(fine, grouped_xyz_norm)   # (B x npatch, n_pts, 3)

        # KL loss
        mean_softmax = F.softmax(logits, dim=-1).mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.vocab_size], device=patch_feat.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean', log_target=True)

        # chem/bio property loss
        hbond_dense, _ = to_dense_batch(batch['hbond'], P['batch'], fill_value=0)  # hbond_dense: (B, L)
        hbond_soft = index_points(hbond_dense.unsqueeze(-1), group_idx).float().mean(-2)  # soft label: (B, npatch, n_pts, 1) -> (B, npatch, 1)
        hphobicity_dense, _ = to_dense_batch(batch['hphobicity'], P['batch'], fill_value=0)
        hphobicity_soft = index_points(hphobicity_dense.unsqueeze(-1), group_idx).float().mean(-2)

        hbond_pred = self.hbond_mlp(feat_decoder)
        hphobicity_pred = self.hphobicity_mlp(feat_decoder)
        if return_chem:   # for visualization
            grouped_xyz = index_points(xyz_dense, group_idx)  # grouped_xyz: (B, npatch, n_pts, 3)
            grouped_xyz = grouped_xyz[patch_mask]
            return hphobicity_pred[patch_mask], hbond_pred[patch_mask], hphobicity_soft[patch_mask], hbond_soft[patch_mask], P['batch'][patch_idx], grouped_xyz

        loss_hbond = F.mse_loss(hbond_pred[patch_mae_mask], hbond_soft[patch_mae_mask])
        loss_hphobicity = F.mse_loss(hphobicity_pred[patch_mae_mask], hphobicity_soft[patch_mae_mask])
        loss_dict = {'recon': loss_recon, 'klv': loss_klv, 'hbond': loss_hbond, 'hphobicity': loss_hphobicity}
        out_dict = {'fine': whole_fine, 'pts': grouped_xyz, 'mask': patch_mask, }
        return loss_dict, out_dict

    def forward(self, batch):
        # receptor
        if self.cfg.masif.resolution == 'atom':
            P_receptor = {'atomxyz': batch['atomxyz_receptor'], 'atomtypes': batch['atomtypes_receptor'], 'batch_atom': batch['atomxyz_receptor_batch'],
                          'xyz': batch['xyz_receptor'], 'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
        else:
            P_receptor = {'resxyz': batch['resxyz_receptor'], 'restypes': batch['restypes_receptor'], 'batch_res': batch['resxyz_receptor_batch'], 'xyz': batch['xyz_receptor'],
                          'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
        patch_feat_receptor, patch_idx_receptor, patch_xyz_receptor, patch_mask_receptor, _, _ = self.point_forward(P_receptor)

        # ligand
        if self.cfg.masif.resolution == 'atom':
            P_ligand = {'atomxyz': batch['atomxyz_ligand'], 'atomtypes': batch['atomtypes_ligand'], 'batch_atom': batch['atomxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}
        else:
            P_ligand = {'resxyz': batch['resxyz_ligand'], 'restypes': batch['restypes_ligand'], 'batch_res': batch['resxyz_ligand_batch'], 'xyz': batch['xyz_ligand'],
                        'normals': batch['normals_ligand'], 'batch': batch['xyz_ligand_batch']}
        patch_feat_ligand, _, _, patch_mask_ligand, _, _ = self.point_forward(P_ligand)

        # Transformer
        patch_feat_receptor = self.patch_forward(patch_feat_receptor, patch_xyz_receptor, patch_feat_=patch_feat_ligand,
                                                 patch_mask=patch_mask_receptor.unsqueeze(1).unsqueeze(2) if patch_mask_receptor is not None else None,
                                                 patch_mask_=patch_mask_ligand.unsqueeze(1).unsqueeze(2) if patch_mask_ligand is not None else None)
        pred_coarse = self.classifier(patch_feat_receptor).squeeze(-1)

        # loss
        pred_coarse = pred_coarse[patch_mask_receptor]
        label_coarse = batch['intf_receptor'][patch_idx_receptor]
        if self.loss_type == 'binary_cross_entropy':
            loss_ep_receptor = F.binary_cross_entropy(pred_coarse, label_coarse.float())
        elif self.loss_type == 'focal_loss':
            loss_ep_receptor = focal_loss(pred_coarse, label_coarse.float(), alpha=0.25, gamma=2.0, reduction='mean')
        loss_dict, out_dict = {'ep': loss_ep_receptor}, {'ep_pred': None, 'ep_pred_coarse': pred_coarse, 'ep_true_coarse': label_coarse,
                                                         'batch_idx': P_receptor['batch'][patch_idx_receptor], 'patch_xyz_coarse': patch_xyz_receptor[patch_mask_receptor]}
        return loss_dict, out_dict
