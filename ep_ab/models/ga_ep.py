import torch
import torch.nn as nn
import torch.nn.functional as F

from ep_ab.modules.common.geometry import construct_3d_basis
from ep_ab.modules.encoders.ga import GAEncoder
from ep_ab.modules.encoders.pair import PairEmbedding
from ep_ab.modules.encoders.residue import ResidueEmbedding
from ep_ab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

resolution_to_num_atoms = {'backbone+CB': 5, 'full': max_num_heavyatoms}


@register_model('ga')
class GraphTransformer_Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_sasa = cfg.use_sasa
        self.use_plm = cfg.use_plm
        dim = 1280 if self.use_plm else cfg.res_feat_dim
        self.pos_weight = cfg.loss_weight.pos

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(dim, num_atoms, use_sasa=cfg.use_sasa)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)
        self.encoder = GAEncoder(dim, cfg.pair_feat_dim, cfg.num_layers)
        self.classifier = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1), nn.Sigmoid())

        # Pretrain
        self.rde = None
        if cfg.checkpoint.path:
            self.ckpt_type = cfg.checkpoint.type
            print(f'Loading {cfg.checkpoint.type} from {cfg.checkpoint.path}')
            ckpt = torch.load(cfg.checkpoint.path, map_location='cpu')
            if self.ckpt_type == 'RDE':
                from ep_ab.models.rde import CircularSplineRotamerDensityEstimator
                self.rde = CircularSplineRotamerDensityEstimator(ckpt['cfg'].model)
            # https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict
            self.rde.load_state_dict(ckpt['model'], strict=False)   # ignore unmatched keys
            for p in self.rde.parameters():
                p.requires_grad_(False)
            self.single_fusion = nn.Sequential(nn.Linear(dim + ckpt['cfg'].model.blocks.node_feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def _encode_pretrain(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        if self.ckpt_type == 'RDE':
            batch['chi_corrupt'] = batch['chi']
            batch['chi_masked_flag'] = torch.zeros_like(batch['aa'])

        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch, remove_structure=False, remove_sequence=False):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
        """
        # This is used throughout embedding and encoding layers to avoid data leakage.
        context_mask = batch['mask_heavyatom'][:, :, BBHeavyAtom.CA]
        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_heavyatom'], mask_atoms=batch['mask_heavyatom'],
                                      fragment_type=batch['fragment_type'], structure_mask=structure_mask, sequence_mask=sequence_mask, sasa=batch['sasa'] if self.use_sasa else None,)
        pair_feat = self.pair_embed(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_heavyatom'], mask_atoms=batch['mask_heavyatom'],
                                    structure_mask=structure_mask, sequence_mask=sequence_mask, )
        if self.use_plm:
            res_feat += batch['plm_feature']

        if self.rde is not None:
            x_pret = self._encode_pretrain(batch)
            res_feat = self.single_fusion(torch.cat([res_feat, x_pret], dim=-1))

        R = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N], )
        t = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        res_feat = self.encoder(R, t, res_feat, pair_feat, mask=batch['mask'])
        return res_feat

    def forward(self, batch):
        res_feat = self.encode(batch)
        ep_pred = self.classifier(res_feat).squeeze(-1)
        loss_ep = F.binary_cross_entropy(ep_pred, batch['epitope'].float(), reduction='none')
        weight = torch.ones_like(loss_ep)
        weight[batch['epitope'] > 0] = self.pos_weight
        loss_ep_weighted = (loss_ep * weight * batch['mask']).sum() / (batch['mask'].sum() + 1e-8)
        loss_dict = {'ep': loss_ep_weighted}
        out_dict = {'ep_pred': ep_pred, 'ep_true': batch['epitope'], }

        return loss_dict, out_dict
