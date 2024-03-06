import argparse

from ep_ab.models import get_model
from ep_ab.utils.convert_pdb2npy import load_seperate_structure
from ep_ab.utils.geometry import atoms_to_points
from ep_ab.utils.misc import seed_all
from ep_ab.utils.protein.points import ProteinPairData
from ep_ab.utils.train import *

Tensor, tensor = torch.LongTensor, torch.FloatTensor


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--H_chain', type=str)
    parser.add_argument('--L_chain', type=str)
    parser.add_argument('--ag_chains', type=str)
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['cfg']

    proteins = load_seperate_structure(args.pdb_path, return_map=False, ligand=args.H_chain + args.L_chain, receptor=args.ag_chains)
    l, r = proteins['ligand'], proteins['receptor']
    atomxyz_ligand, atomtypes_ligand, resxyz_ligand, restypes_ligand = tensor(l["atom_xyz"]), tensor(l["atom_types"]), tensor(l["res_xyz"]), Tensor(l["res_types"])
    batch_atoms_ligand = torch.zeros(len(atomxyz_ligand)).long().to(atomxyz_ligand.device)
    pts_ligand, norms_ligand, _ = atoms_to_points(atomxyz_ligand, batch_atoms_ligand, atomtypes=atomtypes_ligand, resolution=cfg.dataset.surface.resolution,
                                                  sup_sampling=cfg.dataset.surface.sup_sampling, distance=cfg.dataset.surface.distance)
    atomxyz_receptor, atomtypes_receptor, resxyz_receptor, restypes_receptor = tensor(r["atom_xyz"]), tensor(r["atom_types"]), tensor(r["res_xyz"]), Tensor(r["res_types"])
    batch_atoms_receptor = torch.zeros(len(atomxyz_receptor)).long().to(atomxyz_receptor.device)
    pts_receptor, norms_receptor, _ = atoms_to_points(atomxyz_receptor, batch_atoms_receptor, atomtypes=atomtypes_receptor, resolution=cfg.dataset.surface.resolution,
                                                      sup_sampling=cfg.dataset.surface.sup_sampling, distance=cfg.dataset.surface.distance)
    intf_ligand = (torch.cdist(pts_ligand, pts_receptor) < cfg.dataset.surface.intf_cutoff).sum(dim=1) > 0
    intf_receptor = (torch.cdist(pts_receptor, pts_ligand) < cfg.dataset.surface.intf_cutoff).sum(dim=1) > 0
    data = ProteinPairData(xyz_ligand=pts_ligand, normals_ligand=norms_ligand, atomxyz_ligand=atomxyz_ligand, atomtypes_ligand=atomtypes_ligand, resxyz_ligand=resxyz_ligand,
                           restypes_ligand=restypes_ligand, xyz_receptor=pts_receptor, normals_receptor=norms_receptor, atomxyz_receptor=atomxyz_receptor,
                           atomtypes_receptor=atomtypes_receptor, resxyz_receptor=resxyz_receptor, restypes_receptor=restypes_receptor, intf_ligand=intf_ligand,
                           intf_receptor=intf_receptor)

    # Load configs
    seed_all(cfg.train.seed)
    model = get_model(cfg.model).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    with torch.no_grad():
        data = recursive_to(data, args.device)
        P_receptor = {'resxyz': data['resxyz_receptor'], 'restypes': data['restypes_receptor'], 'batch_res': data['resxyz_receptor_batch'], 'xyz': data['xyz_receptor'],
                      'normals': data['normals_receptor'], 'batch': data['xyz_receptor_batch']}
        P_ligand = {'resxyz': data['resxyz_ligand'], 'restypes': data['restypes_ligand'], 'batch_res': data['resxyz_ligand_batch'], 'xyz': data['xyz_ligand'],
                    'normals': data['normals_ligand'], 'batch': data['xyz_ligand_batch']}
        patch_feat_receptor, patch_idx_receptor, patch_xyz_receptor, patch_mask_receptor, patch_xyz_dense_receptor, group_idx_receptor = model.point_forward(P_receptor)
        patch_feat_ligand, _, patch_xyz_ligand, patch_mask_ligand, _, _ = model.point_forward(P_ligand)
        patch_feat_receptor = model.patch_forward(patch_feat_receptor, patch_xyz_receptor, patch_feat_=patch_feat_ligand,
                                                  patch_mask=patch_mask_receptor.unsqueeze(1).unsqueeze(2) if patch_mask_receptor is not None else None,
                                                  patch_mask_=patch_mask_ligand.unsqueeze(1).unsqueeze(2) if patch_mask_ligand is not None else None)
        pred = model.classifier(patch_feat_receptor).squeeze()
    print(f'Prediction for patches: {pred}')
