import argparse
import os

from pymol import cmd
from torch_geometric.data import Batch

from ep_ab.models import get_model
from ep_ab.utils.convert_pdb2npy import load_structure, num2ele
from ep_ab.utils.misc import seed_all
from ep_ab.utils.protein.constants import AA
from ep_ab.utils.protein.points import ProteinPairData
from ep_ab.datasets import load_point_cloud_by_file_extension

Tensor, tensor = torch.LongTensor, torch.FloatTensor


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab_pdb_path', type=str)
    parser.add_argument('--ag_pdb_path', type=str)
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['cfg']

    l = load_structure(args.ab_pdb_path)
    atomxyz_ligand, atomtypes_ligand, resxyz_ligand, restypes_ligand = tensor(l["atom_xyz"]), tensor(l["atom_types"]), tensor(l["res_xyz"]), Tensor(l["res_types"])
    cmd.load(args.ab_pdb_path)
    cmd.show_as('surface')
    cmd.save(os.path.join(args.ab_pdb_path.split('.')[0], '.obj'))
    pts_ligand, norms_ligand = load_point_cloud_by_file_extension(os.path.join(args.ab_pdb_path.split('.')[0], '.obj'))

    r = load_structure(args.ag_pdb_path)
    atomxyz_receptor, atomtypes_receptor, resxyz_receptor, restypes_receptor = tensor(r["atom_xyz"]), tensor(r["atom_types"]), tensor(r["res_xyz"]), Tensor(r["res_types"])
    cmd.load(args.ag_pdb_path)
    cmd.show_as('surface')
    cmd.save(os.path.join(args.ag_pdb_path.split('.')[0], '.obj'))
    pts_receptor, norms_receptor = load_point_cloud_by_file_extension(os.path.join(args.ag_pdb_path.split('.')[0], '.obj'))
    intf_ligand = (torch.cdist(pts_ligand, pts_receptor) < cfg.dataset.surface.intf_cutoff).sum(dim=1) > 0
    intf_receptor = (torch.cdist(pts_receptor, pts_ligand) < cfg.dataset.surface.intf_cutoff).sum(dim=1) > 0
    data = ProteinPairData(xyz_ligand=pts_ligand, normals_ligand=norms_ligand, atomxyz_ligand=atomxyz_ligand, atomtypes_ligand=atomtypes_ligand, resxyz_ligand=resxyz_ligand,
                           restypes_ligand=restypes_ligand, xyz_receptor=pts_receptor, normals_receptor=norms_receptor, atomxyz_receptor=atomxyz_receptor,
                           atomtypes_receptor=atomtypes_receptor, resxyz_receptor=resxyz_receptor, restypes_receptor=restypes_receptor, intf_ligand=intf_ligand,
                           intf_receptor=intf_receptor)
    batch_keys = ["atomxyz_ligand", "resxyz_ligand", "xyz_ligand", "atomxyz_receptor", "resxyz_receptor", "xyz_receptor", "intf_receptor", "intf_pred"]
    data = Batch.from_data_list([data], follow_batch=batch_keys)

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

        patch_xyz_coarse = patch_xyz_receptor[patch_mask_receptor]
        _, patch_idx = torch.cdist(patch_xyz_coarse.cpu(), resxyz_receptor).min(0)
        res_pred_ = pred[patch_idx].cpu().numpy()
        # res_idx = torch.cdist(pts_receptor, resxyz_receptor).min(0)[0] > 2 + 1.05  # 2A for buried residue to the surface point
        # res_pred_[res_idx] = 0

        gt = torch.cdist(resxyz_receptor, resxyz_ligand).min(-1)[0].numpy()
        l_10 = int(len(gt) * 0.1)
        idx = sorted(range(len(res_pred_)), key=lambda i: res_pred_[i])[-l_10:]
        score = (gt[idx] < 4).sum() / l_10
        print(f'L/10: {score}')    # TODO: why randomness? due to surface generation?

    atomxyz, atomtypes, restypes, atom_res = atomxyz_receptor.numpy(), atomtypes_receptor.numpy(), restypes_receptor.numpy(), r['atom_res']
    template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
    with open(f"antigen_{score}.pdb", 'w') as f:
        for i, atomtype in enumerate(atomtypes):
            try:
                resname = AA(restypes[atom_res[i]])._name_
            except:   # padding token break
                break
            xyz = atomxyz[i].tolist()
            atomtype = np.argmax(atomtype)
            f.write(template.format("ATOM", i + 1, num2ele[atomtype], '', resname, 'A', atom_res[i] + 1, '', xyz[0], xyz[1], xyz[2], 1.00,
                                    res_pred_[atom_res[i]] * 100, num2ele[atomtype], ''))
    print('Finished.')


if __name__ == '__main__':
    infer()
