"""
dMaSIF visualization of proteins:
https://colab.research.google.com/github/casperg92/MaSIF_colab/blob/main/dMaSIF_Colab_V1.ipynb#scrollTo=HKgFCDrM1dO0
"""
import argparse
import os
import pickle

import lmdb
import pandas as pd
import random

from torch.utils.data import DataLoader
from tqdm import tqdm as tq

import matplotlib.pyplot as plt
import seaborn as sns

from ep_ab.datasets import SAbDabDataset, PaddingCollate
from ep_ab.datasets.redo import PDB_REDO_Dataset
from ep_ab.models import get_model
from ep_ab.models.surfformer_v1 import index_points
from ep_ab.models.surfformer_v2 import ChamferDistance
from ep_ab.utils.misc import seed_all, load_config
from ep_ab.utils.train import *


def dataset_statstics():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', action='store_true')
    args = parser.parse_args()

    processed_dir = './data/processed'
    _entry_cache_path = os.path.join(processed_dir, 'entry')
    with open(_entry_cache_path, 'rb') as f:
        sabdab_entries = pickle.load(f)
    val_split = [entry['id'] for entry in sabdab_entries if entry['id'][:4] in test_id]

    if args.surface:
        lmdb_path = os.path.join(processed_dir, 'surface.lmdb')
    else:
        lmdb_path = os.path.join(processed_dir, 'structures.lmdb')
    MAP_SIZE = 32 * (1024 * 1024 * 1024)
    db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )

    seq_len, num_ep = [], []
    for idx in val_split:
        with db_conn.begin() as txn:
            x = pickle.loads(txn.get(idx.encode()))
        if args.surface:
            ag = x['P']
            seq_len.append(len(ag['xyz_receptor']))
            num_ep.append(ag['intf_receptor'].sum().item())
        else:
            ag = x['antigen']
            seq_len.append(len(ag['aa']))
            num_ep.append(ag['epitope'].sum().item())

    df = pd.DataFrame({'seq_len': seq_len, 'num_ep': num_ep})
    df.to_csv(f'data_summary_{"surf" if args.surface else "residue"}.csv', index=False)


def codebook_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--multiEpitope_csv', type=str, default=None)  # './multiEpitope_antigens.csv'
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Load configs
    cfg_dataset, _ = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg_model = ckpt['config']
    seed_all(cfg_model.train.seed)
    test_dataset = SAbDabDataset(split=args.split, processed_dir=cfg_dataset.dataset.processed_dir, surface=cfg_dataset.dataset.get('surface', None),
                                 multiEpitope_csv=args.multiEpitope_csv)
    test_loader = DataLoader(test_dataset, batch_size=cfg_dataset.train.batch_size * 2, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)

    # Model
    model = get_model(cfg_model.model).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    ids, classes, labels, xyzs = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tq(test_loader, dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            P = {'resxyz': batch['resxyz_receptor'], 'restypes': batch['restypes_receptor'], 'batch_res': batch['resxyz_receptor_batch'], 'xyz': batch['xyz_receptor'],
                 'normals': batch['normals_receptor'], 'batch': batch['xyz_receptor_batch']}
            patch_feat, patch_idx, patch_xyz, patch_mask, xyz_dense, group_idx = model.point_forward(P)
            logits = model.tokenizer_mlp(patch_feat)  # logits: (B, npatch, ntoken)
            patch_class = F.gumbel_softmax(logits, tau=1.0, dim=2, hard=True).max(-1)[1][patch_mask]
            label_coarse = batch['intf_receptor'][patch_idx]
            batch_id = P['batch'][patch_idx]
            grouped_xyz = index_points(xyz_dense, group_idx)  # grouped_xyz: (B, npatch, n_pts, 3)
            grouped_xyz = grouped_xyz[patch_mask]
            for j in range(batch['size']):
                labels.append(label_coarse[batch_id == j].cpu())
                classes.append(patch_class[batch_id == j].cpu())
                xyzs.append(grouped_xyz[batch_id == j].cpu())
                ids.append(batch['id'][j])

    patch_setup = cfg_model.model.patch_setup
    torch.save([ids, classes, labels, xyzs], f'./patch_{cfg_model.model.vocab_size}_{patch_setup.patch_ratio}_{patch_setup.n_pts_per_patch}.pt')

    patch_xyz_dict, index_dict = {}, {}
    for c, xyz in zip(classes, xyzs):
        for i in range(len(c)):
            if c[i].item() not in index_dict.keys():
                index_dict[c[i].item()] = len(index_dict) + 1
                patch_xyz_dict[index_dict[c[i].item()]] = [xyz[i]]
            else:
                patch_xyz_dict[index_dict[c[i].item()]].append(xyz[i])
    print(f'Num classes: {len(patch_xyz_dict)}')
    cdl2 = ChamferDistance().cuda()
    patch_classes = list(patch_xyz_dict.keys())
    dists = {}
    for i in range(len(patch_classes)):
        for j in range(i, len(patch_classes)):
            coord = torch.stack(random.choices(patch_xyz_dict[patch_classes[i]], k=100), dim=0).cuda()
            coord_ = torch.stack(random.choices(patch_xyz_dict[patch_classes[j]], k=100), dim=0).cuda()
            dists[(i, j)] = cdl2(coord, coord_).item()
    torch.save(dists, f'./dist_{cfg_model.model.vocab_size}_{patch_setup.patch_ratio}_{patch_setup.n_pts_per_patch}.pt')


def reconstruction_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--chem', action='store_true')
    args = parser.parse_args()

    # Load configs
    cfg_dataset, _ = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg_model = ckpt['config']
    patch_cfg = cfg_model.model.patch_setup
    seed_all(cfg_model.train.seed)
    test_dataset = PDB_REDO_Dataset(split=args.split, pdbredo_dir=cfg_dataset.data.pdbredo_dir, clusters_path=cfg_dataset.data.clusters_path,
                                    splits_path=cfg_dataset.data.splits_path, processed_dir=cfg_dataset.data.processed_dir, surface=cfg_dataset.data.surface)
    collate_fn = PaddingCollate(vae=True, min_pts=1 / (cfg_dataset.model.patch_setup.patch_ratio * cfg_dataset.model.mask_ratio))
    test_loader = DataLoader(test_dataset, cfg_dataset.train.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Model
    model = get_model(cfg_model.model).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    pred_pts, gt_pts, whole_pts, dense_l2 = [], [], [], []
    pred_hphs, gt_hphs, pred_hbs, gt_hbs = [], [], [], []
    pred_hph, gt_hph, pred_hb, gt_hb, loss_hph, loss_hb, pts_hph, pts_hb = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for _, batch in enumerate(tq(test_loader, dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            try:
                if args.chem:
                    hphobicity_pred, hbond_pred, hphobicity_gt, hbond_gt, batch_id, grouped_xyz = model.dvae_forward(batch, hard=True, return_chem=True)
                    pred_hphs.append(hphobicity_pred)
                    gt_hphs.append(hphobicity_gt)
                    pred_hbs.append(hbond_pred)
                    gt_hbs.append(hbond_gt)
                    for i in range(batch['size']):
                        pred = hphobicity_pred[batch_id == i]
                        gt = hphobicity_gt[batch_id == i]
                        loss_hphobicity = F.mse_loss(pred, gt)
                        loss_hph.append(loss_hphobicity)
                        if loss_hphobicity <= 1.20:
                            print(f'Min hphobicity loss: {loss_hphobicity}')
                            pred_hph.append(pred)
                            gt_hph.append(gt)
                            pts_hph.append(grouped_xyz[batch_id == i])
                            torch.save([pts_hph, gt_hph, pred_hph], f'./hph_{cfg_model.model.vocab_size}_{patch_cfg.patch_ratio}_{patch_cfg.n_pts_per_patch}.pt')
                        pred_ = hbond_pred[batch_id == i]
                        gt_ = hbond_gt[batch_id == i]
                        loss_hbond = F.mse_loss(pred_, gt_)
                        loss_hb.append(loss_hbond)
                        if loss_hbond <= 0.002:
                            print(f'Min hbond loss: {loss_hbond}')
                            pred_hb.append(pred_)
                            gt_hb.append(gt_)
                            pts_hb.append(grouped_xyz[batch_id == i])
                            torch.save([pts_hb, gt_hb, pred_hb], f'./hb_{cfg_model.model.vocab_size}_{patch_cfg.patch_ratio}_{patch_cfg.n_pts_per_patch}.pt')
                else:
                    loss_dict, out_dict = model.dvae_forward(batch, hard=True)
                    for i in range(batch['size']):
                        mask = out_dict['mask'][i]
                        pred = out_dict['fine'][i][mask].reshape(1, -1, 3)
                        gt = out_dict['pts'][i][mask].reshape(1, -1, 3)
                        l2 = model.cdl2(pred, gt)

                        dense_l2.append(l2)
                        if l2 <= min(dense_l2):
                            print(f'Min L2 loss: {min(dense_l2)}')
                            pred_pts.append(pred.squeeze())
                            gt_pts.append(gt.squeeze())
                            whole_pts.append(batch['xyz'][batch['xyz_batch'] == i])
                            torch.save([pred_pts, gt_pts, whole_pts, dense_l2], f'./recon_{cfg_model.model.vocab_size}_{patch_cfg.patch_ratio}_{patch_cfg.n_pts_per_patch}.pt')
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e) and "Unable to find a valid cuDNN algorithm" not in str(e):
                    raise e
                torch.cuda.empty_cache()
                continue
    torch.save([pred_hph, gt_hph, pred_hb, gt_hb], 'recon.pt')

###########################################
# Attention Map
###########################################


def draw(data, ax, linecolor=None):
    sns.heatmap(data, cbar=True, ax=ax, cmap="Greens", linecolor=linecolor)


def attention_visualize():
    sns.set(font_scale=2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--multiEpitope_csv', type=str, default=None)  # './multiEpitope_antigens.csv'
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Load configs
    print('Loading checkpoint: %s' % args.ckpt)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    seed_all(cfg.train.seed)
    model = get_model(cfg.model).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    test_dataset = SAbDabDataset(split=args.split, processed_dir=cfg.dataset.processed_dir, surface=cfg.dataset.get('surface', None), multiEpitope_csv=args.multiEpitope_csv)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tq(test_loader, dynamic_ncols=True)):
            data = recursive_to(data, args.device)
            P_receptor = {'resxyz': data['resxyz_receptor'], 'restypes': data['restypes_receptor'], 'batch_res': data['resxyz_receptor_batch'], 'xyz': data['xyz_receptor'],
                          'normals': data['normals_receptor'], 'batch': data['xyz_receptor_batch']}
            P_ligand = {'resxyz': data['resxyz_ligand'], 'restypes': data['restypes_ligand'], 'batch_res': data['resxyz_ligand_batch'], 'xyz': data['xyz_ligand'],
                        'normals': data['normals_ligand'], 'batch': data['xyz_ligand_batch']}
            patch_feat_receptor, patch_idx_receptor, patch_xyz_receptor, patch_mask_receptor, _, _ = model.point_forward(P_receptor)
            patch_feat_ligand, _, patch_xyz_ligand, patch_mask_ligand, _, _ = model.point_forward(P_ligand)
            model.patch_forward(patch_feat_receptor, patch_xyz_receptor, patch_feat_=patch_feat_ligand,
                                patch_mask=patch_mask_receptor.unsqueeze(1).unsqueeze(2) if patch_mask_receptor is not None else None,
                                patch_mask_=patch_mask_ligand.unsqueeze(1).unsqueeze(2) if patch_mask_ligand is not None else None)

            # draw distance map
            dist = torch.cdist(patch_xyz_receptor, patch_xyz_ligand).squeeze()
            dist_min, _ = torch.min(dist, dim=-1)
            _, idx = torch.sort(dist_min, )
            sns.heatmap(dist[idx].cpu().numpy(), cmap="Greens")  # xticklabels=str_list, yticklabels=str_list,
            plt.savefig(f'fig/{data["id"][0]}_dist.png')
            plt.close('all')

            # the first four heads
            fig, axs = plt.subplots(4, 3, figsize=(50, 50))
            for h in range(4):
                for layer in range(cfg.model.transformer.n_layers):
                    draw_in = model.blocks[layer].cross_attn.attn[0, h].data.cpu()[idx]
                    draw(draw_in, ax=axs[h, layer])
                    if h == 0:
                        axs[h, layer].title.set_text("Layer: {}".format(layer + 1))
                    if layer == 0:
                        axs[h, layer].set_ylabel("Head: {}".format(h + 1))
            plt.savefig(f'fig/{data["id"][0]}_att_4.png')
            plt.close('all')

            # last four heads
            fig, axs = plt.subplots(4, 3, figsize=(50, 50))
            for h in range(4, 8):
                for layer in range(cfg.model.transformer.n_layers):
                    draw_in = model.blocks[layer].cross_attn.attn[0, h].data.cpu()[idx]
                    draw(draw_in, ax=axs[h - 4, layer])
                    if h - 4 == 0:
                        axs[h - 4, layer].title.set_text("Layer: {}".format(layer + 1))
                    if layer == 0:
                        axs[h - 4, layer].set_ylabel("Head: {}".format(h + 1))
            plt.savefig(f'fig/{data["id"][0]}_att_8.png')
            plt.close('all')


###########################################
# Inference
###########################################


def infer():
    sns.set(font_scale=2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./logs')
    parser.add_argument('--pdbs', type=str, default='')  # 7duo_H_L_B,8byu_H_L_A,4cmh_B_C_A
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Load configs
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    seed_all(cfg.train.seed)
    model = get_model(cfg.model).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    test_dataset = SAbDabDataset(processed_dir=cfg.dataset.processed_dir, surface=cfg.dataset.get('surface', None), test_list=args.pdbs.split(','))
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tq(test_loader, dynamic_ncols=True)):
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
            pred_coarse = model.classifier(patch_feat_receptor).squeeze()
            grouped_xyz_receptor = index_points(patch_xyz_dense_receptor, group_idx_receptor).squeeze()     # (B, npatch, n_pts, 3)
            torch.save([grouped_xyz_receptor.cpu().numpy(), data['xyz_ligand'].cpu().numpy(), pred_coarse.cpu().numpy()], f'./infer_{data["id"][0]}.pt')


if __name__ == '__main__':
    reconstruction_analysis()
