import argparse
import functools
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, recall_score, precision_score, average_precision_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm as tq   # tqdm.auto -> multiprocess error

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.datasets import SAbDabDataset, Collate
from src.models import get_model
from src.utils.misc import *
from src.utils.train import *
from src.utils.transforms import get_transform
import warnings
warnings.filterwarnings("ignore")   # ignore warning for calculating F1 scores


def train(it):
    model.train()
    batch = recursive_to(next(train_iterator), args.device)
    try:
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, cfg.train.loss_weights)
        loss_dict['overall'] = loss
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e) and "no valid convolution" not in str(e) and "find a valid cuDNN" not in str(e): raise e
        torch.cuda.empty_cache()
        print(f'Skipped batch due to OOM', flush=True)
        return ''

    scalar_dict = {'grad': orig_grad_norm, 'lr (1e-5)': optimizer.param_groups[0]['lr'] * 1e5, }
    log_losses(loss_dict, it, 'train', writer, others=scalar_dict)
    logstr = '[Train] Iter %05d | loss %.2f' % (it, loss)
    for k, v in scalar_dict.items():
        logstr += ' | %s %2.2f' % (k, v)

    if not torch.isfinite(loss):
        logger.error('NaN or Inf detected.')
        torch.save({'cfg': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': it,
                    'batch': recursive_to(batch, 'cpu'), }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
        raise KeyboardInterrupt()
    return logstr


def evaluation(it, loader, mode='val'):
    model.eval()
    scalar_accum = ScalarMetricAccumulator()
    res, ep_true_all, ep_pred_all = [], [], []
    with torch.no_grad():
        for _, batch in enumerate(tq(loader, desc='Validate', dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            loss_dict, output = model(batch)
            if mode == 'val':
                loss_dict['overall'] = sum_weighted_losses(loss_dict, cfg.train.loss_weights)
                scalar_accum.add(name='ep_loss', value=loss_dict['ep'], batchsize=batch['size'], mode='mean')

            # per-protein (per-complex) metric
            if 'mask' not in batch:  # torch geometric data does not have 'mask'
                if 'surfformer' in cfg.model.type and cfg.train.eval_resolution == 'patch':  # patch-level validation
                    batch_idx = output['batch_idx']
                    if mode == 'test':
                        res_true, res_pred = [], []
                        for k in range(batch.num_graphs):
                            patch_xyz_coarse = output['patch_xyz_coarse'][batch_idx == k]
                            resxyz_receptor = batch['resxyz_receptor'][batch['resxyz_receptor_batch'] == k]
                            resxyz_ligand = batch['resxyz_ligand'][batch['resxyz_ligand_batch'] == k]
                            _, patch_idx = torch.cdist(patch_xyz_coarse, resxyz_receptor).min(0)

                            res_pred_ = output['ep_pred_coarse'][batch_idx == k][patch_idx]   # assign the closest patch's prediction to residues
                            # xyz_receptor = batch['xyz_receptor'][batch['xyz_receptor_batch'] == k]
                            # res_idx = torch.cdist(xyz_receptor, resxyz_receptor).min(0)[0] > 2 + 1.05  # 2A for buried residue to the surface point
                            # res_pred_[res_idx] = 0

                            res_pred.append(res_pred_.cpu().numpy())
                            res_true.append((torch.cdist(resxyz_receptor, resxyz_ligand).min(-1)[0] < cfg.dataset.intf_cutoff_res).cpu().numpy())
                        pc_list = zip(batch['id'], res_true, res_pred)
                    else:
                        ep_pred_coarse = [output['ep_pred_coarse'][batch_idx == k].cpu().numpy() for k in range(batch.num_graphs)]
                        ep_true_coarse = [output['ep_true_coarse'][batch_idx == k].cpu().numpy() for k in range(batch.num_graphs)]
                        pc_list = zip(batch['id'], ep_true_coarse, ep_pred_coarse)  # (B, K)
                else:
                    batch.intf_pred = output['ep_pred']  # point-level validation
                    pc_list = [[data.id, data.intf_receptor.cpu().numpy(), data.intf_pred.cpu().numpy()] for data in batch.to_data_list()]
            else:
                pc_list = [[idx, ep_true[mask].cpu().numpy(), ep_pred[mask].cpu().numpy()] for idx, mask, ep_true, ep_pred in
                           zip(batch['id'], batch['mask'], output['ep_true'], output['ep_pred'])]
            for idx, ep_true, ep_pred in pc_list:
                roc_auc_pc = roc_auc_score(ep_true, ep_pred)

                top_k = np.argsort(ep_pred)[::-1][:len(ep_pred) // 10]
                l_10_ppv = ep_true[top_k].sum() / (len(ep_pred) // 10)  # positive predictive value at L/10

                # fine the best threshold for F1: https://stackoverflow.com/questions/57060907/compute-maximum-f1-score-using-precision-recall-curve
                best_threshold, mcc = find_optimal_threshold(ep_true, ep_pred)
                balanced_acc = balanced_accuracy_score(ep_true, ep_pred > best_threshold)

                precision_pc = precision_score(ep_true, (ep_pred > best_threshold).astype(int))
                recall_pc = recall_score(ep_true, (ep_pred > best_threshold).astype(int))
                f1_pc = 2 * recall_pc * precision_pc / (recall_pc + precision_pc)

                # https://stats.stackexchange.com/questions/338826/auprc-vs-auc-roc
                # https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
                # https://sinyi-chou.github.io/python-sklearn-precision-recall/
                au_prc_pc = average_precision_score(ep_true, ep_pred)

                ep_true_all.append(ep_true)
                ep_pred_all.append(ep_pred)
                res.append({'complex': idx, 'roc_auc': roc_auc_pc, 'au_prc': au_prc_pc, 'precision': precision_pc, 'recall': recall_pc, 'balanced_acc': balanced_acc, 'f1': f1_pc,
                            'l/10': l_10_ppv, 'mcc': mcc})

    res = pd.DataFrame(res)
    summary = {k: res[k].median() for k in res.columns if k != 'complex'}
    if mode == 'val':
        avg_loss = scalar_accum.get_average('ep_loss')
        if cfg.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        for k, v in summary.items():
            writer.add_scalar('%s/%s' % (mode, k), v, it)
        return summary['roc_auc'], summary

    return summary, np.concatenate(ep_true_all), np.concatenate(ep_pred_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    cfg, cfg_name = load_config(args.config)
    seed_all(cfg.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=cfg_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(cfg)

    # Data & Model
    logger.info(f'Loading dataset (ESM embedding: {cfg.model.get("use_plm", None)})...')
    dataset_ = functools.partial(SAbDabDataset, processed_dir=cfg.dataset.processed_dir, surface=cfg.dataset.get('surface', None), use_plm=cfg.model.get('use_plm', None),
                                 transform=get_transform(cfg.dataset.transform) if 'transform' in cfg.dataset else None)
    train_set = dataset_(split='train', relax_struct=cfg.dataset.relax_struct, pred_struct=cfg.dataset.pred_struct, )
    val_set = dataset_(split='val', relax_struct=cfg.dataset.relax_struct, pred_struct=cfg.dataset.pred_struct, )

    train_iterator = inf_iterator(DataLoader(train_set, batch_size=cfg.train.batch_size, collate_fn=Collate(max_pts=cfg.train.max_pts), shuffle=True, num_workers=args.num_workers))
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, collate_fn=Collate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_set), len(val_set)))
    logger.info('Building model...')
    if args.ckpt is not None:  # Fine-tune
        logger.info('Fine-tuning from checkpoint: %s' % args.ckpt)
        ckpt = torch.load(args.ckpt, map_location='cpu')
        for key in cfg.model.keys():
            if key in ckpt['config'].model.keys():
                cfg.model[key] = ckpt['config'].model[key]
        model = get_model(cfg.model).to(args.device)
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model = get_model(cfg.model).to(args.device)
    logger.info('Number of parameters: %.2f M' % (count_parameters(model) / 1e6))
    optimizer = get_optimizer(cfg.train.optimizer, model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    try:
        best_val_metric, best_i = 0, 0   # use AUCORC as metric
        it_tqdm = tq(range(it_first, cfg.train.max_iters + 1))
        for i in it_tqdm:
            message = train(i)
            it_tqdm.set_description(message)
            if i % cfg.train.val_freq == 0:
                val_metric, val_metrics = evaluation(i, val_loader)
                if val_metric > best_val_metric:
                    best_val_metric, best_i = val_metric, i
                logger.info('[Val] Iter %05d | ROCAUC %.3f | AUCPR %.3f | Precision %.3f | Recall %.3f | BAcc %.3f | F1 %.3f | L/10 %.3f | Best iter %05d (%.4f)' % (
                    i, val_metrics['roc_auc'], val_metrics['au_prc'], val_metrics['precision'], val_metrics['recall'], val_metrics['balanced_acc'], val_metrics['f1'],
                    val_metrics['l/10'], best_i, best_val_metric,))
                if not args.debug and best_i == i:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % i)
                    torch.save({'cfg': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': i,
                                'val_metric': val_metric, }, ckpt_path)
        writer.close()

        logger.info('=== Start Testing ===')
        ckpt = torch.load(os.path.join(ckpt_dir, '%d.pt' % best_i), map_location=args.device)
        model.load_state_dict(ckpt['model'])
        test_dataset = dataset_(split='test', relax_struct=False, pred_struct=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size * 2, collate_fn=Collate(), shuffle=False, num_workers=args.num_workers)
        test_out, test_ep_true_all, test_ep_pred_all = evaluation(0, test_loader, mode='test')

        precision_all_0, recall_all_0, thresholds_all_0 = precision_recall_curve(test_ep_true_all, test_ep_pred_all)
        logger.info('[Test on PDB] ROCAUC %.3f | AUCPR %.3f | Precision %.3f | Recall %.3f | BAcc %.3f | F1 %.3f | L/10 %.3f' % (
            test_out['roc_auc'], test_out['au_prc'], test_out['precision'], test_out['recall'], test_out['balanced_acc'], test_out['f1'], test_out['l/10']))
        if cfg.model.type != 'ga': sys.exit()

        test_relax_dataset = dataset_(split='test', relax_struct=True, pred_struct=False)
        test_relax_loader = DataLoader(test_relax_dataset, batch_size=cfg.train.batch_size * 2, collate_fn=Collate(), shuffle=False, num_workers=args.num_workers)
        test_out, test_ep_true_all, test_ep_pred_all = evaluation(0, test_relax_loader, mode='test')
        precision_all_1, recall_all_1, thresholds_all_1 = precision_recall_curve(test_ep_true_all, test_ep_pred_all)
        logger.info('[Test on Relax] ROCAUC %.3f | AUCPR %.3f | Precision %.3f | Recall %.3f | BAcc %.3f | F1 %.3f | L/10 %.3f' % (
            test_out['roc_auc'], test_out['au_prc'], test_out['precision'], test_out['recall'], test_out['balanced_acc'], test_out['f1'], test_out['l/10']))

        test_pred_dataset = dataset_(split='test', relax_struct=False, pred_struct=True)
        test_pred_loader = DataLoader(test_pred_dataset, batch_size=cfg.train.batch_size * 2, collate_fn=Collate(), shuffle=False, num_workers=args.num_workers)
        test_out, test_ep_true_all, test_ep_pred_all = evaluation(0, test_pred_loader, mode='test')
        precision_all_2, recall_all_2, thresholds_all_2 = precision_recall_curve(test_ep_true_all, test_ep_pred_all)
        logger.info('[Test on Pred] ROCAUC %.3f | AUCPR %.3f | Precision %.3f | Recall %.3f | BAcc %.3f | F1 %.3f | L/10 %.3f' % (
            test_out['roc_auc'], test_out['au_prc'], test_out['precision'], test_out['recall'], test_out['balanced_acc'], test_out['f1'], test_out['l/10']))
        logger.info('Test PDB %d | Relax %d | Pred %d' % (len(test_dataset), len(test_relax_dataset), len(test_pred_dataset)))

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('SabDab')
        ax0.plot(recall_all_0, precision_all_0)
        ax1.plot(recall_all_1, precision_all_1)
        ax2.plot(recall_all_2, precision_all_2)
        ax0.set_xlabel('Recall')
        ax0.set_ylabel('Precision')
        ax1.set_xlabel('Recall')
        ax2.set_xlabel('Recall')
        plt.savefig(f'./pr_curve_all_Pred{cfg.dataset.pred_struct}_Relax{cfg.dataset.relax_struct}.pdf')

    except KeyboardInterrupt:
        logger.info('Terminating...')
