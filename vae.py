import argparse
import functools
import math
import os
import shutil

import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm as tq  # tqdm.auto -> multiprocess error

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.train import *
from src.models import get_model
from src.datasets import Collate
from src.datasets.redo import PDB_REDO_Dataset
from src.utils.misc import inf_iterator, load_config, seed_all, get_logger, get_new_log_dir


def compute_loss(loss_dict, kld_weight, h_weight, niter):
    start, target, ntime_1, ntime_2 = kld_weight.start, kld_weight.target, kld_weight.ntime_1, kld_weight.ntime_2
    hb_, hp_ = h_weight.hbond, h_weight.hphobicity
    if niter > ntime_1 + ntime_2:
        kld_ = target
    elif niter < ntime_1:
        kld_ = 0.
    else:
        kld_ = target + (start - target) * (1. + math.cos(math.pi * float(niter - ntime_1) / ntime_2)) / 2.

    loss = loss_dict['recon'] + kld_ * loss_dict['klv'] + loss_dict['hbond'] * hb_ + loss_dict['hphobicity'] * hp_
    loss_weight = {'recon': 1.0, 'klv': kld_, 'hbond': hb_, 'hphobicity': hp_}
    return loss, loss_weight


def get_temperature(cfg_temp, niter):
    start, target, ntime = cfg_temp.start, cfg_temp.target, cfg_temp.ntime
    if niter > ntime:
        return target
    temp = target + (start - target) * (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
    return temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_redo')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.cfg)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = get_new_log_dir(args.logdir, prefix='%s-resume' % config_name)
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.cfg))):
            shutil.copyfile(args.cfg, os.path.join(log_dir, os.path.basename(args.cfg)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    dataset_ = functools.partial(PDB_REDO_Dataset, pdbredo_dir=config.data.pdbredo_dir, clusters_path=config.data.clusters_path, splits_path=config.data.splits_path,
                                 processed_dir=config.data.processed_dir, surface=config.data.surface)
    train_dataset, val_dataset = dataset_('train'), dataset_('val')
    collate_fn = Collate(vae=True, min_pts=1 / (config.model.patch_setup.patch_ratio * config.model.mask_ratio))
    train_loader = DataLoader(train_dataset, config.train.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, config.train.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    train_iterator = inf_iterator(train_loader)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model & Optimizer & Scheduler
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %.2f M' % (count_parameters(model) / 1e6))
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'], )

    def train(it):
        model.train()
        batch = recursive_to(next(train_iterator), args.device)
        try:  # https://stackoverflow.com/questions/61467751/unable-to-find-a-valid-cudnn-algorithm-to-run-convolution
            temp = get_temperature(config.train.temp, it)
            loss_dict, _ = model.dvae_forward(batch, temperature=temp, hard=False)
            loss, loss_weights = compute_loss(loss_dict, config.train.kld_weight, config.train.h_weight, it)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e) and "Unable to find a valid cuDNN algorithm" not in str(e):
                raise e
            torch.cuda.empty_cache()
            print(f'Skipped batch due to OOM', flush=True)
            return ''

        # Logging
        scalar_dict = {'grad': orig_grad_norm, 'lr(1e-4)': optimizer.param_groups[0]['lr'] * 1e4, 'temp': temp}
        logstr = '[train] Iter %05d | loss %.2f' % (it, loss.item())
        for k, v in loss_dict.items():
            logstr += ' | %s %.2f' % (k, v.item() * loss_weights[k] if isinstance(v, torch.Tensor) else v * loss_weights[k])
        for k, v in scalar_dict.items():
            logstr += ' | %s %.2f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
        write_losses(loss, loss_dict, scalar_dict, it=it, tag='train', writer=writer)
        return logstr

    def validate(it, best_it, best_loss):
        model.eval()
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            for _, batch in enumerate(tq(val_loader, desc='Validate', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict, out_dict = model.dvae_forward(batch, hard=True)

                dense_loss_l2 = []
                for i in range(batch['size']):
                    mask = out_dict['mask'][i]
                    dense_loss_l2.append(model.cdl2(out_dict['fine'][i][mask].reshape(1, -1, 3), out_dict['pts'][i][mask].reshape(1, -1, 3)))
                dense_loss_l2 = sum(dense_loss_l2) / len(dense_loss_l2)
                scalar_accum.add(name='DenseLossL2', value=dense_loss_l2, batchsize=batch['size'], mode='mean')
                scalar_accum.add(name='HbondLoss', value=loss_dict['hbond'], batchsize=batch['size'], mode='mean')
                scalar_accum.add(name='HphobicityLoss', value=loss_dict['hphobicity'], batchsize=batch['size'], mode='mean')

        avg_DenseL2 = scalar_accum.get_average('DenseLossL2')
        avg_Hbond = scalar_accum.get_average('HbondLoss')
        avg_HphobicityLoss = scalar_accum.get_average('HphobicityLoss')
        avg_loss = avg_DenseL2 + avg_Hbond * config.train.h_weight.hbond + avg_HphobicityLoss * config.train.h_weight.hphobicity
        if best_loss > avg_loss:
            best_loss = avg_loss
            best_it = it
        scalar_accum.log(it, 'val', logger=logger, writer=writer, best_it=best_it, best_loss=best_loss)
        loss_dict = {'DenseL2': avg_DenseL2, 'HbondLoss': avg_Hbond, 'HphobicityLoss': avg_HphobicityLoss}
        write_losses(None, loss_dict, {}, it=it, tag='val', writer=writer)

        if it != it_first:  # Don't stop optimizers after resuming from checkpoint
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            elif config.train.scheduler.type == 'CosLR':
                scheduler.step(it)
            else:
                scheduler.step()
        return avg_loss, best_loss, best_it

    try:
        best_val_loss, best_val_it = 1e9, 0
        it_tqdm = tq(range(it_first, config.train.max_iters + 1))
        for it in it_tqdm:
            message = train(it)
            it_tqdm.set_description(message)

            if it % config.train.val_freq == 0:
                avg_val_loss, best_val_loss, best_val_it = validate(it, best_val_it, best_val_loss)
                if not args.debug and best_val_it == it:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': it,
                                'avg_val_loss': avg_val_loss, }, ckpt_path)
        writer.close()
    except KeyboardInterrupt:
        logger.info('Terminating...')
