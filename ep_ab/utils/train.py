import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.data
from sklearn.metrics import matthews_corrcoef

from .misc import BlackHole


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2,))
    elif cfg.type == 'adamw':
        # https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
        param_groups = add_weight_decay(model, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2,))
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr, )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma, )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma, )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it: (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(out, it, tag, writer=BlackHole(), others={}):
    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, best_it=None, best_loss=None, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.2f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)

        if best_loss is not None:
            logstr += ' | Best iter %05d | Best loss %.4f' % (best_it, best_loss)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def write_losses(loss, loss_dict, scalar_dict, it, tag, writer=BlackHole()):
    if loss is not None:
        writer.add_scalar('%s/loss' % tag, loss, it)
    for k, v in loss_dict.items():
        writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in scalar_dict.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor) or isinstance(obj, torch_geometric.data.Data):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    return obj


def reweight_loss_by_sequence_length(length, max_length, mode='sqrt'):
    if mode == 'sqrt':
        w = np.sqrt(length / max_length)
    elif mode == 'linear':
        w = length / max_length
    elif mode is None:
        w = 1.0
    else:
        raise ValueError('Unknown reweighting mode: %s' % mode)
    return w


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2, reduction: str = "none", ) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example, range between 0 and 1 instead of logits.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    return loss


def find_optimal_threshold(y_true, y_probs):
    """
    Finds the optimal classification threshold to maximize the Matthews Correlation Coefficient (MCC).

    Parameters:
    - y_true: Array-like of true binary labels.
    - y_probs: Array-like of predicted probabilities for the positive class.

    Returns:
    - optimal_threshold: The threshold value that maximizes MCC.
    - max_mcc: The maximum MCC value achieved.
    """
    thresholds = np.linspace(0, 1, num=100)  # Generate 100 threshold values between 0 and 1
    max_mcc = -1  # Initialize max MCC to the lowest possible value
    optimal_threshold = 0.5  # Initialize optimal threshold at 0.5

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the current threshold
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate MCC for the current set of predictions
        mcc = matthews_corrcoef(y_true, y_pred)

        # Update max_mcc and optimal_threshold if the current MCC is higher than max_mcc
        if mcc > max_mcc:
            max_mcc = mcc
            optimal_threshold = threshold

    return optimal_threshold, max_mcc
