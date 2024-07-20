import os
import time
import random
import logging
import torch
import torch.linalg
import numpy as np
import yaml
from easydict import EasyDict
from glob import glob

tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
inttensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
numpy = lambda x: x.detach().cpu().numpy()


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class Counter(object):
    def __init__(self, start=0):
        super().__init__()
        self.now = start

    def step(self, delta=1):
        prev = self.now
        self.now += delta
        return prev


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    # https://discuss.pytorch.org/t/torch-backends-cudnn-benchmark-and-runtimeerror-cudnn-error/117382
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    # Ns = batch.bincount()
    try:
        Ns = batch.bincount()
    except:
        print(batch)
        print(max(batch), min(batch))
        raise ValueError
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device))
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def soft_dimension(features):
    """Continuous approximation of the rank of a (N, D) sample.

    Let "s" denote the (D,) vector of eigenvalues of Cov,
    the (D, D) covariance matrix of the sample "features".
    Then,
        R(features) = \sum_i sqrt(s_i) / \max_i sqrt(s_i)

    This quantity encodes the number of PCA components that would be
    required to describe the sample with a good precision.
    It is equal to D if the sample is isotropic, but is generally much lower.

    Up to the re-normalization by the largest eigenvalue,
    this continuous pseudo-rank is equal to the nuclear norm of the sample.
    """

    nfeat = features.shape[-1]
    features = features.view(-1, nfeat)
    x = features - torch.mean(features, dim=0, keepdim=True)
    cov = x.T @ x
    try:
        u, s, v = torch.svd(cov)
        R = s.sqrt().sum() / s.sqrt().max()
    except:
        return -1
    return R.item()
