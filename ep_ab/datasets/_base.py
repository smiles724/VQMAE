import math
import torch
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch

from ep_ab.utils.transforms import get_transform

DEFAULT_PAD_VALUES = {'aa': 21, 'chain_id': ' ', 'icode': ' ', }
DEFAULT_NO_PADDING = {'origin', }


_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls
    return decorator


def get_dataset(cfg):
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    return _DATASET_DICT[cfg.type](cfg, transform=transform)


class PaddingCollate(object):

    def __init__(self, vae=False, min_pts=1e4, max_pts=None, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.vae = vae
        self.min_pts = min_pts
        self.max_pts = max_pts
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([torch.ones([l], dtype=torch.bool), torch.zeros([n - l], dtype=torch.bool)], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        if 'P' in data_list[0]:
            # Surface
            if self.vae:
                batch_keys = ["resxyz", "xyz"]
                data_list = [i['P'] for i in data_list if len(i['P']["xyz"]) >= self.min_pts]
                batch = Batch.from_data_list(data_list, follow_batch=batch_keys)
            else:
                batch_keys = ["atomxyz_ligand", "resxyz_ligand", "xyz_ligand", "atomxyz_receptor", "resxyz_receptor", "xyz_receptor", "intf_receptor", "intf_pred"]
                if self.max_pts is not None:      # filter too long sequences (for FPS)
                    data_list = [i for i in data_list if len(i['P']["xyz_receptor"]) <= self.max_pts and len(i['P']["xyz_ligand"]) <= self.max_pts]

                batch = Batch.from_data_list([i['P'] for i in data_list], follow_batch=batch_keys)     # do not need 'to_dict()'
                batch['min_num_nodes'] = torch.tensor([min(len(i['P']["xyz_receptor"]), len(i['P']["xyz_ligand"])) for i in data_list]).min()
        else:
            # Structure
            max_length = max([data[self.length_ref_key].size(0) for data in data_list])
            keys = self._get_common_keys(data_list)

            max_length = math.ceil(max_length / 8) * 8 if self.eight else max_length
            data_list_padded = []
            for data in data_list:
                data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v for k, v in data.items() if k in keys}
                data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
                data_list_padded.append(data_padded)
            batch = default_collate(data_list_padded)

        batch['size'] = len(data_list)
        return batch
