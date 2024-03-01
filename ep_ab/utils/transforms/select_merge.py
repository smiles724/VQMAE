import torch

from ..protein import constants
from ._base import register_transform


def sum_(values, start):
    total = start
    for value in values:
        total = total + value
    return total


@register_transform('select_and_merge_chains')
class SelectAndMergeChains(object):

    def __init__(self, chains, max_len=None, use_sasa=False):
        super().__init__()
        self.chains = chains.split('+')
        self.max_len = max_len
        self.use_sasa = use_sasa

    def assign_chain_number_(self, data_list):
        chains = set()
        for data in data_list:
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([chains[c] for c in data['chain_id']])

    def _data_attr(self, data, name):
        if name == 'epitope' and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        return data[name]

    def __call__(self, structure):
        data_list = []
        if structure['heavy'] is not None and 'heavy' in self.chains:
            structure['heavy']['fragment_type'] = torch.full_like(structure['heavy']['aa'], fill_value=constants.Fragment.Heavy, )
            data_list.append(structure['heavy'])

        if structure['light'] is not None and 'light' in self.chains:
            structure['light']['fragment_type'] = torch.full_like(structure['light']['aa'], fill_value=constants.Fragment.Light, )
            data_list.append(structure['light'])

        if structure['antigen'] is not None and 'antigen' in self.chains:
            structure['antigen']['fragment_type'] = torch.full_like(structure['antigen']['aa'], fill_value=constants.Fragment.Antigen, )
            structure['antigen']['cdr_flag'] = torch.zeros_like(structure['antigen']['aa'], )
            data_list.append(structure['antigen'])

        self.assign_chain_number_(data_list)

        list_props = {'chain_id': [], 'icode': [], }
        tensor_props = {'chain_nb': [], 'resseq': [], 'res_nb': [], 'aa': [], 'pos_heavyatom': [], 'mask_heavyatom': [], 'fragment_type': [], 'epitope': [],
                        'phi': [], 'phi_mask': [], 'psi': [], 'psi_mask': [], 'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': []}
        if self.use_sasa:
            tensor_props['sasa'] = []
        if 'plm_feature' in structure['antigen'].keys():
            tensor_props['plm_feature'] = []

        for data in data_list:
            for k in list_props.keys():
                attr = self._data_attr(data, k)
                if self.max_len is not None:
                    attr = attr[:self.max_len]
                list_props[k].append(attr)
            for k in tensor_props.keys():
                attr = self._data_attr(data, k)
                if self.max_len is not None:
                    attr = attr[:self.max_len]
                tensor_props[k].append(attr)
        try:   # different versions, 'start=xxx' not supported for Python < 3.8
            list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        except:
            list_props = {k: sum_(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {**list_props, **tensor_props, 'id': structure['id']}
        return data_out
