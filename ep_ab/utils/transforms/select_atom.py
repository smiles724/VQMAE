
from ._base import register_transform


@register_transform('select_atom')
class SelectAtom(object):

    def __init__(self, resolution):
        super().__init__()
        assert resolution in ('full', 'backbone', 'backbone+CB')
        self.resolution = resolution

    def __call__(self, data):
        if self.resolution == 'full':
            data['pos_atoms'] = data['pos_heavyatom']
            data['mask_atoms'] = data['mask_heavyatom']
        elif self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_heavyatom'][:, :4]
            data['mask_atoms'] = data['mask_heavyatom'][:, :4]
        elif self.resolution == 'backbone+CB':
            data['pos_atoms'] = data['pos_heavyatom'][:, :5]
            data['mask_atoms'] = data['mask_heavyatom'][:, :5]
        data['epitope'] = data['epitope'] > 0
        return data
