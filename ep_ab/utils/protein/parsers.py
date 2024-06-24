import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from easydict import EasyDict

from .constants import (AA, max_num_heavyatoms, restype_to_heavyatom_names, BBHeavyAtom)
from .icoord import get_chi_angles, get_backbone_torsions


class ParsingException(Exception):
    pass


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
    return pos_heavyatom, mask_heavyatom


def parse_biopython_structure(entity, sasa=None, unknown_threshold=1.0, max_resseq=None):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({'chain_id': [], 'resseq': [], 'icode': [], 'res_nb': [], 'aa': [], 'pos_heavyatom': [], 'mask_heavyatom': [],
                     'phi': [], 'phi_mask': [], 'psi': [], 'psi_mask': [], 'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],})
    tensor_types = {'resseq': torch.LongTensor, 'res_nb': torch.LongTensor, 'aa': torch.LongTensor, 'pos_heavyatom': torch.stack, 'mask_heavyatom': torch.stack,
                    'phi': torch.FloatTensor, 'phi_mask': torch.BoolTensor, 'psi': torch.FloatTensor, 'psi_mask': torch.BoolTensor, 'chi': torch.stack, 'chi_alt': torch.stack,
                    'chi_mask': torch.stack, 'chi_complete': torch.BoolTensor, }
    if sasa is not None and len(sasa) > 0:
        data['sasa'] = []
        tensor_types['sasa'] = torch.LongTensor

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0  # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))  # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resseq_this = int(res.get_id()[1])
            if max_resseq is not None and resseq_this > max_resseq:
                continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK:
                count_unk += 1
                continue

            data.chain_id.append(chain.get_id())    # Chain info
            data.aa.append(restype)  # Residue types. Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # Backbone torsions
            phi, psi, _ = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

            if sasa is not None:
                try:
                    data.sasa.append(sasa[chain.get_id() + resname + str(res.get_id()[1])])
                except:
                    data.sasa.append(0.0)
                    print(f'Warning: no sasa feature for {chain.get_id() + resname + str(res.get_id()[1])}')

    if len(data.aa) == 0:
        raise ParsingException('No parsed residues.')

    if (count_unk / count_aa) >= unknown_threshold:
        raise ParsingException(f'Too many unknown residues, threshold {unknown_threshold:.2f}.')

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    return data, seq_map
