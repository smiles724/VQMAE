import torch
import enum


class CDR(enum.IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    L1 = 4
    L2 = 5
    L3 = 6


class ChothiaCDRRange:
    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    @classmethod
    def to_cdr(cls, chain_type, resseq):
        assert chain_type in ('H', 'L')
        if chain_type == 'H':
            if cls.H1[0] <= resseq <= cls.H1[1]:
                return CDR.H1
            elif cls.H2[0] <= resseq <= cls.H2[1]:
                return CDR.H2
            elif cls.H3[0] <= resseq <= cls.H3[1]:
                return CDR.H3
        elif chain_type == 'L':
            if cls.L1[0] <= resseq <= cls.L1[1]:  # Chothia VH-CDR1
                return CDR.L1
            elif cls.L2[0] <= resseq <= cls.L2[1]:
                return CDR.L2
            elif cls.L3[0] <= resseq <= cls.L3[1]:
                return CDR.L3


class Fragment(enum.IntEnum):
    Heavy = 1
    Light = 2
    Antigen = 3


##
# Residue identities
"""
This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
non_standard_residue_substitutions = {'2AS': 'ASP', '3AH': 'HIS', '5HP': 'GLU', 'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR', 'ALY': 'LYS', 'ARM': 'ARG',
                                      'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'BCS': 'CYS', 'BHD': 'ASP', 'BMT': 'THR', 'BNN': 'ALA',
                                      'BUC': 'CYS', 'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CCS': 'CYS', 'CEA': 'CYS', 'CGU': 'GLU', 'CHG': 'ALA', 'CLE': 'LEU',
                                      'CME': 'CYS', 'CSD': 'ALA', 'CSO': 'CYS', 'CSP': 'CYS', 'CSS': 'CYS', 'CSW': 'CYS', 'CSX': 'CYS', 'CXM': 'MET', 'CY1': 'CYS', 'CY3': 'CYS',
                                      'CYG': 'CYS', 'CYM': 'CYS', 'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN',
                                      'DHA': 'ALA', 'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU', 'DLY': 'LYS', 'DNP': 'ALA', 'DPN': 'PHE', 'DPR': 'PRO', 'DSN': 'SER',
                                      'DSP': 'ASP', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL', 'EFC': 'CYS', 'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU', 'GL3': 'GLY',
                                      'GLZ': 'GLY', 'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA', 'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE', 'HTR': 'TRP',
                                      'HYP': 'PRO', 'IAS': 'ASP', 'IIL': 'ILE', 'IYR': 'TYR', 'KCX': 'LYS', 'LLP': 'LYS', 'LLY': 'LYS', 'LTR': 'TRP', 'LYM': 'LYS', 'LYZ': 'LYS',
                                      'MAA': 'ALA', 'MEN': 'ASN', 'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'LEU', 'MPQ': 'GLY', 'MSA': 'GLY', 'MSE': 'MET', 'MVA': 'VAL', 'NEM': 'HIS',
                                      'NEP': 'HIS', 'NLE': 'LEU', 'NLN': 'LEU', 'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER', 'OCS': 'CYS', 'OMT': 'MET', 'PAQ': 'TYR', 'PCA': 'GLU',
                                      'PEC': 'CYS', 'PHI': 'PHE', 'PHL': 'PHE', 'PR3': 'CYS', 'PRR': 'ALA', 'PTR': 'TYR', 'PYX': 'CYS', 'SAC': 'SER', 'SAR': 'GLY', 'SCH': 'CYS',
                                      'SCS': 'CYS', 'SCY': 'CYS', 'SEL': 'SER', 'SEP': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SOC': 'CYS', 'STY': 'TYR',
                                      'SVA': 'SER', 'TIH': 'ALA', 'TPL': 'TRP', 'TPO': 'THR', 'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR', 'TYI': 'TYR', 'TYQ': 'TYR',
                                      'TYS': 'TYR', 'TYY': 'TYR'}

ressymb_to_resindex = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17,
                       'W': 18, 'Y': 19, 'X': 20, }

three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
                'MSE': 'M',  # this is almost the same AA as MET. The sulfur is just replaced by Selen
                'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S', 'SEC': 'U', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'XLE': 'J'}


class AA(enum.IntEnum):
    ALA = 0
    CYS = 1
    ASP = 2
    GLU = 3
    PHE = 4
    GLY = 5
    HIS = 6
    ILE = 7
    LYS = 8
    LEU = 9
    MET = 10
    ASN = 11
    PRO = 12
    GLN = 13
    ARG = 14
    SER = 15
    THR = 16
    VAL = 17
    TRP = 18
    TYR = 19
    UNK = 20

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:  # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:  # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (value in ressymb_to_resindex) or (value in non_standard_residue_substitutions) or (value in cls._member_names_) or (value in cls._member_map_.values())


num_aa_types = len(AA)


##
# Atom identities
num2name = {0: 'N', 1: 'CA', 2: 'C', 3: 'O', 4: 'CB', 14: 'OXT'}
num2ele = {0: 'N', 1: 'C', 2: 'C', 3: 'O', 4: 'C', 14: 'C'}


class BBHeavyAtom(enum.IntEnum):
    N = 0
    CA = 1
    C = 2
    O = 3
    CB = 4
    OXT = 5


chi_angles_atoms = {AA.ALA: [],  # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
                    AA.ARG: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
                    AA.ASN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], AA.ASP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], AA.CYS: [['N', 'CA', 'CB', 'SG']],
                    AA.GLN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
                    AA.GLU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], AA.GLY: [],
                    AA.HIS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']], AA.ILE: [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
                    AA.LEU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    AA.LYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
                    AA.MET: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']], AA.PHE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    AA.PRO: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']], AA.SER: [['N', 'CA', 'CB', 'OG']], AA.THR: [['N', 'CA', 'CB', 'OG1']],
                    AA.TRP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], AA.TYR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    AA.VAL: [['N', 'CA', 'CB', 'CG1']], }

max_num_heavyatoms = 15
restype_to_heavyatom_names = {AA.ALA: ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', 'OXT'],
                              AA.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', '', 'OXT'],
                              AA.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', '', 'OXT'],
                              AA.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', 'OXT'],
                              AA.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', '', 'OXT'],
                              AA.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', '', 'OXT'],
                              AA.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', 'OXT'],
                              AA.GLY: ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', 'OXT'],
                              AA.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', '', 'OXT'],
                              AA.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', 'OXT'],
                              AA.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', 'OXT'],
                              AA.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', 'OXT'],
                              AA.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', '', 'OXT'],
                              AA.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', 'OXT'],
                              AA.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', 'OXT'],
                              AA.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', '', 'OXT'],
                              AA.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', '', 'OXT'],
                              AA.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
                              AA.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', '', 'OXT'],
                              AA.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '', 'OXT'],
                              AA.UNK: ['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], }
for names in restype_to_heavyatom_names.values(): assert len(names) == max_num_heavyatoms

backbone_atom_coordinates = {AA.ALA: [(-0.525, 1.363, 0.0),  # N
                                      (0.0, 0.0, 0.0),  # CA
                                      (1.526, -0.0, -0.0),  # C
                                      ], AA.ARG: [(-0.524, 1.362, -0.0),  # N
                                                  (0.0, 0.0, 0.0),  # CA
                                                  (1.525, -0.0, -0.0),  # C
                                                  ], AA.ASN: [(-0.536, 1.357, 0.0),  # N
                                                              (0.0, 0.0, 0.0),  # CA
                                                              (1.526, -0.0, -0.0),  # C
                                                              ], AA.ASP: [(-0.525, 1.362, -0.0),  # N
                                                                          (0.0, 0.0, 0.0),  # CA
                                                                          (1.527, 0.0, -0.0),  # C
                                                                          ], AA.CYS: [(-0.522, 1.362, -0.0),  # N
                                                                                      (0.0, 0.0, 0.0),  # CA
                                                                                      (1.524, 0.0, 0.0),  # C
                                                                                      ], AA.GLN: [(-0.526, 1.361, -0.0),  # N
                                                                                                  (0.0, 0.0, 0.0),  # CA
                                                                                                  (1.526, 0.0, 0.0),  # C
                                                                                                  ], AA.GLU: [(-0.528, 1.361, 0.0),  # N
                                                                                                              (0.0, 0.0, 0.0),  # CA
                                                                                                              (1.526, -0.0, -0.0),  # C
                                                                                                              ], AA.GLY: [(-0.572, 1.337, 0.0),  # N
                                                                                                                          (0.0, 0.0, 0.0),  # CA
                                                                                                                          (1.517, -0.0, -0.0),  # C
                                                                                                                          ], AA.HIS: [(-0.527, 1.36, 0.0),  # N
                                                                                                                                      (0.0, 0.0, 0.0),  # CA
                                                                                                                                      (1.525, 0.0, 0.0),  # C
                                                                                                                                      ], AA.ILE: [(-0.493, 1.373, -0.0),  # N
                                                                                                                                                  (0.0, 0.0, 0.0),  # CA
                                                                                                                                                  (1.527, -0.0, -0.0),  # C
                                                                                                                                                  ],
                             AA.LEU: [(-0.52, 1.363, 0.0),  # N
                                      (0.0, 0.0, 0.0),  # CA
                                      (1.525, -0.0, -0.0),  # C
                                      ], AA.LYS: [(-0.526, 1.362, -0.0),  # N
                                                  (0.0, 0.0, 0.0),  # CA
                                                  (1.526, 0.0, 0.0),  # C
                                                  ], AA.MET: [(-0.521, 1.364, -0.0),  # N
                                                              (0.0, 0.0, 0.0),  # CA
                                                              (1.525, 0.0, 0.0),  # C
                                                              ], AA.PHE: [(-0.518, 1.363, 0.0),  # N
                                                                          (0.0, 0.0, 0.0),  # CA
                                                                          (1.524, 0.0, -0.0),  # C
                                                                          ], AA.PRO: [(-0.566, 1.351, -0.0),  # N
                                                                                      (0.0, 0.0, 0.0),  # CA
                                                                                      (1.527, -0.0, 0.0),  # C
                                                                                      ], AA.SER: [(-0.529, 1.36, -0.0),  # N
                                                                                                  (0.0, 0.0, 0.0),  # CA
                                                                                                  (1.525, -0.0, -0.0),  # C
                                                                                                  ], AA.THR: [(-0.517, 1.364, 0.0),  # N
                                                                                                              (0.0, 0.0, 0.0),  # CA
                                                                                                              (1.526, 0.0, -0.0),  # C
                                                                                                              ], AA.TRP: [(-0.521, 1.363, 0.0),  # N
                                                                                                                          (0.0, 0.0, 0.0),  # CA
                                                                                                                          (1.525, -0.0, 0.0),  # C
                                                                                                                          ], AA.TYR: [(-0.522, 1.362, 0.0),  # N
                                                                                                                                      (0.0, 0.0, 0.0),  # CA
                                                                                                                                      (1.524, -0.0, -0.0),  # C
                                                                                                                                      ], AA.VAL: [(-0.494, 1.373, -0.0),  # N
                                                                                                                                                  (0.0, 0.0, 0.0),  # CA
                                                                                                                                                  (1.527, -0.0, -0.0),  # C
                                                                                                                                                  ], }

bb_oxygen_coordinate = {AA.ALA: (2.153, -1.062, 0.0), AA.ARG: (2.151, -1.062, 0.0), AA.ASN: (2.151, -1.062, 0.0), AA.ASP: (2.153, -1.062, 0.0), AA.CYS: (2.149, -1.062, 0.0),
                        AA.GLN: (2.152, -1.062, 0.0), AA.GLU: (2.152, -1.062, 0.0), AA.GLY: (2.143, -1.062, 0.0), AA.HIS: (2.15, -1.063, 0.0), AA.ILE: (2.154, -1.062, 0.0),
                        AA.LEU: (2.15, -1.063, 0.0), AA.LYS: (2.152, -1.062, 0.0), AA.MET: (2.15, -1.062, 0.0), AA.PHE: (2.15, -1.062, 0.0), AA.PRO: (2.148, -1.066, 0.0),
                        AA.SER: (2.151, -1.062, 0.0), AA.THR: (2.152, -1.062, 0.0), AA.TRP: (2.152, -1.062, 0.0), AA.TYR: (2.151, -1.062, 0.0), AA.VAL: (2.154, -1.062, 0.0), }

backbone_atom_coordinates_tensor = torch.zeros([21, 3, 3])
bb_oxygen_coordinate_tensor = torch.zeros([21, 3])


def make_coordinate_tensors():
    for restype, atom_coords in backbone_atom_coordinates.items():
        for atom_id, atom_coord in enumerate(atom_coords):
            backbone_atom_coordinates_tensor[restype][atom_id] = torch.FloatTensor(atom_coord)

    for restype, bb_oxy_coord in bb_oxygen_coordinate.items():
        bb_oxygen_coordinate_tensor[restype] = torch.FloatTensor(bb_oxy_coord)


make_coordinate_tensors()

chi_angles_mask = {AA.ALA: [False, False, False, False],  # ALA
                   AA.ARG: [True, True, True, True],  # ARG
                   AA.ASN: [True, True, False, False],  # ASN
                   AA.ASP: [True, True, False, False],  # ASP
                   AA.CYS: [True, False, False, False],  # CYS
                   AA.GLN: [True, True, True, False],  # GLN
                   AA.GLU: [True, True, True, False],  # GLU
                   AA.GLY: [False, False, False, False],  # GLY
                   AA.HIS: [True, True, False, False],  # HIS
                   AA.ILE: [True, True, False, False],  # ILE
                   AA.LEU: [True, True, False, False],  # LEU
                   AA.LYS: [True, True, True, True],  # LYS
                   AA.MET: [True, True, True, False],  # MET
                   AA.PHE: [True, True, False, False],  # PHE
                   AA.PRO: [True, True, False, False],  # PRO
                   AA.SER: [True, False, False, False],  # SER
                   AA.THR: [True, False, False, False],  # THR
                   AA.TRP: [True, True, False, False],  # TRP
                   AA.TYR: [True, True, False, False],  # TYR
                   AA.VAL: [True, False, False, False],  # VAL
                   AA.UNK: [False, False, False, False],  # UNK
                   }

chi_pi_periodic = {AA.ALA: [False, False, False, False],  # ALA
                   AA.ARG: [False, False, False, False],  # ARG
                   AA.ASN: [False, False, False, False],  # ASN
                   AA.ASP: [False, True, False, False],  # ASP
                   AA.CYS: [False, False, False, False],  # CYS
                   AA.GLN: [False, False, False, False],  # GLN
                   AA.GLU: [False, False, True, False],  # GLU
                   AA.GLY: [False, False, False, False],  # GLY
                   AA.HIS: [False, False, False, False],  # HIS
                   AA.ILE: [False, False, False, False],  # ILE
                   AA.LEU: [False, False, False, False],  # LEU
                   AA.LYS: [False, False, False, False],  # LYS
                   AA.MET: [False, False, False, False],  # MET
                   AA.PHE: [False, True, False, False],  # PHE
                   AA.PRO: [False, False, False, False],  # PRO
                   AA.SER: [False, False, False, False],  # SER
                   AA.THR: [False, False, False, False],  # THR
                   AA.TRP: [False, False, False, False],  # TRP
                   AA.TYR: [False, True, False, False],  # TYR
                   AA.VAL: [False, False, False, False],  # VAL
                   AA.UNK: [False, False, False, False],  # UNK
                   }