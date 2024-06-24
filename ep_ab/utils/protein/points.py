import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data

tensor, inttensor = torch.FloatTensor, torch.LongTensor


def numpy(x):
    return x.detach().cpu().numpy()


class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __call__(self, data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.atom_xyz = torch.matmul(R1, data.atom_xyz.T).T
        data.xyz = torch.matmul(R1, data.xyz.T).T
        data.normals = torch.matmul(R1, data.normals.T).T

        data.atom_xyz_p2 = torch.matmul(R2, data.atom_xyz_p2.T).T
        data.xyz_p2 = torch.matmul(R2, data.xyz_p2.T).T
        data.normals_p2 = torch.matmul(R2, data.normals_p2.T).T

        data.rand_rot1 = R1
        data.rand_rot2 = R2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures(object):
    r"""Centers a protein"""

    def __call__(self, data):
        pb_upper = 3.0
        pb_lower = -3.0

        chem = data.chemical_features
        chem_p2 = data.chemical_features_p2

        pb = chem[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb = chem[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp = chem[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb = torch.clamp(pb, pb_lower, pb_upper)
        pb = (pb - pb_lower) / (pb_upper - pb_lower)
        pb = 2 * pb - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp = hp / 4.5
        hp_p2 = hp_p2 / 4.5

        data.chemical_features = torch.stack([pb, hb, hp]).T
        data.chemical_features_p2 = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class ProteinPairData(Data):
    def __init__(self, xyz_ligand=None, normals_ligand=None, atomxyz_ligand=None, atomtypes_ligand=None, resxyz_ligand=None, restypes_ligand=None, intf_ligand=None,
                 xyz_receptor=None, normals_receptor=None, atomxyz_receptor=None, atomtypes_receptor=None, resxyz_receptor=None, restypes_receptor=None, intf_receptor=None):
        super().__init__()
        """ https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html """
        self.xyz_ligand = xyz_ligand
        self.normals_ligand = normals_ligand
        self.atomxyz_ligand = atomxyz_ligand
        self.atomtypes_ligand = atomtypes_ligand
        self.resxyz_ligand = resxyz_ligand
        self.restypes_ligand = restypes_ligand
        self.intf_ligand = intf_ligand

        self.xyz_receptor = xyz_receptor
        self.normals_receptor = normals_receptor
        self.atomxyz_receptor = atomxyz_receptor
        self.atomtypes_receptor = atomtypes_receptor
        self.resxyz_receptor = resxyz_receptor
        self.restypes_receptor = restypes_receptor
        self.intf_receptor = intf_receptor


class ProteinData(Data):
    def __init__(self, xyz=None, normals=None, atomxyz=None, atomtypes=None, resxyz=None, restypes=None, face=None, chemical_features=None, iface_labels=None,
                 center_location=None, hphobicity=None, charges=None, hbond=None):
        super().__init__()
        self.xyz, self.normals = xyz, normals
        self.atomxyz, self.atomtypes = atomxyz, atomtypes
        self.resxyz, self.restypes = resxyz, restypes
        self.hphobicity, self.charges, self.hbond = hphobicity, charges, hbond

        self.face = face
        self.chemical_features = chemical_features
        self.iface_labels = iface_labels
        self.center_location = center_location

        self.restypes_mut = None

    def __inc__(self, key, value, *args, **kwargs):  # https://blog.csdn.net/tagagi/article/details/125374881
        if key == "face":
            return self.xyz.size(0)
        return super(ProteinData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if ("index" in key) or ("face" in key):
            return 1
        return 0


def load_protein(pdb_id, data_dir, no_pre_compute=True):
    """Loads a protein structures mesh and its features"""
    # Load the data, and read the connectivity information:
    triangles = (None if no_pre_compute else inttensor(np.load(os.path.join(data_dir, pdb_id + "_triangles.npy"), allow_pickle=True)).T)
    points = None if no_pre_compute else tensor(np.load(os.path.join(data_dir, pdb_id + "_xyz.npy")))
    iface_labels = (None if no_pre_compute else tensor(np.load(os.path.join(data_dir, pdb_id + "_iface_labels.npy")).reshape((-1, 1))))  # Interface labels
    chemical_features = (None if no_pre_compute else tensor(np.load(os.path.join(data_dir, pdb_id + "_features.npy"))))  # Features
    normals = (None if no_pre_compute else tensor(np.load(os.path.join(data_dir, pdb_id + "_normals.npy"))))  # Normals
    center_location = None if no_pre_compute else torch.mean(points, axis=0, keepdims=True)

    atom_xyz = tensor(np.load(os.path.join(data_dir, pdb_id + "_atomxyz.npy")))
    atomtypes = tensor(np.load(os.path.join(data_dir, pdb_id + "_atomtypes.npy")))
    protein_data = ProteinData(xyz=points, face=triangles, chemical_features=chemical_features, iface_labels=iface_labels, normals=normals, center_location=center_location,
                               atom_xyz=atom_xyz, atomtypes=atomtypes, )
    return protein_data
