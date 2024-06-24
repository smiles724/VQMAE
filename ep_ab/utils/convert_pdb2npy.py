import numpy as np
from Bio.PDB import *
from tqdm import tqdm
from ep_ab.utils.protein.constants import AA

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}  # 6 atom types in the residue
num2ele = {0: 'C', 1: 'H', 2: 'O', 3: 'N', 4: 'S', 5: 'SE'}


def load_complex_structure(fname, return_map=False, ligand=None, receptor=None):
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    residues = structure.get_residues()

    atom_coords_ligand, atom_types_ligand, res_coords_ligand, res_types_ligand = [], [], [], []
    atom_coords_receptor, atom_types_receptor, res_coords_receptor, res_types_receptor = [], [], [], []
    chain_ids, resseqs, icodes = [], [], []
    for i, res in enumerate(residues):
        resname = res.get_resname()
        chain_id = res.parent.get_id()
        if AA.is_aa(resname) and resname != 'UNK' and (res.has_id('CA') and res.has_id('C') and res.has_id('N')):  # skip unknown residues
            if chain_id in ligand:
                res_types_ligand.append(int(AA(resname)))
                for atom in res.get_atoms():
                    if atom.element in ele2num.keys():
                        atom_coords_ligand.append(atom.get_coord())
                        atom_types_ligand.append(ele2num[atom.element])
                    if atom.name == 'CA':
                        res_coords_ligand.append(atom.get_coord())

            elif chain_id in receptor:
                res_types_receptor.append(int(AA(resname)))
                for atom in res.get_atoms():
                    if atom.element in ele2num.keys():
                        atom_coords_receptor.append(atom.get_coord())
                        atom_types_receptor.append(ele2num[atom.element])
                    if atom.name == 'CA':
                        res_coords_receptor.append(atom.get_coord())
            else:
                continue

            if return_map and chain_id in ligand:
                resseq_this = int(res.get_id()[1])
                icode_this = res.get_id()[2]
                chain_ids.append(chain_id)
                resseqs.append(resseq_this)
                icodes.append(icode_this)
    assert len(res_coords_ligand) == len(res_types_ligand) and len(res_coords_receptor) == len(res_types_receptor)

    seq_map = {}
    atom_coords_ligand = np.stack(atom_coords_ligand)
    res_coords_ligand = np.stack(res_coords_ligand)
    res_types_ligand = np.stack(res_types_ligand)
    atom_coords_receptor = np.stack(atom_coords_receptor)
    res_coords_receptor = np.stack(res_coords_receptor)
    res_types_receptor = np.stack(res_types_receptor)
    atom_types_ligand_array = np.zeros((len(atom_types_ligand), len(ele2num)))  # one-hot embeddings
    for i, t in enumerate(atom_types_ligand):
        atom_types_ligand_array[i, t] = 1.0
    atom_types_receptor_array = np.zeros((len(atom_types_receptor), len(ele2num)))  # one-hot embeddings
    for i, t in enumerate(atom_types_receptor):
        atom_types_receptor_array[i, t] = 1.0

    if return_map:
        for i, (chain_id, resseq, icode) in enumerate(zip(chain_ids, resseqs, icodes)):
            if (chain_id, resseq, icode) in seq_map.keys():
                print(f'Warning: repeated item of {(chain_id, resseq, icode)}')
                continue
            seq_map[(chain_id, resseq, icode)] = i

    ligand = {"atom_xyz": atom_coords_ligand, "atom_types": atom_types_ligand_array, 'res_xyz': res_coords_ligand, 'res_types': res_types_ligand}
    receptor = {"atom_xyz": atom_coords_receptor, "atom_types": atom_types_receptor_array, 'res_xyz': res_coords_receptor, 'res_types': res_types_receptor}
    return ligand, receptor


def load_structure(fname, return_map=False):
    """Loads a .pdb to return atom coordinates, atom types, and residue types."""
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    residues = structure.get_residues()

    atom_coords, atom_types, res_coords, res_types, atom_res = [], [], [], [], []
    chain_ids, resseqs, icodes = [], [], []
    for i, res in enumerate(residues):
        resname = res.get_resname()
        if AA.is_aa(resname) and resname != 'UNK' and (res.has_id('CA') and res.has_id('C') and res.has_id('N')):        # skip unknown residues
            res_types.append(int(AA(resname)))

            for atom in res.get_atoms():
                if atom.element in ele2num.keys():
                    atom_coords.append(atom.get_coord())
                    atom_types.append(ele2num[atom.element])
                    atom_res.append(i)
                if atom.name == 'CA':
                    res_coords.append(atom.get_coord())

            if return_map:
                chain_id = res.parent.get_id()
                resseq_this = int(res.get_id()[1])
                icode_this = res.get_id()[2]
                chain_ids.append(chain_id)
                resseqs.append(resseq_this)
                icodes.append(icode_this)

    seq_map = {}
    atom_coords = np.stack(atom_coords)
    atom_res = np.stack(atom_res)
    res_coords = np.stack(res_coords)
    res_types = np.stack(res_types)
    atom_types_array = np.zeros((len(atom_types), len(ele2num)))   # one-hot embeddings
    for i, t in enumerate(atom_types):
        atom_types_array[i, t] = 1.0

    if return_map:
        for i, (chain_id, resseq, icode) in enumerate(zip(chain_ids, resseqs, icodes)):
            if (chain_id, resseq, icode) in seq_map.keys():
                print(f'Warning: repeated item of {(chain_id, resseq, icode)}')
                continue
            seq_map[(chain_id, resseq, icode)] = i
    return {"atom_xyz": atom_coords, "atom_types": atom_types_array, 'res_xyz': res_coords, 'res_types': res_types, 'seq_map': seq_map, 'atom_res': atom_res}


def convert_pdbs(pdb_dir, npy_dir):
    print("Converting PDBs")
    for p in tqdm(pdb_dir.glob("*.pdb")):
        protein = load_structure(p)
        np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
