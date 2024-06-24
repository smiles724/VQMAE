import logging
import collections
import math
import random
import os
import pickle
from typing import Mapping, List, Dict, Tuple, Optional

import lmdb
import numpy as np
import torch
from Bio.PDB import PDBExceptions, Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

from ep_ab.utils.convert_pdb2npy import ele2num, num2ele
from ep_ab.utils.geometry import atoms_to_points
from ep_ab.utils.protein.constants import AA
from ep_ab.utils.protein.points import ProteinData
from ep_ab.utils.computeHydro import computeHydrophobicity
from ep_ab.utils.computeCharges import computeSatisfied_CO_HN, computeChargeHelper
Tensor, tensor = torch.LongTensor, torch.FloatTensor
ClusterIdType, PdbCodeType, ChainIdType = str, str, str


def _process_surface(cif_path, structure_id, resolution, sup_sampling, distance):
    surfaces = {'id': structure_id, 'chains': []}
    parser = MMCIFParser(QUIET=True)
    try:
        residues = []
        structure = parser.get_structure(structure_id, cif_path)[0]

        atoms = Selection.unfold_entities(structure, "A")
        satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)

        for chain in structure.get_chains():
            try:
                atom_xyz, atom_types, res_coords, res_types = [], [], [], []
                for i, res in enumerate(chain.get_residues()):
                    resname = res.get_resname()
                    if AA.is_aa(resname) and (res.has_id('CA') and res.has_id('C') and res.has_id('N')):  # skip unknown residues
                        res_types.append(int(AA(resname)))
                        residues.append(res)
                        for atom in res.get_atoms():
                            if atom.element in ele2num.keys():
                                atom_xyz.append(atom.get_coord())
                                atom_types.append(ele2num[atom.element])
                            if atom.name == 'CA':
                                res_coords.append(atom.get_coord())
                atom_xyz, res_coords, res_types = np.stack(atom_xyz), np.stack(res_coords), np.stack(res_types)
                atom_types_array = np.zeros((len(atom_types), len(ele2num)))  # one-hot embeddings
                for i, t in enumerate(atom_types):
                    atom_types_array[i, t] = 1.0

                batch_atoms = torch.zeros(len(atom_xyz)).long()
                atom_xyz, res_coords, res_types, atom_types_array = tensor(atom_xyz), tensor(res_coords), Tensor(res_types), tensor(atom_types_array)
                pts, norms, _ = atoms_to_points(atom_xyz, batch_atoms, atomtypes=atom_types_array, resolution=resolution, sup_sampling=sup_sampling, distance=distance)

                # compute chemical/biological properties
                knn_atom_idx = torch.cdist(pts, atom_xyz).topk(1, dim=1, largest=False)[1].squeeze()
                knn_res_idx = torch.cdist(pts, res_coords).topk(1, dim=1, largest=False)[1].squeeze()
                knn_res_types = res_types[knn_res_idx].tolist()
                hp = computeHydrophobicity(knn_res_types)
                hbond = torch.zeros(len(pts))
                for ix in range(len(pts)):
                    res = residues[knn_res_idx[ix]]
                    res_id = res.get_id()
                    atom_name = num2ele[atom_types[knn_atom_idx[ix]]]
                    res_name = AA(knn_res_types[ix]).name
                    if not (atom_name == 'H' and res_id in satisfied_HN) and not (atom_name == 'O' and res_id in satisfied_CO):
                        hbond[ix] = computeChargeHelper(atom_name, res, res_name, pts[ix])  # Ignore atom if it is BB

                parsed = {'P': ProteinData(xyz=pts, normals=norms, atomxyz=atom_xyz, atomtypes=atom_types_array, resxyz=res_coords, restypes=res_types,
                                           hphobicity=hp, hbond=hbond), 'chain': chain.id}
            except (PDBExceptions.PDBConstructionException, Exception, KeyError, ValueError, FileNotFoundError, EOFError) as e:
                logging.warning('<{}_{}> {}: {}.'.format(structure_id, chain.id, e.__class__.__name__, str(e)))
                continue
            surfaces[structure_id + '_' + chain.id] = parsed  # index by pdbcode + chain id
            surfaces['chains'].append(chain.id)
    except (PDBExceptions.PDBConstructionException, Exception, KeyError, ValueError, FileNotFoundError, EOFError) as e:
        logging.warning('[{}] {}: {}.'.format(structure_id, e.__class__.__name__, str(e)))
    if len(surfaces['chains']) > 0:
        return surfaces
    return None


class PDB_REDO_Dataset(Dataset):
    MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB

    def __init__(self, split, pdbredo_dir, clusters_path, splits_path, processed_dir, num_preprocess_jobs=8, surface=None, reset=False):
        super().__init__()
        self.pdbredo_dir = pdbredo_dir
        self.clusters_path = clusters_path
        self.splits_path = splits_path
        self.processed_dir = processed_dir
        self.num_preprocess_jobs = num_preprocess_jobs
        os.makedirs(processed_dir, exist_ok=True)

        # Load clusters and splits
        self.clusters: Mapping[ClusterIdType, List[Tuple[PdbCodeType, ChainIdType]]] = collections.defaultdict(list)
        self.splits: Mapping[str, List[ClusterIdType]] = collections.defaultdict(list)
        self._load_clusters_and_splits()

        # Load surface
        self.surface = surface
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._preprocess_structures(reset)

        # Sanitize clusters
        self.sanitized_clusters_path = os.path.join(self.processed_dir, 'sanitized_clusters.pkl')
        self._sanitize_clusters(reset)

        # Select clusters of the split
        self._clusters_of_split = [c for c in self.splits[split] if c in self.clusters]

    @property
    def keys_path(self):
        return os.path.join(self.processed_dir, 'keys.pkl')

    @property
    def lmdb_path(self):
        return os.path.join(self.processed_dir, 'surface.lmdb')

    def _load_clusters_and_splits(self):
        with open(self.clusters_path, 'r') as f:
            lines = f.readlines()
        current_cluster = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_cluster = word[1:-1]
                else:
                    pdbcode, chain_id = word.split(':')
                    self.clusters[current_cluster].append((pdbcode, chain_id))

        with open(self.splits_path, 'r') as f:
            lines = f.readlines()
        current_split = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_split = word[1:-1]
                else:
                    self.splits[current_split].append(word)

    def _sanitize_clusters(self, reset):
        if os.path.exists(self.sanitized_clusters_path) and not reset:
            with open(self.sanitized_clusters_path, 'rb') as f:
                self.clusters = pickle.load(f)
            return

        # Step 1: Find structures and chains that do not exist in PDB_REDO
        clusters_raw = self.clusters
        pdbcode_to_chains: Dict[PdbCodeType, List[ChainIdType]] = collections.defaultdict(list)
        for pdbcode, pdbchain_list in clusters_raw.items():
            for pdbcode, chain in pdbchain_list:
                pdbcode_to_chains[pdbcode].append(chain)

        pdb_removed, chain_removed = 0, 0
        pdbcode_to_chains_ok = {}
        self._connect_db()
        for pdbcode, chain_list in tqdm(pdbcode_to_chains.items(), desc='Sanitize'):
            if pdbcode not in self.db_keys:
                pdb_removed += 1
                continue
            data = self._get_from_db(pdbcode)
            ch_exists = []
            for ch in chain_list:
                if ch in data['chains']:
                    ch_exists.append(ch)
                else:
                    chain_removed += 1
            if len(ch_exists) > 0:
                pdbcode_to_chains_ok[pdbcode] = ch_exists
            else:
                pdb_removed += 1

        print(f'[INFO] Structures removed: {pdb_removed}. Chains removed: {chain_removed}.')
        pdbchains_allowed = set((p, c) for p, clist in pdbcode_to_chains_ok.items() for c in clist)

        # Step 2: Rebuild the clusters according to the allowed chains.
        pdbchain_to_clust = {}
        for clust_name, pdbchain_list in clusters_raw.items():
            for pdbchain in pdbchain_list:
                if pdbchain in pdbchains_allowed:
                    pdbchain_to_clust[pdbchain] = clust_name

        clusters_sanitized = collections.defaultdict(list)
        for pdbchain, clust_name in pdbchain_to_clust.items():
            clusters_sanitized[clust_name].append(pdbchain)

        print('[INFO] %d clusters after sanitization (from %d).' % (len(clusters_sanitized), len(clusters_raw)))
        with open(self.sanitized_clusters_path, 'wb') as f:
            pickle.dump(clusters_sanitized, f)
        self.clusters = clusters_sanitized

    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(self.lmdb_path, map_size=self.MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)

    def get_all_pdbcodes(self):
        pdbcodes = set()
        for _, pdbchain_list in self.clusters.items():
            for pdbcode, _ in pdbchain_list:
                pdbcodes.add(pdbcode)
        return pdbcodes

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset:
            return
        pdbcodes = self.get_all_pdbcodes()
        tasks, n_missing = [], 0
        for pdbcode in pdbcodes:
            cif_path = os.path.join(self.pdbredo_dir, f"{pdbcode}_final.cif")
            if os.path.exists(cif_path):
                tasks.append(delayed(_process_surface)(cif_path, pdbcode, self.surface.resolution, self.surface.sup_sampling, self.surface.distance))
            else:
                n_missing += 1
        print(f'[WARNING] {n_missing} / {len(pdbcodes)} CIF not found.')

        # Split data into chunks
        chunk_size = 8192
        task_chunks = [tasks[i * chunk_size:(i + 1) * chunk_size] for i in range(math.ceil(len(tasks) / chunk_size))]

        db_conn = lmdb.open(self.lmdb_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False, )  # Establish database connection
        keys = []
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(task for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}"))
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    key = data['id']
                    keys.append(key)
                    txn.put(key=key.encode(), value=pickle.dumps(data))
                    stored += 1
                print(f"[INFO] {stored} processed for chunk#{i + 1}")
        db_conn.close()
        print(f'Loading {len(keys)} complex surfaces successfully. ')

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

    def _get_from_db(self, pdbcode):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdbcode.encode()))  # Made a copy
        return data

    def __len__(self):
        return len(self._clusters_of_split)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index, None)

        # Select cluster
        clust = self._clusters_of_split[index[0]]
        pdbchain_list = self.clusters[clust]

        # Select a pdb-chain from the cluster and retrieve the data point
        if index[1] is None:
            pdbcode, chain = random.choice(pdbchain_list)
        else:
            pdbcode, chain = pdbchain_list[index[1]]
        data = self._get_from_db(pdbcode)[pdbcode + '_' + chain]  # Made a copy
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--pdbredo_dir', type=str, default='./data/pdb_redo/cif')
    parser.add_argument('--clusters_path', type=str, default='./data/pdbredo_clusters.txt')
    parser.add_argument('--splits_path', type=str, default='./data/pdbredo_splits.txt')
    parser.add_argument('--processed_dir', type=str, default='./')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()

    surface = {'resolution': 1.0, 'distance': 1.05, 'variance': 0.1, 'sup_sampling': 10}
    dataset = PDB_REDO_Dataset(args.split, pdbredo_dir=args.pdbredo_dir, clusters_path=args.clusters_path, splits_path=args.splits_path,
                               processed_dir=args.processed_dir, surface=surface, reset=args.reset, )
    print(dataset[0])
