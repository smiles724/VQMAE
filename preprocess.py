import datetime
import logging
import os
import sys
import pickle
import subprocess

import joblib
import lmdb
import pandas as pd
import torch
from Bio import PDB
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from Bio.PDB import Select
from Bio.PDB.PDBIO import PDBIO
from tqdm.auto import tqdm

from ep_ab.utils.protein import constants, parsers


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    return val


def nan_to_none(val):
    if val != val or not val:
        return None
    return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _preprocess_sabdab_entries(path):
    df = pd.read_csv(path, sep='\t')
    entries_all = []
    for i, row in tqdm(df.iterrows(), dynamic_ncols=True, desc='Loading entries', total=len(df), ):
        ag_chains = split_sabdab_delimited_str(nan_to_empty_string(row['antigen_chain']))
        resolution = parse_sabdab_resolution(row['resolution'])
        ag_type, ag_name = nan_to_none(row['antigen_type']), nan_to_none(row['antigen_name'])
        H_chain, L_chain = nan_to_none(row['Hchain']), nan_to_none(row['Lchain'])

        # Filtering, heavy and light chains both exist
        if ag_type not in ALLOWED_AG_TYPES or resolution is None or resolution > RESOLUTION_THRESHOLD or H_chain is None or L_chain is None or len(ag_chains) == 0:
            continue

        if H_chain in ag_chains or L_chain in ag_chains:
            print(f'Heavy chain {H_chain} or light chain {L_chain} overlaps with antigen chains {ag_chains}.')
            continue

        entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(pdbcode=row['pdb'], H=nan_to_empty_string(row['Hchain']), L=nan_to_empty_string(row['Lchain']), Ag=''.join(ag_chains))
        entry = {'id': entry_id, 'pdbcode': row['pdb'], 'H_chain': H_chain, 'L_chain': L_chain, 'ag_chains': ag_chains, 'ag_type': ag_type, 'ag_name': ag_name,
                 'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'), 'resolution': resolution, 'method': row['method'], 'scfv': row['scfv'], }
        entries_all.append(entry)

    with open(_entry_cache_path, 'wb') as f:
        pickle.dump(entries_all, f)


def preprocess_sabdab_structure(task, ag_len_threshold=20, cutoff=10.0, sasa=False):
    entry, pdb_path = task['entry'], task['pdb_path']
    parser = PDB.PDBParser(QUIET=True)
    parsed = {'id': entry['id'], 'heavy': None, 'light': None, 'antigen': None, 'antigen_seqmap': None, }
    try:
        model = parser.get_structure(entry['id'], pdb_path)[0]               # ValueError: invalid literal for int() with base 10: 'X'
        if len(entry['ag_chains']) == 0 or entry['H_chain'] is None or entry['L_chain'] is None:
            raise ValueError(f'Missing antigen, H-chain or L-chain.')

        sasa_dict = {}
        if sasa:
            for c in entry['ag_chains']:
                with open(task['sasa_path'] + '_' + c, 'rb') as f:
                    sasa_dict.update(pickle.load(f))

        chains = [model[c] for c in entry['ag_chains']]
        parsed['antigen'], parsed['antigen_seqmap'] = parsers.parse_biopython_structure(chains, sasa=sasa_dict)

        if len(parsed['antigen']['aa']) < ag_len_threshold:                 # filter too small antigens
            raise ValueError(f"Antigen too short with length {len(parsed['antigen']['aa'])} < {ag_len_threshold}")

        parsed['antigen']['epitope'] = torch.zeros_like(parsed['antigen']['aa'])  # initialize epitope label
        parsed['heavy'] = parsers.parse_biopython_structure(model[entry['H_chain']])[0]     # Chothia, end of Heavy chain Fv
        parsed['light'] = parsers.parse_biopython_structure(model[entry['L_chain']])[0]     # Chothia, end of Light chain Fv
        ab_pos_heavyatom = torch.cat([parsed['heavy']['pos_heavyatom'], parsed['light']['pos_heavyatom']])
        parsed['antigen'] = _label_epitope(parsed['antigen'], ab_pos_heavyatom, cutoff)
        if parsed['antigen']['epitope'].sum() == 0:
            raise ValueError(f'No epitope found in cutoff == {cutoff}.')

    except (PDBExceptions.PDBConstructionException, parsers.ParsingException, KeyError, ValueError, FileNotFoundError, EOFError) as e:
        logging.warning('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))
        return None
    return parsed


def _label_epitope(parsed_antigen, ab_pos_heavyatom, cutoff):
    antigen_ca = parsed_antigen['pos_heavyatom'][:, constants.BBHeavyAtom.CA]    # use CA to compute distance
    ab_ca = ab_pos_heavyatom[:, constants.BBHeavyAtom.CA]
    dist = torch.cdist(antigen_ca, ab_ca)
    dist_indicator = torch.min(dist, dim=-1)[0] < cutoff
    parsed_antigen['epitope'][dist_indicator] = 1  # not += 1
    return parsed_antigen


def _preprocess_structures(path, relax_or_pred=False):
    if not relax_or_pred:
        data_list = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(preprocess_sabdab_structure)(task, sasa=True) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))
    else:
        data_list = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(preprocess_relax_or_pred_sabdab_structure)(task) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))

    ids = []
    db_conn = lmdb.open(path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False, )
    with db_conn.begin(write=True, buffers=True) as txn:
        for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
            if data is not None:
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
    with open(path + '-ids', 'wb') as f:
        pickle.dump(ids, f)
    print(f'Loading {len(ids)} complexes successfully with Biopython. ')


def separate_sabdab_structure(task):
    entry, pdb_path = task['entry'], task['pdb_path']

    if not os.path.exists(os.path.join(ag_dir, f'{entry["id"]}_ag.pdb')):
        parser = PDB.PDBParser(QUIET=True)
        try:
            model = parser.get_structure(entry['id'], pdb_path)[0]  # ValueError: invalid literal for int() with base 10: 'X'
            if len(entry['ag_chains']) == 0 or entry['H_chain'] is None or entry['L_chain'] is None:
                raise ValueError(f'Missing antigen, H-chain or L-chain.')

            class ChainSelect(Select):
                def accept_chain(self, chain):
                    return True if chain.get_id() in entry['ag_chains'] else False

            io = PDBIO()
            io.set_structure(model)
            io.save(os.path.join(ag_dir, f'{entry["id"]}_ag.pdb'), ChainSelect())
        except (PDBExceptions.PDBConstructionException, parsers.ParsingException, KeyError, ValueError,) as e:
            print('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))


def preprocess_relax_or_pred_sabdab_structure(task, sasa=False):
    entry, pdb_path = task['entry'], task['pdb_ag_path']
    if len(pdb_path) == 0: return None
    parser = PDB.PDBParser(QUIET=True)
    parsed = {'id': entry['id'], 'heavy': None, 'heavy_seqmap': None, 'light': None, 'light_seqmap': None, 'antigen': None, 'antigen_seqmap': None, }

    try:
        model = parser.get_structure(entry['id'], pdb_path)[0]     # ValueError: invalid literal for int() with base 10: 'X' | empty file after Rosetta relaxation
        assert len(entry['ag_chains']) > 0, 'No antigen chain is found.'

        sasa_dict = None
        if sasa:
            subprocess.run(["python", "sasa_feature.py", f"{pdb_path}", f"{task['sasa_path']}"])
            with open(task['sasa_path'], 'rb') as f:
                sasa_dict = pickle.load(f)

        chains = [i for i in model.child_list]   # remaining chains all belong to antigen
        parsed['antigen'], parsed['antigen_seqmap'] = parsers.parse_biopython_structure(chains, sasa=sasa_dict)

    except (PDBExceptions.PDBConstructionException, parsers.ParsingException, KeyError, ValueError, FileNotFoundError) as e:
        logging.warning('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))
        return None
    return parsed


if __name__ == '__main__':
    try:
        reset = sys.argv[1]
    except:
        reset = False
    RESOLUTION_THRESHOLD = 4.0
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB
    n_jobs = max(joblib.cpu_count() // 2, 1)
    ALLOWED_AG_TYPES = {'protein', 'protein | protein', 'protein | protein | protein', 'protein | protein | protein | protein | protein', 'protein | protein | protein | protein', }
    echo = False

    # data path root
    processed_dir = './data/processed'
    chothia_dir = '../all_structures/chothia'
    _entry_cache_path = os.path.join(processed_dir, 'entry')
    if not os.path.exists(chothia_dir):
        raise FileNotFoundError(f"SAbDab structures not found in {chothia_dir}. Please download from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/")

    # preprocess entries
    if not os.path.exists(_entry_cache_path) or reset:
        print('Start preprocessing SAbDab entries...')
        summary_path = '../sabdab_summary_all.tsv'
        _preprocess_sabdab_entries(summary_path)
    with open(_entry_cache_path, 'rb') as f:
        sabdab_entries = pickle.load(f)

    tasks, pdbs = [], []
    sasa_path = os.path.join(processed_dir, 'SASA')
    os.makedirs(sasa_path, exist_ok=True)
    for entry in sabdab_entries:
        pdb_path = os.path.join(chothia_dir, '{}.pdb'.format(entry['pdbcode']))   # different entries can use the same PDB
        if not os.path.exists(pdb_path):
            continue
        pdbs.append(entry['pdbcode'])
        tasks.append({'id': entry['id'], 'entry': entry, 'pdb_path': pdb_path, 'sasa_path': os.path.join(sasa_path, f'{entry["pdbcode"]}')})
    print(f'Load {len(tasks)} entries from {len(set(pdbs))} unique PDBs successfully.')

    # preprocess complex structures
    _structure_cache_path = os.path.join(processed_dir, 'structures.lmdb')
    if not os.path.exists(_structure_cache_path) or reset:
        print('Start preprocessing SAbDab structures...')
        if os.path.exists(_structure_cache_path):
            os.unlink(_structure_cache_path)
        _preprocess_structures(_structure_cache_path)
        print('Done processing SAbDab structures successfully.')

    # separate the complex
    ag_dir = os.path.join(processed_dir, './pdb_ag/')
    os.makedirs(ag_dir, exist_ok=True)
    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(separate_sabdab_structure)(task) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))
    print('Finish separating antigen structures. Please relax_struct them by running relax_struct.py!')

    # preprocess relaxed antigen structures
    _structure_relax_cache_path = os.path.join(processed_dir, 'structures_relax.lmdb')
    if not os.path.exists(_structure_relax_cache_path) or reset:
        print('Staring preprocess relaxed pdb files...')
        if os.path.exists(_structure_relax_cache_path):
            os.unlink(_structure_relax_cache_path)
        ag_relax_dir = os.path.join(processed_dir, 'pdb_ag_relax/')
        for task in tasks:
            pdb_ag_relax_path = os.path.join(ag_relax_dir, '{}_ag_relax.pdb'.format(task['id']))
            if not os.path.exists(pdb_ag_relax_path):
                if echo: logging.warning(f"PDB not found: {pdb_ag_relax_path}")
                pdb_ag_relax_path = ''
            task['pdb_ag_path'] = pdb_ag_relax_path
            # task['sasa_path'] = os.path.join(sasa_path, f'{task["entry"]["pdbcode"]}')
        print('Finish relaxed structure filtering...')
        _preprocess_structures(_structure_relax_cache_path, relax_or_pred=True)

    # preprocess predicted antigen structures
    _esmfold_path = os.path.join(processed_dir, 'pdb_ag_esmfold/')
    _esmfold_relax_cache_path = os.path.join(processed_dir, 'structures_esmfold.lmdb')
    if not os.path.exists(_esmfold_relax_cache_path) or reset:
        print('Staring preprocess relaxed pdb files...')
        for task in tasks:
            pdb_ag_esmfold_path = os.path.join(_esmfold_path, '{}_esmfold.pdb'.format(task['id']))
            if not os.path.exists(pdb_ag_esmfold_path):
                if echo: logging.warning(f"PDB not found: {pdb_ag_esmfold_path}")
                pdb_ag_esmfold_path = ''
            task['pdb_ag_path'] = pdb_ag_esmfold_path
            # task['sasa_path'] = os.path.join(sasa_path, f'{task["entry"]["pdbcode"]}')
        print('Finish relaxed structure filtering...')
        _preprocess_structures(_esmfold_relax_cache_path, relax_or_pred=True)

