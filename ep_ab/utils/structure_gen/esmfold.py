"""
script to generate antigen sequences using ESM-FOLD
pip install "fair-esm[esmfold]" -i https://pypi.tuna.tsinghua.edu.cn/simple
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install pyproject-toml   # for OPENFOLD
git cfg --global http.sslVerify false   # https://stackoverflow.com/questions/68801315/gnutls-handshake-failed-the-tls-connection-was-non-properly-terminated-while
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


Or use https://github.com/facebookresearch/esm/environment.yml
conda create --name esm
conda env create -f environment.yml
"""
import pickle

import esm
import lmdb
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from ep_ab.utils.misc import *
from ep_ab.utils.protein.constants import AA, three_to_one, non_standard_residue_substitutions


def get_seq(aa_list, chain_nb_list=None):
    seq = ''
    for i, x in enumerate(aa_list):
        if chain_nb_list is not None:
            if i > 0 and chain_nb_list[i] != chain_i:
                seq += ':'  # separator for multimer
            chain_i = chain_nb_list[i]
        if AA(x).name not in three_to_one.keys():
            seq += three_to_one[non_standard_residue_substitutions[AA(x).name]]
        else:
            seq += three_to_one[AA(x).name]
    return seq


processed_dir = './data/processed'
_structure_cache_path = os.path.join(processed_dir, 'structures.lmdb')
_entry_cache_path = os.path.join(processed_dir, 'entry')
fasta_cache = os.path.join(processed_dir, 'sequences_esmfold.fasta')
if not os.path.exists(fasta_cache):
    with open(_structure_cache_path + '-ids', 'rb') as f:
        db_ids = pickle.load(f)
    with open(_entry_cache_path, 'rb') as f:
        sabdab_entries = pickle.load(f)
    sabdab_entries = list(filter(lambda e: e['id'] in db_ids, sabdab_entries))
    db_conn = lmdb.open(_structure_cache_path, map_size=32 * (1024 * 1024 * 1024), create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )

    seqs, ids = [], []
    with db_conn.begin() as txn:
        for entry in sabdab_entries:
            idx = entry['id']
            data = pickle.loads(txn.get(idx.encode()))['antigen']

            # Multimer prediction can be done with chains separated by ':'
            seq = get_seq(data['aa'].numpy().tolist(), data['chain_id'])

            if idx not in ids:         # 7mtb_H_L_G duplicate
                seqs.append(seq)
                ids.append(idx)

    # https://github.com/gcorso/DiffDock/blob/main/datasets/pdbbind_lm_embedding_preparation.py
    records = []
    for (index, seq) in zip(ids, seqs):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ''
        records.append(record)
    SeqIO.write(records, fasta_cache, "fasta")
    print("End extracting FAST. Please continue to generate ESM foldings.")


fasta_sequences = SeqIO.parse(open(fasta_cache), 'fasta')
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
model.set_chunk_size(128)

_esmfold_path = os.path.join(processed_dir, 'pdb_ag_esmfold/')
os.makedirs(_esmfold_path, exist_ok=True)

fasta_sequences = [i for i in fasta_sequences]
for fasta in tqdm(fasta_sequences):
    name, sequence = fasta.id, str(fasta.seq)
    if not os.path.exists(os.path.join(_esmfold_path, f"{name}_esmfold.pdb")):
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)
        except:
            print(f'{name} failed due to OOM.')
            continue

        with open(os.path.join(_esmfold_path, f"{name}_esmfold.pdb"), "w") as f:
            f.write(output)
