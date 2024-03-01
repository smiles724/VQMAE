import argparse
import pathlib
import shutil
import pickle

import lmdb
from esm import FastaBatchedDataset, pretrained, MSATransformer
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from ep_ab.utils.misc import *
from ep_ab.utils.structure_gen.esmfold import get_seq


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_cache)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches)
    print(f"Read {fasta_cache} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {layer: t[i, 1: truncate_len + 1].clone() for layer, t in representations.items()}
                if "mean" in args.include:
                    result["mean_representations"] = {layer: t[i, 1: truncate_len + 1].mean(0).clone() for layer, t in representations.items()}
                if "bos" in args.include:
                    result["bos_representations"] = {layer: t[i, 0].clone() for layer, t in representations.items()}
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(result, args.output_file, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
                                     )
    parser.add_argument('cfg', type=str)
    parser.add_argument("model_location", type=str, help="PyTorch model file OR name of pretrained model to download (see README for models)", )
    parser.add_argument("output_dir", type=pathlib.Path, help="output directory for extracted representations", )
    parser.add_argument("ab_ag", type=str, default='antigen', choices=['partner', 'antigen'])

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument("--repr_layers", type=int, default=[-1], nargs="+", help="layers indices from which to extract representations (0 to num_layers, inclusive)", )
    parser.add_argument("--include", type=str, nargs="+", choices=["mean", "per_tok", "bos", "contacts"], help="specify which representations to return", required=True, )
    parser.add_argument("--truncation_seq_length", type=int, default=1022, help="truncate sequences longer than the given value", )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()

    config, _ = load_config(args.config)
    seed_all(config.train.seed)

    # run train.py first to preprocess the dataset
    processed_dir = './data/processed'
    _structure_cache_path = os.path.join(processed_dir, 'structures.lmdb')
    _entry_cache_path = os.path.join(processed_dir, 'entry')
    fasta_cache = os.path.join(processed_dir, 'sequences_esm.fasta')
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
                data = pickle.loads(txn.get(idx.encode()))[args.ab_ag]   # select antigen or partner
                seq = get_seq(data['aa'].numpy().tolist())

                if idx not in ids:  # 7mtb_H_L_G duplicate
                    seqs.append(seq)
                    ids.append(idx)

        # https://github.com/gcorso/DiffDock/blob/main/datasets/pdbbind_lm_embedding_preparation.py
        records = []
        for (index, seq) in zip(ids, seqs):
            record = SeqRecord(Seq(seq), str(index))
            record.description = ''
            records.append(record)
        SeqIO.write(records, fasta_cache, "fasta")
        print("End extracting FAST. Please continue to generate ESM embeddings.")

    run(args)
    output_path = os.path.join(processed_dir, f'{args.ab_ag}_esm2_embeddings.pt')
    dict = {}
    for filename in tqdm(os.listdir(args.output_dir)):
        dict[filename.split('.')[0]] = torch.load(os.path.join(args.output_dir, filename))['representations'][33]
    torch.save(dict, output_path)
    shutil.rmtree(args.output_dir)
