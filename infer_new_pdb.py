import argparse

import torch.utils.tensorboard
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from Bio import PDB

from ep_ab.utils.transforms import get_transform
from ep_ab.models import get_model
from ep_ab.utils.misc import *
from ep_ab.utils.train import *
from ep_ab.utils.protein import parsers
from ep_ab.utils.protein.constants import AA, three_to_one, non_standard_residue_substitutions


def design_for_pdb(args):
    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.train.seed)

    # Structure loading
    data_id = os.path.basename(args.pdb_path)
    print(f'Loading data {data_id} from {args.pdb_path}')
    pdb_path = args.pdb_path
    data = {'id': data_id, 'heavy': None, 'heavy_seqmap': None, 'light': None, 'light_seqmap': None, 'antigen': None, 'antigen_seqmap': None, }
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(data_id, pdb_path)[0]  # ValueError: invalid literal for int() with base 10: 'X'
    if args.ag_chain is None:
        chains = [c for c in structure.child_list]  # use all chains
    else:
        chains = [structure[c] for c in args.ag_chain]
    data['antigen'], data['antigen_seqmap'] = parsers.parse_biopython_structure(chains)
    if config.dataset.test.use_plm:
        seq = ''
        for x in data['antigen']['aa'].numpy().tolist():
            if AA(x).name not in three_to_one.keys():
                seq += three_to_one[non_standard_residue_substitutions[AA(x).name]]
            else:
                seq += three_to_one[AA(x).name]

        import esm
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()

        batch_labels, batch_strs, batch_tokens = batch_converter([("protein1", seq), ])
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)    # Extract per-residue representations (on CPU)
        data['antigen']['plm_feature'] = results["representations"][33][0, 1: -1]

    transform = get_transform(config.dataset.test.transform) if 'transform' in config.dataset.test else None
    data = transform(data)

    # Load checkpoint and model
    print('Loading model cfg and checkpoints: %s' % args.ckpt)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg_ckpt = ckpt['cfg']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    print(str(lsd))

    pos, aa, resseq, res_nb, chain_nb = [], [], [], [], []
    torch.set_grad_enabled(False)
    model.eval()
    loader = DataLoader([data], batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())
    for batch in loader:
        res_feat = model.encode(recursive_to(batch, args.device))
        ep_pred = model.classifier(res_feat).squeeze(-1)[0].cpu().numpy()   # save to b-factor in PDB

        pos = batch['pos_heavyatom'][0].cpu().numpy()
        aa = batch['aa'][0].cpu().numpy()
        resseq = batch['resseq'][0].cpu().numpy()
        chain_nb = batch['chain_nb'][0].cpu().numpy()

    atoms_ = ['N', 'CA', 'C', 'O', 'CB']
    symbols_ = ['N', 'C', 'C', 'O', 'C']
    template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
    chains_ = [c.id for c in structure.child_list] if args.ag_chain is None else args.ag_chain.split()

    atom_number = 0
    with open(args.out_path, 'w') as f:
        for i in range(len(pos)):
            try:
                resname = AA(aa[i])._name_
            except:   # padding token break
                break

            for j in range(5):
                atom_number += 1
                xyz = pos[i][j].tolist()
                f.write(template.format("ATOM", atom_number, atoms_[j], '', resname, chains_[chain_nb[i]], resseq[i], '', xyz[0], xyz[1], xyz[2], 1.00, ep_pred[i], symbols_[j], ''))
    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, default='./1alu.pdb')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--ag_chain', type=str, default=None, help='Chain id of the antigen.')
    parser.add_argument('-c', '--cfg', type=str, default='./configs/train/epitope_pred.yml')
    parser.add_argument('-o', '--out_path', type=str, default='./epitope_pred.pdb')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()
    design_for_pdb(args)
