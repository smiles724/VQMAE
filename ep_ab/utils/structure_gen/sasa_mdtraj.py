import pickle
import sys
import mdtraj as mdj  # pip install mdtraj -i https://pypi.tuna.tsinghua.edu.cn/simple

pdb_path = sys.argv[1]
save_path = sys.argv[2]
# print(pdb_path)
# print(save_path)

# Solvent Accessible Surface Area (SASA)
mdj_pdb = mdj.load_pdb(pdb_path)
# shrake_rupley would quit if two atoms are too close to each other
sasa = mdj.shrake_rupley(mdj_pdb, mode='residue')[0]
res = mdj_pdb._topology._residues

sasa_dict = {}
for i in range(len(sasa)):
    sasa_dict[res[i].chain.chain_id + str(res[i])] = sasa[i]

with open(save_path, 'wb') as f:
    pickle.dump(sasa_dict, f)
# print(len(sasa_dict))
