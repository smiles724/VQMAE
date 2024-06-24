import os
import tqdm
import pickle
# Python
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

# Core Includes
from rosetta.core.select.movemap import *

# Protocol Includes
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax

pyrosetta.init()

# Import a pose
ag_dir = './data/processed/pdb_ag/'
ag_relax_dir = './data/processed/pdb_ag_relax/'
os.makedirs(ag_relax_dir, exist_ok=True)

bad_idx = []
energy_before, energy_after = [], []
ag_tqdm = tqdm.tqdm(os.listdir(ag_dir))
for ag_pdb in ag_tqdm:
    idx = ag_pdb.split('.')[0]

    if not os.path.exists(os.path.join(ag_relax_dir, f'{idx}_relax.pdb')):
        try:
            original_pose = pose_from_pdb(os.path.join(ag_dir, ag_pdb))
            pose = original_pose.clone()

            scorefxn = get_score_function()
            # before = scorefxn.score(original_pose)

            # set a scorefunction
            fr = FastRelax()
            fr.set_scorefxn(scorefxn)

            # decrease the amount of minimization cycles.
            # This is generally only recommended for cartesian, but we just want everything to run fast at the moment.
            fr.max_iter(2)   # 100 too long, change to 2 or 3 steps
            # Skip for tests
            if not os.getenv("DEBUG"):
                fr.apply(pose)

            pose.dump_pdb(os.path.join(ag_relax_dir, f'{idx}_relax.pdb'))
            ag_tqdm.set_description(f"relaxed {scorefxn(pose):.2f}, original {scorefxn(original_pose):.2f}, delta {scorefxn(pose) - scorefxn(original_pose)} ", )
            energy_before.append(scorefxn(original_pose))
            energy_after.append(scorefxn(pose))

        except:
            bad_idx.append(idx)

print(f'{len(ag_dir) - len(bad_idx)} succeed, {len(bad_idx)} failed.')
with open('./energy_change', 'wb') as f:
    pickle.dump([energy_before, energy_after], f)
