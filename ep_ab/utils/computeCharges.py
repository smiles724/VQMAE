from Bio.PDB import *
import numpy as np

from ep_ab.utils.chemistry import (polarHydrogens, acceptorAngleAtom, acceptorPlaneAtom, hbond_std_dev, donorAtom, )


def computeChargeHelper(atom_name, res, res_name, v):
    """ computeCharges.py: Wrapper function to compute hydrogen bond potential (free electrons/protons) in the surface """
    """ https://github.com/LPDI-EPFL/masif/tree/master/source/triangulation """
    """ https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/data_preparation/01-pdb_extract_and_triangulate.py#L22 """

    # Check if it is a polar hydrogen.
    if isPolarHydrogen(atom_name, res_name):
        try:
            donor_atom_name = donorAtom[atom_name]
        except:
            return 0.0
        a = res[donor_atom_name].get_coord()  # N/O
        b = res[atom_name].get_coord()  # H
        # Donor-H is always 180.0 degrees, = pi
        angle_deviation = computeAngleDeviation(a, b, v, np.pi)
        angle_penalty = computeAnglePenalty(angle_deviation)
        return 1.0 * angle_penalty
    # Check if it is an acceptor oxygen or nitrogen
    elif isAcceptorAtom(atom_name, res, res_name):
        try:
            acceptor_atom = res[atom_name]    # "O" may not in residue
            b = acceptor_atom.get_coord()
            a = res[acceptorAngleAtom[atom_name]].get_coord()
        except:
            return 0.0
        # 120 degress for acceptor
        angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
        # TODO: This should not be 120 for all atoms, i.e. for HIS it should be ~125.0
        angle_penalty = computeAnglePenalty(angle_deviation)
        plane_penalty = 1.0
        if atom_name in acceptorPlaneAtom:
            try:
                d = res[acceptorPlaneAtom[atom_name]].get_coord()
            except:
                return 0.0
            plane_deviation = computePlaneDeviation(d, a, b, v)
            plane_penalty = computeAnglePenalty(plane_deviation)
        return -1.0 * angle_penalty * plane_penalty  # Compute the
    return 0.0


# Compute the absolute value of the deviation from theta
def computeAngleDeviation(a, b, c, theta):
    return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)


# Compute the angle deviation from a plane
def computePlaneDeviation(a, b, c, d):
    dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


# angle_deviation from ideal value. TODO: do a more data-based solution
def computeAnglePenalty(angle_deviation):
    # Standard deviation: hbond_std_dev
    return max(0.0, 1.0 - (angle_deviation / hbond_std_dev) ** 2)


def isPolarHydrogen(atom_name, res_name):
    if atom_name in polarHydrogens[res_name]:
        return True
    return False


def isAcceptorAtom(atom_name, res, res_name):
    if atom_name.startswith("O"):
        return True
    elif res_name == "HIS":
        if atom_name == "ND1" and "HD1" not in res:
            return True
        if atom_name == "NE2" and "HE2" not in res:
            return True
    return False


# Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
def computeSatisfied_CO_HN(atoms):
    ns = NeighborSearch(atoms)
    satisfied_CO, satisfied_HN = set(), set()
    for atom1 in atoms:
        res1 = atom1.get_parent()
        if atom1.get_id() == "O":
            neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
            for atom2 in neigh_atoms:
                if atom2.get_id() == "H":
                    res2 = atom2.get_parent()
                    # Ensure they belong to different residues.
                    if res2.get_id() != res1.get_id():
                        # Compute the angle N-H:O, ideal value is 180 (but in
                        # helices it is typically 160) 180 +-30 = pi
                        angle_N_H_O_dev = computeAngleDeviation(res2["N"].get_coord(), atom2.get_coord(), atom1.get_coord(), np.pi, )
                        # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                        angle_H_O_C_dev = computeAngleDeviation(atom2.get_coord(), atom1.get_coord(), res1["C"].get_coord(), 8 * np.pi / 9, )
                        ## Allowed deviations: 30 degrees (pi/6) and 20 degrees (pi/9)
                        if angle_N_H_O_dev - np.pi / 6 < 0 and angle_H_O_C_dev - np.pi / 9 < 0.0:
                            satisfied_CO.add(res1.get_id())
                            satisfied_HN.add(res2.get_id())
    return satisfied_CO, satisfied_HN
