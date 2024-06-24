import os
from subprocess import Popen, PIPE

import numpy
from default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin

"""
Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
https://github.com/LPDI-EPFL/masif/blob/master/source/triangulation/computeAPBS.py
"""


def computeAPBS(vertices, pdb_file, tmp_file_base):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """
    fields = tmp_file_base.split("/")[0:-1]
    directory = "/".join(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]
    args = [pdb2pqr_bin, "--ff=parse", "--whitespace", "--noopt", "--apbs-input", pdbname, filename_base, ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    args = [apbs_bin, filename_base + ".in"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    vertfile = open(directory + "/" + filename_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    args = [multivalue_bin, filename_base + ".csv", filename_base + ".dx", filename_base + "_out.csv", ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])

    remove_fn = os.path.join(directory, filename_base)
    os.remove(remove_fn)
    os.remove(remove_fn + '.csv')
    os.remove(remove_fn + '.dx')
    os.remove(remove_fn + '.in')
    os.remove(remove_fn + '-input.p')
    os.remove(remove_fn + '_out.csv')

    return charges


if __name__ == '__main__':
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)
