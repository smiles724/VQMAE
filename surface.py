import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor

# from plyfile import PlyData, PlyElement
from pykeops.torch.cluster import grid_cluster


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device))
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric. Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """

    if batch is None:  # Single protein case:
        labels = grid_cluster(x, scale).long()
        C = labels.max() + 1

        # We append a "1" to the input vectors, in order to compute both the numerator and denominator of the "average"  fraction in one pass through the data.
        x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        D = x_1.shape[1]
        points = torch.zeros_like(x_1[:C])
        points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
        return (points[:, :-1] / points[:, -1:]).contiguous()

    else:  # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know how to do more elegantly (this type of computation is not super well supported by PyTorch).
        batch_size = torch.max(batch).item() + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))

    return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def soft_distances(x, y, batch_x, batch_y, smoothness=0.01, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
    y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)

    if atomtypes is not None:
        # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i": (N, 6)  -> (N, 1, 1)  (There are 6 atom types)
        atomic_radii = torch.cuda.FloatTensor([170, 110, 152, 155, 180, 190], device=x.device)
        atomic_radii = atomic_radii / atomic_radii.min()
        atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
        smoothness = torch.sum(smoothness * atomtype_radii, dim=1, keepdim=False)  # n_atoms, 1
        smoothness_i = LazyTensor(smoothness[:, None, None])
        mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
        mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
        mean_smoothness = (smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j)  # n_atoms, n_points, 1
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * ((-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)).view(-1)
    else:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(-1)

    return soft_dists


def atoms_to_points_normals(atoms, batch, distance=1.05, smoothness=0.5, resolution=1.0, nits=4, atomtypes=None, sup_sampling=20, variance=0.1):
    """Turns a collection of atoms into an oriented point cloud.

    Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

    Args:
        atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
        batch (integer Tensor): (N,) batch vector, as in PyTorch_geometric. Start from 0!
        distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
        smoothness (float, optional): radii of the atoms, if atom types are
            not provided. Defaults to 0.5.
        resolution (float, optional): side length of the cubic cells in
            the final sub-sampling pass. Defaults to 1.0.
        nits (int, optional): number of iterations . Defaults to 4.
        atomtypes (Tensor, optional): (N,6) one-hot encoding of the atom
            chemical types. Defaults to None.

    Returns:
        (Tensor): (M,3) coordinates for the surface points `x_i`.
        (Tensor): (M,3) unit normals `n_i`.
        (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
    """
    # a) Parameters for the soft distance function and its level set:
    T = distance

    N, D = atoms.shape
    B = sup_sampling  # Sup-sampling ratio

    # Batch vectors:
    batch_atoms = batch
    batch_z = batch[:, None].repeat(1, B).view(N * B)

    # b) Draw N*B points at random in the neighborhood of our atoms
    z = atoms[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atoms)
    z = z.view(-1, D)  # (N*B, D)

    # We don't want to backprop through a full network here!
    atoms = atoms.detach().contiguous()
    z = z.detach().contiguous()

    # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
    with torch.enable_grad():
        if z.is_leaf: z.requires_grad = True

        # c) Iterative loop: gradient descent along the potential ".5 * (dist - T)^2" with respect to the positions z of our samples
        for it in range(nits):
            dists = soft_distances(atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes, )
            Loss = ((dists - T) ** 2).contiguous().sum()
            g = torch.autograd.grad(Loss, z)[0]
            z.data -= 0.5 * g

        # d) Only keep the points which are reasonably close to the level set:
        dists = soft_distances(atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
        margin = (dists - T).abs()
        mask = margin < variance * T

        # d') And remove the points that are trapped *inside* the protein:
        zz = z.detach()
        zz.requires_grad = True
        for it in range(nits):
            dists = soft_distances(atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes, )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, zz)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
            zz = zz + 1.0 * T * normals

        dists = soft_distances(atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
        mask = mask & (dists > 1.5 * T)

        z = z[mask].contiguous().detach()
        batch_z = batch_z[mask].contiguous().detach()

        # e) Subsample the point cloud:
        points, batch_points = subsample(z, batch_z, scale=resolution)

        # f) Compute the normals on this smaller point cloud:
        p = points.detach()
        p.requires_grad = True
        dists = soft_distances(atoms, p, batch_atoms, batch_points, smoothness=smoothness, atomtypes=atomtypes, )
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, p)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
    points = points - 0.5 * normals
    return points.detach(), normals.detach(), batch_points.detach()

from Bio.PDB import *

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}  # 6 atom types in the residue
Tensor = torch.LongTensor
tensor = torch.FloatTensor


def load_seperate_structure(fname, return_map=False, ligand=None, receptor=None):
    """Loads a .pdb to return coordinates and atom types."""
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords_ligand, types_ligand, restypes_ligand = [], [], []
    coords_receptor, types_receptor, restypes_receptor = [], [], []
    chain_ids, resseqs, icodes = [], [], []
    for i, atom in enumerate(atoms):
        if atom.element in ele2num.keys():
            res = atom.parent
            resname = res.get_resname()
            chain_id = res.parent.get_id()
            if chain_id in ligand:
                coords_ligand.append(atom.get_coord())
                types_ligand.append(ele2num[atom.element])
                # restypes_ligand.append(int(AA(resname)))
            elif chain_id in receptor:
                coords_receptor.append(atom.get_coord())
                types_receptor.append(ele2num[atom.element])
                # restypes_receptor.append(int(AA(resname)))
            else:
                continue

    coords_ligand = np.stack(coords_ligand)
    # restypes_ligand = np.stack(restypes_ligand)
    types_ligand_array = np.zeros((len(types_ligand), len(ele2num)))   # one-hot embeddings
    for i, t in enumerate(types_ligand):
        types_ligand_array[i, t] = 1.0

    coords_receptor = np.stack(coords_receptor)
    # restypes_receptor = np.stack(restypes_receptor)
    types_receptor_array = np.zeros((len(types_receptor), len(ele2num)))   # one-hot embeddings
    for i, t in enumerate(types_receptor):
        types_receptor_array[i, t] = 1.0

    seq_map = {}
    return {'ligand': {"xyz": coords_ligand, "types": types_ligand_array, 'restypes': restypes_ligand, 'seq_map': seq_map},
            'receptor': {"xyz": coords_receptor, "types": types_receptor_array, 'restypes': restypes_receptor}}


if __name__ == '__main__':
    import numpy as np
    from biopandas.pdb import PandasPdb

    cut_off = 1.5
    raw_data_path = './'
    pdb_file = ['1h2e_A', ]
    for file in pdb_file:
        # proteins = load_seperate_structure(raw_data_path + f'{file}.pdb', return_map=False, ligand='A', receptor='B')
        # ligand, receptor = proteins['ligand'], proteins['receptor']
        #
        # atomxyz_ligand, atomtypes_ligand, restypes_ligand = tensor(ligand["xyz"]), tensor(ligand["types"]), Tensor(ligand["restypes"])
        # batch_atoms_ligand = torch.zeros(len(atomxyz_ligand)).long().to(atomxyz_ligand.device)
        # # pts_ligand, norms_ligand, _ = atoms_to_points_normals(atomxyz_ligand.cuda(), batch_atoms_ligand.cuda(), atomtypes=atomtypes_ligand.cuda())
        # atomxyz_receptor, atomtypes_receptor, restypes_receptor = tensor(receptor["xyz"]), tensor(receptor["types"]), Tensor(receptor["restypes"])
        # batch_atoms_receptor = torch.zeros(len(atomxyz_receptor)).long().to(atomxyz_receptor.device)
        # for sup in [1, 10, 50]:
        #     pts_ligand, norms_ligand, _ = atoms_to_points_normals(atomxyz_ligand.cuda(), batch_atoms_ligand.cuda(), atomtypes=atomtypes_ligand.cuda(), sup_sampling=sup)
        #     pts_receptor, norms_receptor, _ = atoms_to_points_normals(atomxyz_receptor.cuda(), batch_atoms_receptor.cuda(), atomtypes=atomtypes_receptor.cuda(), sup_sampling=sup)
        #     torch.save({'ligand': [atomxyz_receptor, pts_receptor, norms_receptor], 'receptor': [atomxyz_ligand, pts_ligand, norms_ligand]}, f'surf_{file}_{sup}.pt')

        df = PandasPdb().read_pdb(raw_data_path + f'{file}.pdb').df['ATOM']
        x = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
        batch = torch.zeros(len(x)).long()

        for sup in [50]:
            p, normals, batch_p = atoms_to_points_normals(x, batch, distance=1.05, sup_sampling=sup)
            torch.save([x, p, normals], f'surf_{file}_{sup}.pt')
