import biotite.structure as struc
from biotite.structure import AtomArray
import biotite.structure.io.pdb as pdb
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import numpy as np
from scipy.stats import circvar
from typing import Tuple, Union
from .align_loops import length_independent_dihedral_label

def get_atom_array(entry: pd.Series, data_dir: str) -> AtomArray:
    pdb_file = os.path.join(data_dir, entry['pdb_file'])
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"File {pdb_file} does not exist.")
    loaded_pdb_file = pdb.PDBFile.read(pdb_file)
    atom_array = loaded_pdb_file.get_structure(model=1)
    atom_array = atom_array[atom_array.chain_id == entry.chain_id]
    atom_array = atom_array[np.isin(atom_array.res_id, entry.residues)]
    return atom_array

def get_atom_array_parallel(df: pd.DataFrame, data_dir: str, num_workers: int=8) -> list:
    with Pool(num_workers) as pool:
        atom_arrays = pool.starmap(get_atom_array, [(entry, data_dir) for _, entry in df.iterrows()])
    return atom_arrays
    

def superimpose_structural_homologs(reference_atom_array: AtomArray, subject_atom_array: AtomArray) -> tuple:
    # Superimpose the structures using the biotite library
    try:
        superimposed, _, ref_indices, sub_indices = struc.superimpose_structural_homologs(
            reference_atom_array, subject_atom_array, max_iterations=1
        )
    except ValueError as e:
        if "No anchors found, the structures are too dissimilar" in str(e):
            return None, None, None, None
    return superimposed, _, ref_indices, sub_indices

def get_tm_score(reference_atom_array: AtomArray, subject_atom_array: AtomArray, reference_length: str ='shorter') -> float:
    """
    Calculate the TM-score using the biotite library
    reference_length = 'shorter' or 'longer' means the score is symmetric 
    get_tm_score(atom_array1, atom_array2) == get_tm_score(atom_array2, atom_array1)

    Parameters
    ----------
    reference_atom_array : AtomArray
        The reference structure.
    subject_atom_array : AtomArray
        The subject structure.
    reference_length : str
        The length of the reference structure. Can be 'shorter' or 'longer' or 'reference'.
        Default is 'shorter'.
    
    Returns
    -------
    float
        The TM-score between the reference and subject structures.
        0.0 if the structures are too dissimilar and cannot be superimposed.
    """
    superimposed, _, ref_indices, sub_indices = superimpose_structural_homologs(
        reference_atom_array, subject_atom_array
    )
    if superimposed is None:
        return 0.0
    return struc.tm_score(reference_atom_array, superimposed, ref_indices, sub_indices, reference_length=reference_length)

def get_rmsd(reference_atom_array: AtomArray, subject_atom_array: AtomArray) -> float:
    """
    Calculate the RMSD using the biotite library

    Parameters
    ----------
    reference_atom_array : AtomArray
        The reference structure.
    subject_atom_array : AtomArray
        The subject structure.
    
    Returns
    -------
    float
        The TM-score between the reference and subject structures.
        0.0 if the structures are too dissimilar and cannot be superimposed.
    """
    superimposed, _, ref_indices, sub_indices = superimpose_structural_homologs(
        reference_atom_array, subject_atom_array
    )
    if superimposed is None:
        return 0.0
    return struc.rmsd(reference_atom_array, superimposed, ref_indices, sub_indices)


def get_structure_alignment_score(entry1: pd.Series, entry2: pd.Series, data_dir: str, metric: str = 'tm_score') -> float:
    if metric not in ['rmsd', 'tm_score']:
        raise ValueError("Metric must be either 'rmsd' or 'tm_score'")
    atom_array1 = get_atom_array(entry1, data_dir)
    atom_array2 = get_atom_array(entry2, data_dir)
    if metric == 'rmsd':
        return get_rmsd(atom_array1, atom_array2)
    return get_tm_score(atom_array1, atom_array2)


def get_structure_alignment_score_all_pairs(df: pd.DataFrame, data_dir: str, num_workers: int = 8, metric: str = 'tm_score') -> pd.DataFrame:
    # Calculate the RMSD or TM-score for each entry in the DataFrame
    if metric not in ['rmsd', 'tm_score']:
        raise ValueError("Metric must be either 'rmsd' or 'tm_score'")

    loop_ids = list(df['loop_id'])

    # IO takes a long time, so first calculate the atom arrays for all entries
    atom_arrays = get_atom_array_parallel(df, data_dir, num_workers)

    indices = range(len(df))
    entry_index_pairs = [(i, j) for i in indices for j in indices if i < j]
    score_func = get_rmsd if metric == 'rmsd' else get_tm_score
    with Pool(num_workers) as pool:
        scores = pool.starmap(score_func, [(atom_arrays[index1], atom_arrays[index2]) for index1, index2 in entry_index_pairs])

    return scores, [(loop_ids[index1], loop_ids[index2]) for index1, index2 in entry_index_pairs]


def _to_minus_pi_pi(angles, high, low):
    """
    Convert `angles` that are given in the range [`low`, `high`] to the
    canonical range [−π, +π] – exactly the same transformation that is used
    inside `scipy.stats.circ*`.
    """
    return (angles - low) * 2.0 * np.pi / (high - low) - np.pi


def _masked_circvar(a, m):
    """
    Circular variance along axis-0 of `a` (shape (n, d)) respecting *True* in
    the boolean mask `m` (same shape).
    Returned shape: (d,).

    Formula (see scipy.stats._circfuncs):
        C = Σ cos(θ) / N
        S = Σ sin(θ) / N
        R = √(C² + S²)
        var = 1 − R
    """
    # number of valid observations per column
    n = m.sum(axis=0).astype(float)

    # avoid division by zero; keep NaN where n == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        C = (np.cos(a) * m).sum(axis=0) / n
        S = (np.sin(a) * m).sum(axis=0) / n
        R = np.sqrt(C * C + S * S)
        var = 1.0 - R

    # n == 0  →  var = nan
    var[n == 0] = np.nan
    return var


def angle_circ_variance(A: np.ndarray,
                        mask: np.ndarray | None = None,
                        high: float = np.pi,
                        low: float = -np.pi,
                        average_over_res: bool = True) -> np.ndarray:
    """
    Circular variance of the dihedral angles in a set of structures while
    honouring an (optional) mask that selects which residues of which
    structures are taken into account.

    Parameters
    ----------
    A : ndarray, shape (n, d, 3)
        Rows = structures, columns organised as
        [ φ1 … φd | ψ1 … ψd | ω1 … ωd ].
    mask : ndarray, shape (n, d), bool, optional
        `True` → the residue is used, `False` → it is ignored.
        If ``None`` every residue is used.
    high, low : float, optional
        End-points of the angular range (`scipy.stats.circvar` convention).
        Default: −π .. +π
    average_over_res : bool, optional
        If True (default) the result is averaged over the d residues,
        giving one variance for φ, one for ψ and one for ω.
        If False the full (3, d) array is returned.

    Returns
    -------
    var : ndarray
        Shape (3,) if `average_over_res` is True, otherwise (3, d).
        Order: φ, ψ, ω.
    """
    if A.ndim != 3 or A.shape[2] != 3:
        raise ValueError(
            "A must have shape (n, d, 3) with the last dimension "
            "being (φ, ψ, ω); got {}".format(A.shape)
        )

    n, d, _ = A.shape

    # ------------------------------------------------------------------
    # Build / check mask
    # ------------------------------------------------------------------
    if mask is None:
        mask = np.ones((n, d), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n, d):
            raise ValueError(
                f"mask must have shape {(n, d)}, got {mask.shape}"
            )

    # bring angles to [−π, +π]
    A_std = _to_minus_pi_pi(A, high, low)          # (n, d, 3)

    # ------------------------------------------------------------------
    # Compute variance for each of the three angle types separately
    # ------------------------------------------------------------------
    # result will finally be (3, d)
    var = np.empty((3, d), dtype=float)

    # φ, ψ, ω → index 0, 1, 2 in the last dimension
    for i in range(3):
        # slice shape: (n, d)
        ang_i = A_std[:, :, i]
        var[i] = _masked_circvar(ang_i, mask)

    # ------------------------------------------------------------------
    # Average over the residues if requested
    # ------------------------------------------------------------------
    if average_over_res:
        # use nanmean so residues that were entirely masked become invisible
        var = np.nanmean(var, axis=-1)   # shape (3,)

    return var


def dihedral_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise dihedral distance between two sets of structures.

    Parameters
    ----------
    A : ndarray, shape (n, d, 3)
    B : ndarray, shape (m, d, 3)

    Returns
    -------
    D : ndarray, shape (n, m)
        D[i, j] = < 2(1 - cos(φ_i - φ_j)) >_d
                + < 2(1 - cos(ψ_i - ψ_j)) >_d
                + < 2(1 - cos(ω_i - ω_j)) >_d
        averaged first over the d residues and then over the three
        angle types (φ, ψ, ω).
    """
    n, d_A, three_A = A.shape
    m, d_B, three_B = B.shape
    assert three_A == three_B == 3, "Input arrays must have shape (n, 3, d) and (m, 3, d)"
    if d_A != d_B:
        raise ValueError("Input arrays must have the same number of residues (d_A == d_B)")
    A = A.transpose(0, 2, 1)  # (n, 3, d)
    B = B.transpose(0, 2, 1)  # (m, 3, d)

    # result: (n, m, 3, d)
    diff = A[:, None, :, :] - B[None, :, :, :]

    D = 2 * (1 - np.cos(diff))      # (n, m, 3, d)
    D = D.mean(axis=-1).mean(axis=-1)   # average over d, then over (φ,ψ,ω)
    return D


def dihedral_distance_pairwise(A: np.ndarray,
                               mask: np.ndarray | None = None,
                               *,  # force keyword-only arguments after this
                               nan_if_empty: bool = True) -> np.ndarray:
    """
    Pairwise dihedral distance between all structures in `A`
    with optional masking of individual residues.

    Parameters
    ----------
    A
        ndarray, shape (n, d, 3);  angles must be in radians.
    mask
        ndarray, shape (n, d), bool.
        mask[i, k] == True  ->  residue k of structure i is used.
        If ``None`` everything is used.
    nan_if_empty
        If a pair (i, j) has no common un-masked residues,
        return NaN (default) or 4.0.

    Returns
    -------
    D : ndarray, shape (n, n)
        D[i, j] = ⟨ 2(1−cos(φ_i−φ_j)) ⟩_d
                + ⟨ 2(1−cos(ψ_i−ψ_j)) ⟩_d
                + ⟨ 2(1−cos(ω_i−ω_j)) ⟩_d
        averaged first over the **un-masked** residues,
        then over the three angle types (φ, ψ, ω).
    """
    if A.ndim != 3 or A.shape[-1] != 3:
        raise ValueError("A must have shape (n, d, 3)")

    n, d, three = A.shape        # three == 3, asserted above

    # ---------------------------
    # 1.  Prepare the mask
    # ---------------------------
    if mask is None:
        mask = np.ones((n, d), dtype=bool)
    if mask.shape != (n, d):
        raise ValueError(f"mask must have shape ({n}, {d}), got {mask.shape}")

    # valid_mask[i,j,k] == residue k is present in both i and j
    valid_mask = mask[:, None, :] & mask[None, :, :]        # (n, n, d)

    # --------------------------------------
    # 2.  raw dihedral‐metric for every pair
    # --------------------------------------
    # diff: (n, n, d, 3)   →  transpose to (n, n, 3, d) later
    diff = A[:, None, :, :] - A[None, :, :, :]
    diff = diff.transpose(0, 1, 3, 2)                       # (n, n, 3, d)

    Dij = 2.0 * (1.0 - np.cos(diff))                        # (n, n, 3, d)

    # ------------------------------------------
    # 3.  apply mask and average over dimensions
    # ------------------------------------------
    # multiply by mask, sum over residues, then divide by the count
    residue_counts = valid_mask.sum(axis=-1)                # (n, n)

    # avoid division by zero; keep a copy for later
    counts_safe = np.where(residue_counts == 0, 1, residue_counts)

    # broadcast mask to (n, n, 1, d) so it matches Dij
    Dij_masked_sum = (Dij * valid_mask[:, :, None, :]).sum(axis=-1)   # (n, n, 3)
    Dij_mean_per_res = Dij_masked_sum / counts_safe[:, :, None]       # (n, n, 3)

    # average over the 3 dihedral types
    D = Dij_mean_per_res.mean(axis=-1)                                # (n, n)

    # --------------------------------------------------
    # 4.  Handle pairs with zero common residues (opt.)
    # --------------------------------------------------
    if nan_if_empty:
        D = np.where(residue_counts == 0, np.nan, D)
    else:
        D = np.where(residue_counts == 0, 4.0, D)

    return D


def eval_cluster(input: Tuple[np.ndarray, Union[np.ndarray, None]], mean: bool = False):
    A = input[0]  # angles, shape (n, d, 3)
    mask = input[1]  # mask, shape (n, d) or None
    if A.shape[0] == 1: # if only one structure, return 1.0
        return 1.0, np.zeros(3)
    if mean:
        if mask is not None:
            raise NotImplementedError("mean=True is not supported with a mask")
        # how many structures are within 0.47 of the mean
        D = dihedral_distance(A, A.mean(axis=0, keepdims=True))
    else:
        # how many structures are within 0.47 of the mean
        D = dihedral_distance_pairwise(A, mask=mask)
        upper_tri_mask = np.triu(np.ones(D.shape), k=1).astype(bool)
        D = D[upper_tri_mask] # upper triangle
    proportion_within_threshold = np.mean((D < 0.47))
    return proportion_within_threshold, angle_circ_variance(A, mask=mask, average_over_res=True)


def eval_clusters(angles: np.ndarray, cluster_index: np.ndarray, mask: np.ndarray = None, num_workers: int = 8, return_mean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    unique_clusters, counts = np.unique(cluster_index, return_counts=True)
    if mask is None:
        with Pool(num_workers) as pool:
            results = pool.map(eval_cluster, [(angles[cluster_index == cluster], None) for cluster in unique_clusters])
    else:
        with Pool(num_workers) as pool:
            results = pool.map(eval_cluster, [(angles[cluster_index == cluster], mask[cluster_index==cluster]) for cluster in unique_clusters])
    
    proportion_within_threshold = np.array([result[0] for result in results])
    circular_variance = np.array([result[1] for result in results])

    if not return_mean:
        return proportion_within_threshold, circular_variance

    weighted_proportion_within_threshold = np.sum(proportion_within_threshold * counts) / np.sum(counts)
    weighted_circular_variance = np.sum(circular_variance * counts[:, np.newaxis], axis=0) / np.sum(counts) # (3,) one for each angle
    
    return float(weighted_proportion_within_threshold), *weighted_circular_variance.tolist()


def eval_cluster_length_independent(input):
    angles, loop_coords, stem_coords, mask, max_residue_difference, run_alignment = input
    D = dihedral_distance_pairwise(angles, mask=mask)

    loop_lengths = np.sum(mask, axis=1)
    length_difference = np.abs(loop_lengths[:, None] - loop_lengths[None, :])
    if np.any(length_difference > 0): # if loops differ in length by more than 1 residue
        D[length_difference > max_residue_difference] = 4.0  # max dihedral distance
        if run_alignment:
            row_indices, col_indices = np.where(length_difference <= max_residue_difference)
            for r, c in zip(row_indices, col_indices):
                if r > c: 
                    continue # upper triangle only
                d = length_independent_dihedral_label(
                    angles1=angles[r][mask[r]],
                    angles2=angles[c][mask[c]],
                    loop_coords1=loop_coords[r][mask[r]],
                    loop_coords2=loop_coords[c][mask[c]],
                    stem_coords1=stem_coords[r],
                    stem_coords2=stem_coords[c],
                )
                D[r, c] = D[c, r] = d  # fill both directions

    upper_tri_mask = np.triu(np.ones(D.shape), k=1).astype(bool)
    D = D[upper_tri_mask] # upper triangle
    proportion_within_threshold = np.mean((D < 0.47))
    circular_variance = angle_circ_variance(angles, mask=mask, average_over_res=True)
    return proportion_within_threshold, circular_variance


def eval_clusters_length_independent(angles: np.ndarray, loop_coords: np.ndarray, stem_coords: np.ndarray,
                                     cluster_index: np.ndarray, mask: np.ndarray,
                                     num_workers: int = 8, run_alignment: bool = True, return_mean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    # mask, shape (n,d). True for residues, False for special tokens.
    unique_clusters, counts = np.unique(cluster_index, return_counts=True)
    unique_clusters = unique_clusters[counts > 1]  # only consider clusters with more than one structure
    counts = counts[counts > 1]  # only consider clusters with more than one structure

    args = []
    for cluster in unique_clusters:
        cluster_mask = cluster_index == cluster
        args.append((angles[cluster_mask], loop_coords[cluster_mask], stem_coords[cluster_mask], mask[cluster_mask], 1, run_alignment))

    with Pool(num_workers) as pool:
        results = pool.map(eval_cluster_length_independent, args)
    
    proportion_within_threshold = np.array([result[0] for result in results])
    circular_variance = np.array([result[1] for result in results])

    if not return_mean:
        return proportion_within_threshold, circular_variance

    weighted_proportion_within_threshold = np.sum(proportion_within_threshold * counts) / np.sum(counts)
    weighted_circular_variance = np.sum(circular_variance * counts[:, np.newaxis], axis=0) / np.sum(counts) # (3,) one for each angle
    
    return float(weighted_proportion_within_threshold), *weighted_circular_variance.tolist()


if __name__ == "__main__":
    # test usage of mask for padded residues
    angles = np.random.rand(10, 5, 3)  # 10 structures, 5 residues, 3 angles
    angles_padded = np.concatenate([angles, np.zeros((10, 3, 3))], axis=1)  # add 3 padded residues
    mask = angles_padded[:, :, 0] != 0
    dihedral_distance1 = dihedral_distance_pairwise(angles_padded, mask=mask)
    dihedral_distance2 = dihedral_distance(angles, angles)
    dihedral_distance3 = dihedral_distance_pairwise(angles, mask=None)
    assert np.allclose(dihedral_distance1, dihedral_distance2), "dihedral_distance_pairwise does not match dihedral_distance"
    assert np.allclose(dihedral_distance1, dihedral_distance3), "dihedral_distance_pairwise does not match dihedral_distance with no mask"

    angles = np.random.rand(10, 8, 3)  # 10 structures, 8 residues, 3 angles
    angles_padded = np.concatenate([angles, np.zeros((10, 3, 3))], axis=1)  # add 3 padded residues
    mask = angles_padded[:, :, 0] != 0
    clusters = np.random.randint(0, 2, size=(10,))  # example cluster indices

    proportion1, circ_var1 = eval_clusters(angles, clusters, return_mean=False)
    proportion2, circ_var2 = eval_clusters(angles_padded, clusters, mask=mask, return_mean=False)
    assert np.isclose(proportion1, proportion2).all(), "eval_clusters does not match with padded angles"
    assert np.allclose(circ_var1, circ_var2), "circular variance does not match with padded angles"

    proportion1, phi1, psi1, omega1 = eval_clusters(angles, clusters, return_mean=True)
    proportion2, phi2, psi2, omega2 = eval_clusters(angles_padded, clusters, mask=mask, return_mean=True)
    assert np.isclose(proportion1, proportion2), "eval_clusters does not match with padded angles"
    assert np.allclose(phi1, phi2), "phi circular variance does not match"
    assert np.allclose(psi1, psi2), "psi circular variance does not match"
    assert np.allclose(omega1, omega2), "omega circular variance does not match"