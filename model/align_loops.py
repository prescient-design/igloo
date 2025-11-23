import numpy 
import pandas as pd
import numpy as np
import dtaidistance.dtw_ndim as dtw
from typing import Tuple, List, Union

def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd

def align_loops(loop_coords1: np.ndarray, loop_coords2: np.ndarray,
                stem_coords1: np.ndarray, stem_coords2: np.ndarray,
                max_residue_difference: int = 1) -> Union[List[Tuple[int, int]], None]:
    """
    Align two loops using DTW and return the distance.
    Length of loop1 and loop2 should differ by at most k residues.

    1. The stems (+/- 5 amino acids) are aligned with Kabsch algorithm.
    2. The loops are transformed to aligned coordinates using the rotation and translation from aligning the stems.
    3. The DTW warp path is computed between the aligned loops - because we use DTW, it repeats a 
       residue in the shorter loop if necessary (rather than adding a gap).
    
    :param loop_coords1: Calpha coordinates of the first loop [n,3].
    :param loop_coords2: Calpha coordinates of the second loop [m,3].
    :param stem_coords1: Calpha coordinates of the stem for the first loop [Nstem=5, 3].
    :param stem_coords2: Calpha coordinates of the stem for the second loop [Nstem=5, 3].
    :return: mapping of indices from loop1 to loop2 [max(n,m), 2], or None if the loops cannot be aligned.
    """
    if abs(loop_coords1.shape[0] - loop_coords2.shape[0]) > max_residue_difference:
        raise ValueError(f"Loops differ in length by more than {max_residue_difference} residues.")
    if loop_coords1.shape[1] != 3 or loop_coords2.shape[1] != 3:
        raise ValueError("Loop coordinates must be of shape [n, 3].")
    if stem_coords1.shape[1] != 3 or stem_coords2.shape[1] != 3:
        raise ValueError("Stem coordinates must be of shape [Nstem, 3].")
    if stem_coords1.shape[0] != stem_coords2.shape[0]:
        raise ValueError("Stem coordinates must have the same number of residues.")

    # Center the stem coordinates around the origin
    centroid1 = np.mean(stem_coords1, axis=0)
    centroid2 = np.mean(stem_coords2, axis=0)
    stem_coords1_centered = stem_coords1 - centroid1
    stem_coords2_centered = stem_coords2 - centroid2
    
    # Apply Kabsch algorithm to align the stems
    R, t, rmsd_stem = kabsch_numpy(stem_coords1_centered, stem_coords2_centered)
    # assert np.allclose(t, 0.0, atol=1e-4), f"coordinates are not centered around the origin for Kabsch algorithm, t={t}"

    if rmsd_stem > 1.0: # if the stems cannot be aligned within 1 Angstrom, return None
        return None 
    
    # Transform the loop coordinates using the rotation and translation from the stem alignment
    loop_coords1_aligned = np.dot(loop_coords1 - centroid1, R.T) + centroid2 + t
    loop_coords2_aligned = loop_coords2

    # Compute the DTW warp path between the aligned loops
    dtw_path = dtw.warping_path(loop_coords1_aligned, loop_coords2_aligned)
    dtw_path = np.array(dtw_path)

    loop_distances = np.linalg.norm(loop_coords1_aligned[dtw_path[:, 0]] - loop_coords2_aligned[dtw_path[:, 1]], axis=1).mean()
    if loop_distances > 2.0: # if the loops cannot be aligned within 2 Angstrom, return None
        return None
    return dtw_path


def length_independent_dihedral_label(
        angles1: np.ndarray, angles2: np.ndarray, # N x 3
        loop_coords1: np.ndarray, loop_coords2: np.ndarray,
        stem_coords1: np.ndarray, stem_coords2: np.ndarray,
        max_residue_difference: int = 1) -> float:
    dtw_path = align_loops(loop_coords1, loop_coords2, stem_coords1, stem_coords2, max_residue_difference)
    if dtw_path is None:
        return 4.0 # max dihedral distance
    angles1 = angles1[dtw_path[:, 0]]
    angles2 = angles2[dtw_path[:, 1]]
    diff = angles1-angles2 # N x 3
    D = 2 * (1-np.cos(diff.T)) # 3 x N
    D = D.mean(axis=1).mean() # average over the N residues then the dihedral angles
    return D


if __name__ == "__main__":
    # reproducing the length independent alignment in the Nowak et al. paper, Fig 1B
    # https://www.tandfonline.com/doi/full/10.1080/19420862.2016.1158370#d1e413

    from metrics import dihedral_distance

    df = pd.read_parquet("/data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.parquet")

    index1 = 93163 # 13759_4jo2_IM.pdb, L1
    index2 = 57438 # 8409_3lhp_IM.pdb, L1
    print(df.iloc[index1])
    print(df.iloc[index2])
    loop_coords1 = np.array(df.iloc[index1]["c_alpha_atoms"].tolist())
    loop_coords2 = np.array(df.iloc[index2]["c_alpha_atoms"].tolist())

    stem_coords1 = np.array(df.iloc[index1]["stem_c_alpha_atoms"].tolist())
    stem_coords2 = np.array(df.iloc[index2]["stem_c_alpha_atoms"].tolist())

    dtw_path = align_loops(loop_coords1, loop_coords2, stem_coords1, stem_coords2)
    seq1 = df.iloc[index1]["loop_sequence"]
    seq2 = df.iloc[index2]["loop_sequence"]
    print("Aligned sequences:")
    print("".join([seq1[i] for i in dtw_path[:, 0]]))
    print("".join([seq2[i] for i in dtw_path[:, 1]]))

    phi1 = np.array(df.iloc[index1]["phi"].tolist())
    phi2 = np.array(df.iloc[index2]["phi"].tolist())
    psi1 = np.array(df.iloc[index1]["psi"].tolist())
    psi2 = np.array(df.iloc[index2]["psi"].tolist())
    omega1 = np.array(df.iloc[index1]["omega"].tolist())
    omega2 = np.array(df.iloc[index2]["omega"].tolist())

    angles1 = np.stack((phi1, psi1, omega1), axis=1)[dtw_path[:, 0]]
    angles2 = np.stack((phi2, psi2, omega2), axis=1)[dtw_path[:, 1]]

    d = dihedral_distance(angles1[np.newaxis, :, :], angles2[np.newaxis, :, :])
    print(f"Dihedral distance: {d[0][0]:3g}")

    d = length_independent_dihedral_label(
        np.stack((phi1, psi1, omega1), axis=1),
        np.stack((phi2, psi2, omega2), axis=1),
        loop_coords1, loop_coords2,
        stem_coords1, stem_coords2
    )
    print(f"Length independent dihedral distance: {d:3g}")

    d1 = length_independent_dihedral_label(
        np.stack((phi2, psi2, omega2), axis=1),
        np.stack((phi1, psi1, omega1), axis=1),
        loop_coords2, loop_coords1,
        stem_coords2, stem_coords1
    )
    print(f"Length independent dihedral distance: {d1:3g}")