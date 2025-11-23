import pandas as pd
from igloo.metrics import get_structure_alignment_score_all_pairs
import os
import numpy as np
import argparse
import time

def parse_args():
    argparser = argparse.ArgumentParser(description="Calculate TM scores for clusters")
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for parallel processing",
    )
    argparser.add_argument(
        "--metric",
        type=str,
        default="tm_score",
        choices=["tm_score", "rmsd"],
        help="Metric to use for structure alignment",
    )
    argparser.add_argument(
        "--max_sample",
        type=int,
        default=10000,
        help="Maximum number of proteins to calculate pairwise TM scores for",
    )
    argparser.add_argument(
        "--cluster_file",
        type=str,
        required=True,
        help="Path to the parquet file containing loop_id, pdb_file, chain_id, residues, and cluster assignments",
    )
    argparser.add_argument(
        "--cluster_key",
        type=str,
        default="cluster_id",
        help="Key to use for cluster assignments in the data file",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output TM scores. One parquet file per cluster.",
    )
    argparser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the PDB files for the structures",
    )
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_workers = args.num_workers
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.cluster_file)
    print("Number of canonical clusters:", df[args.cluster_key].nunique())

    for cluster in df[args.cluster_key].unique():
        cluster_df = df[df[args.cluster_key] == cluster]
        if len(cluster_df) > args.max_sample:
            cluster_df = cluster_df.sample(args.max_sample, random_state=42)
        
        # A pairwise TM scores for 1000 proteins takes about 36 seconds, scales O(n^2)
        start = time.time()
        tm_scores, entry_ids = get_structure_alignment_score_all_pairs(cluster_df.copy(), args.data_dir, num_workers, metric=args.metric)
        end = time.time()
        print(f"Cluster {cluster} (N={len(cluster_df)}) {args.metric} calculated in {end - start:.2f} seconds")

        tm_scores = np.array(tm_scores)
        cluster_tm_scores = pd.DataFrame({
            'entry_id_1': [entry[0] for entry in entry_ids],
            'entry_id_2': [entry[1] for entry in entry_ids],
            'tm_score': tm_scores
        })
        cluster_name = str(cluster).replace("*", "x") # Kelow cluster names use *
        cluster_tm_scores.to_parquet(os.path.join(args.output_dir, f"{cluster_name}_{args.metric}.parquet"))

