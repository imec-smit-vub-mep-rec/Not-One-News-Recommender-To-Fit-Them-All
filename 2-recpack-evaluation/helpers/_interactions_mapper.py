"""
Simple script that takes in a csv file with interactions (user_id, article_id, impression_time)
and a folder containing the clusters eg cluster_0_merged.csv and cluster_1_merged.csv

Per cluster, extract the user_id for the cluster and store it in a map

Then, output a csv with the interactions, but with the corresponding cluster_id added as a new column to the csv

Usage:
python _interactions_mapper.py --interactions_path <path_to_interactions_csv> --clusters_folder <path_to_clusters_folder> --output_path <path_to_output_csv>
python _interactions_mapper.py --interactions_path datasets/adressa/interactions.csv --clusters_folder datasets/adressa/1.input.clusters --output_path datasets/adressa/interactions_mapped.csv
"""

import pandas as pd
import argparse
import os
import re
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Map interactions to user clusters')
    parser.add_argument('--interactions_path', type=str,
                        required=True, help='Path to interactions CSV file')
    parser.add_argument('--clusters_folder', type=str, required=True,
                        help='Path to folder containing cluster files')
    parser.add_argument('--output_path', type=str,
                        required=True, help='Path to output CSV file')
    return parser.parse_args()


def load_interactions(interactions_path: str) -> pd.DataFrame:
    """Load interactions from CSV file"""
    return pd.read_csv(interactions_path)


def create_user_to_cluster_map(clusters_folder: str) -> Dict[str, int]:
    """Create a mapping from user_id to cluster_id"""
    user_to_cluster = {}

    # Get all cluster files in the folder
    cluster_files = [f for f in os.listdir(
        clusters_folder) if f.startswith('cluster_') and f.endswith('.csv')]

    for cluster_file in cluster_files:
        # Extract cluster ID from filename
        match = re.match(r'cluster_(\d+)_merged\.csv', cluster_file)
        if match:
            cluster_id = int(match.group(1))

            # Load cluster file
            cluster_df = pd.read_csv(
                os.path.join(clusters_folder, cluster_file))

            # Map each user in this cluster to the cluster ID
            for user_id in cluster_df['user_id'].unique():
                user_to_cluster[user_id] = cluster_id

    return user_to_cluster


def map_interactions_to_clusters(interactions_df: pd.DataFrame, user_to_cluster: Dict[str, int]) -> pd.DataFrame:
    """Add cluster_id column to interactions DataFrame"""
    # Create a new column with the cluster ID for each user
    interactions_df['cluster_id'] = interactions_df['user_id'].map(
        user_to_cluster)

    # Handle users that don't belong to any cluster
    interactions_df['cluster_id'] = interactions_df['cluster_id'].fillna(
        -1).astype(int)

    return interactions_df


def main():
    args = parse_args()

    # Load interactions
    interactions_df = load_interactions(args.interactions_path)

    # Create user to cluster mapping
    user_to_cluster = create_user_to_cluster_map(args.clusters_folder)

    # Map interactions to clusters
    result_df = map_interactions_to_clusters(interactions_df, user_to_cluster)

    # Save result
    result_df.to_csv(args.output_path, index=False)
    print(f"Mapped interactions saved to {args.output_path}")

    # Print some stats
    total_users = len(interactions_df['user_id'].unique())
    mapped_users = sum(
        1 for user in interactions_df['user_id'].unique() if user in user_to_cluster)
    print(
        f"Mapped {mapped_users} out of {total_users} users ({mapped_users/total_users*100:.2f}%)")


if __name__ == "__main__":
    main()
