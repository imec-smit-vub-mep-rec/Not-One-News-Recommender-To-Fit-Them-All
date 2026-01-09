"""
Script that summarizes a given interaction matrix with columns user_id, item_id, impression_time, and cluster_id.
It reports the number of interactions, users, and items in the matrix.
It also reports the sparsity of the matrix.
It reports the number of users and items with 0,1,2,3,4,5 interactions, per cluster.
It reports the proportion of users that have less than 3 interactions, between 3-5 interactions, and more than 5 interactions, per cluster.

Usage:
python _summarize_interactions.py --interaction_matrix_path <path_to_interaction_matrix> --user_col <user_column_name> --item_col <item_column_name> --cluster_col <cluster_column_name>
"""

import pandas as pd
import argparse
import numpy as np
from collections import Counter


def load_interaction_matrix(path):
    """
    Load the interaction matrix from the given path.

    Args:
        path (str): Path to the interaction matrix.

    Returns:
        pd.DataFrame: The interaction matrix.
    """
    # Try different formats
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format for {path}")


def get_interaction_counts(matrix, col_name):
    """
    Get the number of interactions per user or item.

    Args:
        matrix (pd.DataFrame): The interaction matrix.
        col_name (str): The column name to count interactions for.

    Returns:
        Counter: A counter of the number of interactions.
    """
    return Counter(matrix[col_name].value_counts().values)


def summarize_interaction_matrix(matrix, user_col='user_id', item_col='item_id', cluster_col=None):
    """
    Summarize the interaction matrix.

    Args:
        matrix (pd.DataFrame): The interaction matrix.
        user_col (str): The column name for users.
        item_col (str): The column name for items.
        cluster_col (str, optional): The column name for clusters.

    Returns:
        dict: A dictionary with the summary statistics.
    """
    num_interactions = len(matrix)
    unique_users = matrix[user_col].nunique()
    unique_items = matrix[item_col].nunique()

    # Calculate sparsity
    potential_interactions = unique_users * unique_items
    sparsity = 1 - (num_interactions / potential_interactions)

    print(f"Number of interactions: {num_interactions}")
    print(f"Number of users: {unique_users}")
    print(f"Number of items: {unique_items}")
    print(f"Matrix sparsity: {sparsity:.4f}")

    # User interaction distribution
    user_interact_counts = matrix[user_col].value_counts()
    item_interact_counts = matrix[item_col].value_counts()

    user_count_distribution = {
        i: (user_interact_counts == i).sum() for i in range(6)}
    item_count_distribution = {
        i: (item_interact_counts == i).sum() for i in range(6)}

    print("\nUser interaction distribution:")
    for i in range(6):
        print(f"  Users with {i} interactions: {user_count_distribution[i]}")

    print("\nItem interaction distribution:")
    for i in range(6):
        print(f"  Items with {i} interactions: {item_count_distribution[i]}")

    # If clustering information is available
    if cluster_col and cluster_col in matrix.columns:
        print("\nPer cluster statistics:")
        clusters = matrix[cluster_col].unique()

        for cluster in clusters:
            cluster_data = matrix[matrix[cluster_col] == cluster]
            cluster_users = cluster_data[user_col].nunique()

            user_interact_per_cluster = cluster_data[user_col].value_counts()

            # Calculate proportions
            # less_than_3 = (user_interact_per_cluster < 3).sum() / cluster_users
            only_one = (user_interact_per_cluster == 1).sum() / cluster_users
            only_two = (user_interact_per_cluster == 2).sum() / cluster_users
            between_3_and_5 = ((user_interact_per_cluster >= 3) & (
                user_interact_per_cluster <= 5)).sum() / cluster_users
            more_than_5 = (user_interact_per_cluster > 5).sum() / cluster_users

            print(f"\nCluster {cluster}:")
            print(f"  Number of users: {cluster_users}")
            print(
                f"  Proportion of users with only 1 interaction: {only_one:.4f}")
            print(
                f"  Proportion of users with only 2 interactions: {only_two:.4f}")
            print(
                f"  Proportion of users with 3-5 interactions (incl. 3 and 5): {between_3_and_5:.4f}")
            print(
                f"  Proportion of users with > 5 interactions: {more_than_5:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize interaction matrix.')
    parser.add_argument('--interaction_matrix_path', type=str, required=True,
                        help='Path to the interaction matrix file')
    parser.add_argument('--user_col', type=str, default='user_id',
                        help='Column name for user IDs')
    parser.add_argument('--item_col', type=str, default='article_id',
                        help='Column name for item IDs')
    parser.add_argument('--cluster_col', type=str, default='cluster_id',
                        help='Column name for cluster IDs (if available)')

    args = parser.parse_args()

    # Load the interaction matrix
    matrix = load_interaction_matrix(args.interaction_matrix_path)

    # Summarize the interaction matrix
    summarize_interaction_matrix(
        matrix,
        user_col=args.user_col,
        item_col=args.item_col,
        cluster_col=args.cluster_col
    )


if __name__ == "__main__":
    main()
