"""
Function to split the data into multiple parts with specified percentages and save as separate CSV files.
Ensures all data from the same user stays together in the same split.

Usage:
python random_split_generator.py \
  --behaviors datasets/ekstra-large/behaviors.parquet \
  --articles datasets/ekstra-large/articles.parquet \
  --output_dir results/data_splits \
  --split_percentages 70 20 10 \
  --random_state 42


  
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split


def load_data(behaviors_path, articles_path):
    """
    Load and prepare the datasets from parquet files.

    Parameters:
    -----------
    behaviors_path : str
        Path to the behaviors.parquet file
    articles_path : str
        Path to the articles.parquet file

    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with behaviors and article information
    """
    # Load the datasets
    print(f"Loading behaviors from {behaviors_path}")
    print(f"Loading articles from {articles_path}")

    articles_df = pd.read_parquet(articles_path)
    behaviors_df = pd.read_parquet(behaviors_path)

    # Print column names of behaviors_df
    print("Columns in behaviors_df:", behaviors_df.columns.tolist())

    # Split behaviors into those with and without article_id
    behaviors_with_article = behaviors_df[behaviors_df['article_id'].notna()]
    # behaviors_without_article = behaviors_df[behaviors_df['article_id'].isna()]

    # Merge behaviors with articles to get topic information only for rows with article_id
    merged_with_articles = behaviors_with_article.merge(
        articles_df[['article_id', 'category_str', 'sentiment_score']],
        on='article_id',
        how='left'
    )

    # Combine the merged data with the behaviors without article_id
    merged_df = merged_with_articles

    # Print first 5 rows of merged_df
    print("Sample of merged data:")
    print(merged_df.head())

    # Print column names of merged_df
    print("Columns in merged_df:", merged_df.columns.tolist())

    # Ensure all required columns are present
    required_columns = [
        'impression_id', 'article_id', 'impression_time', 'read_time',
        'scroll_percentage', 'device_type', 'article_ids_inview',
        'article_ids_clicked', 'user_id', 'is_sso_user', 'gender',
        'postcode', 'age', 'is_subscriber', 'session_id',
        'next_read_time', 'next_scroll_percentage', 'category_str',
        'sentiment_score'
    ]

    # Check for missing columns
    missing_columns = [
        col for col in required_columns if col not in merged_df.columns]
    if missing_columns:
        print(
            f"Warning: The following required columns are missing: {missing_columns}")
        print("These columns will be filled with NaN values in the output.")

        # Add missing columns with NaN values
        for col in missing_columns:
            merged_df[col] = np.nan

    return merged_df


def split_dataframe(df, output_dir, split_percentages, random_state=42):
    """
    Split the dataframe into multiple parts with specified percentages and save as separate CSV files.
    Ensures all data from the same user stays together in the same split.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to split
    output_dir : str
        Directory to save the split files
    split_percentages : list of float
        List of percentages for each split (should sum to 100)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    list
        List of file paths for the created splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group by user_id to ensure all user data stays together
    user_groups = df.groupby('user_id')
    unique_users = df['user_id'].unique()
    total_users = len(unique_users)

    # Shuffle users to ensure random distribution
    np.random.seed(random_state)
    shuffled_users = np.random.permutation(unique_users)

    # Validate split percentages
    total_percentage = sum(split_percentages)
    if abs(total_percentage - 100) > 0.01:  # Allow for small floating point errors
        print(
            f"Warning: Split percentages sum to {total_percentage}%, not 100%")
        print("Adjusting percentages to sum to 100%")
        # Adjust percentages proportionally
        ratio = 100 / total_percentage
        split_percentages = [pct * ratio for pct in split_percentages]

    # Convert percentages to user counts
    split_sizes = [int(total_users * pct / 100) for pct in split_percentages]

    # Handle rounding errors
    remaining = total_users - sum(split_sizes)
    if remaining > 0:
        # Add remaining users to the largest split
        largest_split_idx = split_sizes.index(max(split_sizes))
        split_sizes[largest_split_idx] += remaining
    elif remaining < 0:
        # Remove excess users from the largest split
        largest_split_idx = split_sizes.index(max(split_sizes))
        # remaining is negative, so this subtracts
        split_sizes[largest_split_idx] += remaining

    # Create splits
    file_paths = []
    start_idx = 0

    for i, size in enumerate(split_sizes):
        end_idx = start_idx + size

        # Ensure we don't go beyond the user count
        if end_idx > total_users:
            end_idx = total_users

        # Get the users for this split
        split_users = shuffled_users[start_idx:end_idx]

        # Get all data for these users
        split_df = df[df['user_id'].isin(split_users)]

        output_file = os.path.join(output_dir, f"split_{i+1}.csv")
        split_df.to_csv(output_file, index=False)
        print(
            f"Saved {len(split_df)} rows ({len(split_users)} users) to {output_file}")
        file_paths.append(output_file)

        start_idx = end_idx

        # Break if we've reached the end of the users
        if end_idx >= total_users:
            break

    return file_paths


"""
Example usage:
python user_split_generator.py \
  --behaviors datasets/ekstra-large/behaviors.parquet \
  --articles datasets/ekstra-large/articles.parquet \
  --output_dir results/data_splits \
  --split_percentages 70 20 10 \
  --random_state 42
"""


def main():
    parser = argparse.ArgumentParser(
        description='Split a merged dataframe into multiple CSV files')
    parser.add_argument('--behaviors', required=True,
                        help='Path to behaviors.parquet file')
    parser.add_argument('--articles', required=True,
                        help='Path to articles.parquet file')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for split files')
    parser.add_argument('--split_percentages', type=float, nargs='+', default=[33.33, 33.33, 33.34],
                        help='Percentages for each split (should sum to 100)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Load and combine data
    merged_df = load_data(args.behaviors, args.articles)

    # Split the dataframe
    file_paths = split_dataframe(
        merged_df,
        args.output_dir,
        args.split_percentages,
        args.random_state
    )

    print("\nSplit summary:")
    total_rows = len(merged_df)
    print(f"Total rows: {total_rows}")

    start_idx = 0
    for i, file_path in enumerate(file_paths):
        if i < len(args.split_percentages):
            split_size = len(pd.read_csv(file_path))
            print(
                f"Split {i+1}: {split_size} rows ({split_size/total_rows*100:.1f}%)")


if __name__ == "__main__":
    main()
