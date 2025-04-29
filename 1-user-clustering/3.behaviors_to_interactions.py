"""
This script takes in a behaviors file and creates an interactions file
- It removes all rows where article_id is "homepage", empty, None or NaN
- It removes all rows where user_id is empty, None or NaN
- It removes all rows where impression_time is empty, None or NaN

- It keeps three columns: user_id, article_id, impression_time
- It writes the result as interactions.csv in the output directory

- It prints the following statistics:
    - Number of interactions (rows in the interactions file)
    - Number of unique users (unique user_ids in the interactions file)
    - Number of unique articles (unique article_ids in the interactions file)
    - Number of interactions per user (mean, min, max, std)
    - Number of interactions per article (mean, min, max, std)
    
Usage: python _behaviors_to_interactions.py --input <path-to-behaviors-file> --output-dir <path-to-output-directory>
python _behaviors_to_interactions.py --input ./datasets/adressa-large-0416/behaviors.parquet --output-dir ./datasets/adressa-large/interactions
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert behaviors file to interactions file')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to behaviors file')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Directory to write interactions file')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read behaviors file based on file extension
    print(f"Reading behaviors file from {args.input}...")
    file_path = Path(args.input)

    if file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(args.input)
    else:
        # Assume CSV for other file types
        df = pd.read_csv(args.input)

    # Count initial rows
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")

    # Filter rows
    df = df.dropna(subset=['user_id', 'article_id', 'impression_time'])
    df = df[df['article_id'] != 'homepage']
    df = df[df['article_id'] != '']

    # Keep only required columns
    df = df[['user_id', 'article_id', 'impression_time']]

    # Convert impression_time to unix timestamp in seconds if not already in that format
    if df['impression_time'].dtype != 'int64':
        df['impression_time'] = pd.to_datetime(
            df['impression_time']).astype(np.int64) // 10**9

    # Write interactions file
    output_path = Path(args.output_dir) / 'interactions.csv'
    print(f"Writing interactions file to {output_path}...")
    df.to_csv(output_path, index=False)

    # Calculate statistics
    n_interactions = len(df)
    n_users = df['user_id'].nunique()
    n_articles = df['article_id'].nunique()

    # Interactions per user
    interactions_per_user = df.groupby('user_id').size()
    mean_interactions_per_user = interactions_per_user.mean()
    min_interactions_per_user = interactions_per_user.min()
    max_interactions_per_user = interactions_per_user.max()
    std_interactions_per_user = interactions_per_user.std()

    # Interactions per article
    interactions_per_article = df.groupby('article_id').size()
    mean_interactions_per_article = interactions_per_article.mean()
    min_interactions_per_article = interactions_per_article.min()
    max_interactions_per_article = interactions_per_article.max()
    std_interactions_per_article = interactions_per_article.std()

    # Print statistics
    print("\nStatistics:")
    print(f"Number of interactions: {n_interactions}")
    print(f"Number of unique users: {n_users}")
    print(f"Number of unique articles: {n_articles}")

    print("\nInteractions per user:")
    print(f"  Mean: {mean_interactions_per_user:.2f}")
    print(f"  Min: {min_interactions_per_user}")
    print(f"  Max: {max_interactions_per_user}")
    print(f"  Std: {std_interactions_per_user:.2f}")

    print("\nInteractions per article:")
    print(f"  Mean: {mean_interactions_per_article:.2f}")
    print(f"  Min: {min_interactions_per_article}")
    print(f"  Max: {max_interactions_per_article}")
    print(f"  Std: {std_interactions_per_article:.2f}")

    # Print rows retained
    print(f"\nRows before filtering: {initial_rows}")
    print(f"Rows after filtering: {n_interactions}")
    print(f"Percentage retained: {(n_interactions/initial_rows)*100:.2f}%")


if __name__ == "__main__":
    main()
