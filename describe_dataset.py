"""
Script that describes the content of a dataset folder.
It analyzes the interactions.csv file and reports:
- Number of unique users
- Number of interactions
- Number of unique sessions (if available)
- Number of days the logs span
- Number of unique articles

Usage:
python describe_dataset.py <input_folder>
python describe_dataset.py ./datasets/adressa-one_week
"""

import pandas as pd
import argparse
import os
from pathlib import Path
import zipfile


def find_interactions_file(folder_path):
    """
    Find the interactions file in the given folder.
    Looks for interactions.csv or interactions.csv.zip
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        str: Path to the interactions file, or None if not found
    """
    folder = Path(folder_path)
    
    # Check for interactions.csv
    csv_path = folder / 'interactions.csv'
    if csv_path.exists():
        return str(csv_path)
    
    # Check for interactions.csv.zip
    zip_path = folder / 'interactions.csv.zip'
    if zip_path.exists():
        return str(zip_path)
    
    return None


def load_interactions(file_path):
    """
    Load interactions from CSV file, handling both regular CSV and zipped CSV.
    
    Args:
        file_path (str): Path to the interactions file
        
    Returns:
        pd.DataFrame: The interactions dataframe
    """
    file_path_obj = Path(file_path)
    
    if file_path.endswith('.zip'):
        # Extract and read from zip
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Look for interactions.csv inside the zip
            csv_name = 'interactions.csv'
            if csv_name in zip_ref.namelist():
                with zip_ref.open(csv_name) as f:
                    return pd.read_csv(f)
            else:
                raise ValueError(f"Could not find interactions.csv in {file_path}")
    else:
        # Regular CSV file
        return pd.read_csv(file_path)


def describe_dataset(folder_path):
    """
    Describe the dataset in the given folder.
    
    Args:
        folder_path (str): Path to the dataset folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find interactions file
    interactions_file = find_interactions_file(folder)
    if interactions_file is None:
        raise ValueError(f"Could not find interactions.csv or interactions.csv.zip in {folder_path}")
    
    print(f"Loading interactions from: {interactions_file}")
    df = load_interactions(interactions_file)
    
    print(f"\nDataset Description for: {folder_path}")
    print("=" * 60)
    
    # Basic statistics
    n_interactions = len(df)
    print(f"\nNumber of interactions: {n_interactions:,}")
    
    # Check required columns
    required_cols = ['user_id', 'article_id', 'impression_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Unique users
    n_users = df['user_id'].nunique()
    print(f"Number of unique users: {n_users:,}")
    
    # Unique articles
    n_articles = df['article_id'].nunique()
    print(f"Number of unique articles: {n_articles:,}")
    
    # Unique sessions (if available)
    if 'session_id' in df.columns:
        n_sessions = df['session_id'].nunique()
        print(f"Number of unique sessions: {n_sessions:,}")
    else:
        print("Number of unique sessions: N/A (session_id column not found)")
    
    # Time span
    # Handle different timestamp formats
    impression_time = df['impression_time'].copy()
    
    # Convert to datetime if it's a unix timestamp (integer)
    if pd.api.types.is_integer_dtype(impression_time):
        # Assume unix timestamp in seconds
        impression_time = pd.to_datetime(impression_time, unit='s')
    else:
        # Try to parse as datetime
        impression_time = pd.to_datetime(impression_time)
    
    min_time = impression_time.min()
    max_time = impression_time.max()
    time_span = max_time - min_time
    n_days = time_span.days + (1 if time_span.seconds > 0 else 0)  # Round up to include partial days
    
    print(f"\nTime span:")
    print(f"  Start: {min_time}")
    print(f"  End: {max_time}")
    print(f"  Number of days: {n_days}")
    
    # Additional statistics
    print(f"\nAdditional statistics:")
    print(f"  Average interactions per user: {n_interactions / n_users:.2f}")
    print(f"  Average interactions per article: {n_interactions / n_articles:.2f}")
    
    if 'session_id' in df.columns:
        print(f"  Average interactions per session: {n_interactions / n_sessions:.2f}")
        print(f"  Average sessions per user: {n_sessions / n_users:.2f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Describe the content of a dataset folder')
    parser.add_argument('input_folder', type=str,
                        help='Path to the dataset folder containing interactions.csv')
    
    args = parser.parse_args()
    
    describe_dataset(args.input_folder)


if __name__ == "__main__":
    main()

