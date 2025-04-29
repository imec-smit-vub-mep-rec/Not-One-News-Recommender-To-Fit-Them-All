"""
Script that summarizes the unique users and articles in the datasets
Also reports average number of occurences of each user and article
Compares these 2 datasets:
- datasets/adressa/lien/adressa_one_week.csv
- datasets/adressa/large/combined_clusters.csv
"""

import pandas as pd
import os


def summarize_dataset(df, dataset_name):
    """Summarize the dataset"""
    print(f"Dataset: {dataset_name}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Number of unique articles: {df['article_id'].nunique()}")

    # Calculate average number of occurences of each user and article
    user_counts = df['user_id'].value_counts()
    article_counts = df['article_id'].value_counts()
    print(f"Average number of occurences of each user: {user_counts.mean()}")
    print(
        f"Average number of occurences of each article: {article_counts.mean()}")


def main():
    # Load the datasets
    df_one_week = pd.read_csv('datasets/adressa/lien/adressa_one_week.csv')
    df_large = pd.read_csv('datasets/adressa/large/combined_clusters.csv')

    # Summarize the datasets
    summarize_dataset(df_one_week, 'Adressa One Week')
    summarize_dataset(df_large, 'Adressa Large')


if __name__ == '__main__':
    main()
