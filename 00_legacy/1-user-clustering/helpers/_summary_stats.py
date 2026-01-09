import pandas as pd
import os
from tabulate import tabulate

# Paths to datasets
adressa_PATH = os.path.join(
    '..', '1-user-clustering', 'datasets', 'adressa-large-0416')
EKSTRA_PATH = os.path.join('..', '1-user-clustering',
                           'datasets', 'ekstra-large')


def calculate_stats(dataset_path, dataset_name, remove_empty_articles=False):
    """Calculate summary statistics for a dataset."""
    print(f"\nProcessing {dataset_name} dataset...")

    # Load data
    behaviors_path = os.path.join(dataset_path, 'behaviors.parquet')
    articles_path = os.path.join(dataset_path, 'articles.parquet')

    behaviors = pd.read_parquet(behaviors_path)
    articles = pd.read_parquet(articles_path)

    if remove_empty_articles:
        behaviors = behaviors[behaviors['article_id'].notna() &
                              (behaviors['article_id'] != 'empty') &
                              (behaviors['article_id'] != 'homepage')]

    # Calculate statistics
    n_users = behaviors['user_id'].nunique()
    n_articles = articles.shape[0]
    n_unique_articles_in_behaviors = behaviors['article_id'].nunique()
    n_impressions = behaviors.shape[0]

    # Check if subscriber column exists
    subscriber_percentage = None
    if 'subscriber' in behaviors.columns:
        subscriber_percentage = (
            behaviors['is_subscriber'].sum() / behaviors.shape[0]) * 100

    # Create statistics dictionary
    stats = {
        'Dataset': dataset_name,
        'Unique Users': f"{n_users:,}",
        'Unique Articles (articles.parquet)': f"{n_articles:,}",
        'Unique Articles (in behaviors)': f"{n_unique_articles_in_behaviors:,}",
        'Impressions': f"{n_impressions:,}"
    }

    if subscriber_percentage is not None:
        stats['Subscriber Percentage'] = f"{subscriber_percentage:.2f}%"

    return stats


def main():
    """Main function to calculate and display statistics for both datasets."""
    print("Calculating summary statistics for adressa and ekstra datasets...\n")

    # Calculate statistics for both datasets
    adressa_stats = calculate_stats(adressa_PATH, 'Adressa (with homepage)')
    adressa_stats_no_homepage = calculate_stats(
        adressa_PATH, 'Adressa (no homepage)', remove_empty_articles=True)
    ekstra_stats = calculate_stats(EKSTRA_PATH, 'Ekstra')
    ekstra_stats_no_homepage = calculate_stats(
        EKSTRA_PATH, 'Ekstra (no homepage)', remove_empty_articles=True)

    # Convert stats to a tabular format
    headers = list(adressa_stats.keys())
    rows = [list(adressa_stats.values()), list(adressa_stats_no_homepage.values(
    )), list(ekstra_stats.values()), list(ekstra_stats_no_homepage.values())]

    # Display results in a table
    print("\nSummary Statistics:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
