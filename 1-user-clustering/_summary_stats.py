import pandas as pd
import os
from tabulate import tabulate

# Paths to datasets
ADDRESSA_PATH = os.path.join('..', '1-user-clustering', 'datasets', 'addressa-large')
EKSTRA_PATH = os.path.join('..', '1-user-clustering', 'datasets', 'ekstra-large')

def calculate_stats(dataset_path, dataset_name):
    """Calculate summary statistics for a dataset."""
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load data
    behaviors_path = os.path.join(dataset_path, 'behaviors.parquet')
    articles_path = os.path.join(dataset_path, 'articles.parquet')
    
    behaviors = pd.read_parquet(behaviors_path)
    articles = pd.read_parquet(articles_path)
    
    # Calculate statistics
    n_users = behaviors['user_id'].nunique()
    n_articles = articles.shape[0]
    n_unique_articles_in_behaviors = behaviors['article_id'].nunique()
    n_impressions = behaviors.shape[0]
    
    # Check if subscriber column exists
    subscriber_percentage = None
    if 'subscriber' in behaviors.columns:
        subscriber_percentage = (behaviors['subscriber'].sum() / behaviors.shape[0]) * 100
    
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
    print("Calculating summary statistics for addressa and ekstra datasets...\n")
    
    # Calculate statistics for both datasets
    addressa_stats = calculate_stats(ADDRESSA_PATH, 'Addressa')
    ekstra_stats = calculate_stats(EKSTRA_PATH, 'Ekstra')
    
    # Convert stats to a tabular format
    headers = list(addressa_stats.keys())
    rows = [list(addressa_stats.values()), list(ekstra_stats.values())]
    
    # Display results in a table
    print("\nSummary Statistics:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
