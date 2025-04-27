"""
Function to perform user clustering on a dataset in Ekstra format (behaviors and articles)

Usage:
python user_clustering.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import List, Dict, Tuple
import json
from sklearn.impute import SimpleImputer
from kneed import KneeLocator

result_folder = 'ekstra-small'
dataset = 'ekstra-small'
# Create results directory if it doesn't exist
os.makedirs(f'results/user_clusters/{result_folder}', exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare the datasets."""
    # Load the datasets
    articles_location = f'datasets/{dataset}/articles.parquet'
    behaviors_location = f'datasets/{dataset}/behaviors.parquet'
    articles_df = pd.read_parquet(articles_location)
    behaviors_df = pd.read_parquet(behaviors_location)

    # Print column names of behaviors_df
    print("Columns in behaviors_df:", behaviors_df.columns.tolist())

    # Split behaviors into those with and without article_id
    behaviors_with_article = behaviors_df[behaviors_df['article_id'].notna()]
    behaviors_without_article = behaviors_df[behaviors_df['article_id'].isna()]

    # Merge behaviors with articles to get topic information only for rows with article_id
    merged_with_articles = behaviors_with_article.merge(
        articles_df[['article_id', 'category_str', 'sentiment_score']],
        on='article_id',
        how='left'
    )

    # Combine the merged data with the behaviors without article_id
    merged_df = pd.concat(
        [merged_with_articles, behaviors_without_article], ignore_index=True)

    # Print first 5 rows of merged_df
    print(merged_df.head())

    # Print column names of merged_df
    print("Columns in merged_df:", merged_df.columns.tolist())

    return merged_df, articles_df

# Create a csv table per article_id, with the following columns:
# - article_id
# - category_str
# - sentiment_score
# - count of impressions (based on behaviors.parquet)
# - count of unique categories (based on behaviors.parquet)
# - count of unique users that accessed this article (based on behaviors.parquet)
# - average reading time (based on behaviors.parquet)
# - average scroll depth (based on behaviors.parquet)


def create_article_table(merged_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    """Create a csv table per article_id, with the following columns:
    - article_id
    - category_str
    - sentiment_score
    - count of impressions (based on behaviors.parquet)
    - count of unique categories (based on behaviors.parquet)
    - count of unique users that accessed this article (based on behaviors.parquet)
    - average reading time (based on behaviors.parquet)
    # - average scroll depth (based on behaviors.parquet) - not accurate
    """
    article_table = merged_df.groupby('article_id').agg(
        count_impressions=('impression_id', 'count'),
        count_unique_categories=('category_str', 'nunique'),
        count_unique_users=('user_id', 'nunique'),
        avg_reading_time=('read_time', 'mean'),
        # avg_scroll_depth=('scroll_percentage', 'mean') - not accurate
    ).reset_index()

    # Merge with articles_df to get category_str and sentiment_score
    article_table = article_table.merge(
        articles_df[['article_id', 'category_str', 'sentiment_score']],
        on='article_id',
        how='left'
    )

    # Save the article table to a csv file
    article_table.to_csv(
        f'results/user_clusters/{result_folder}/article_table.csv', index=False)

    return article_table


def create_user_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create a csv table per user_id, with the following columns:
    - user_id
    - count of sessions (based on behaviors.parquet, by unique session_id)
    - count of impressions (based on behaviors.parquet, by all impressions (behaviors for which article_id is not null))
    - count of unique categories (based on behaviors.parquet, by all articles for which article_id is not null)
    - count of unique articles viewed (based on behaviors.parquet, by all articles for which article_id is not null)
    - average reading time (based on behaviors.parquet, by all articles for which article_id is not null)
    - average scroll depth (based on behaviors.parquet, by all articles for which article_id is not null)
    - average session length (number of articles viewed per session)
    - average number of categories per session
    - average session duration (based on behaviors.parquet, by unique session_id)
    - percentage of sessions in the morning, afternoon, evening and night
    - subscription status
    """
    # First, get the subscription status for each user (it's the same for all rows of a user)
    subscription_status = merged_df.groupby('user_id')['is_subscriber'].first()

    # Convert Unix timestamp to datetime
    # Print the first 5 impression_time values
    print(merged_df['impression_time'].head())
    merged_df['impression_time'] = pd.to_datetime(
        merged_df['impression_time'], unit='ms')

    user_table = merged_df.groupby('user_id').agg(
        count_sessions=('session_id', 'nunique'),  # Number of sessions
        # Number of total impressions
        count_total_impressions=('impression_id', 'count'),
        # Number of impressions on homepage (where article_id is not defined)
        count_total_homepage_impressions=(
            'article_id', lambda x: x.isna().sum()),
        # Number of impressions on articles (where article_id is defined)
        count_total_article_impressions=(
            'article_id', lambda x: x.notna().sum()),
        # Number of unique categories (excluding empty strings)
        count_total_unique_categories=(
            'category_str', lambda x: x[(x != '') & (x.notna())].nunique()),
        # Number of unique articles
        count_total_unique_articles=('article_id', 'nunique'),
        avg_reading_time=('read_time', 'mean'),  # Average reading time
        avg_session_length=('session_id', lambda x: x.groupby(
            x).size().mean()),  # Average number of articles per session
        percentage_morning=('impression_time', lambda x: (
            (x.dt.hour >= 6) & (x.dt.hour < 12)).mean()),
        percentage_afternoon=('impression_time', lambda x: (
            (x.dt.hour >= 12) & (x.dt.hour < 18)).mean()),
        percentage_evening=('impression_time', lambda x: (
            (x.dt.hour >= 18) & (x.dt.hour < 24)).mean()),
        percentage_night=('impression_time', lambda x: (
            (x.dt.hour >= 0) & (x.dt.hour < 6)).mean())
    ).reset_index()

    # Calculate average categories per session separately, only considering valid categories
    valid_categories_df = merged_df[(merged_df['category_str'] != '')
                                    & (merged_df['category_str'].notna())]
    session_categories = valid_categories_df.groupby(['user_id', 'session_id'])[
        'category_str'].nunique()
    # Only consider sessions that have at least one valid category
    session_categories = session_categories[session_categories > 0]
    avg_categories_per_session = session_categories.groupby('user_id').mean()

    # Calculate average session duration separately
    session_durations = merged_df.groupby(['user_id', 'session_id'])['impression_time'].agg(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    avg_session_duration = session_durations.groupby('user_id').mean()

    # Calculate proportion of time spent on articles vs homepage
    total_reading_time = merged_df.groupby('user_id')['read_time'].sum()
    article_reading_time = merged_df[merged_df['article_id'].notna()].groupby('user_id')[
        'read_time'].sum()
    homepage_reading_time = merged_df[merged_df['article_id'].isna()].groupby('user_id')[
        'read_time'].sum()
    proportion_article_time = article_reading_time / total_reading_time

    # Calculate average reading time on homepage in seconds
    avg_reading_time_homepage = merged_df[merged_df['article_id'].isna()].groupby('user_id')[
        'read_time'].mean()
    avg_reading_time_articles = merged_df[merged_df['article_id'].notna()].groupby(
        'user_id')['read_time'].mean()

    # Calculate average category switches per session
    # First, filter for valid categories
    valid_df = merged_df[(merged_df['category_str'] != '')
                         & (merged_df['category_str'].notna())]
    # Sort by user_id, session_id, and impression_time
    sorted_df = valid_df.sort_values(
        ['user_id', 'session_id', 'impression_time'])

    # Calculate category switches per session
    category_switches = []
    for (user_id, session_id), group in sorted_df.groupby(['user_id', 'session_id']):
        categories = group['category_str'].values
        if len(categories) > 1:  # Only count switches if there are at least 2 articles
            switches = sum(1 for i in range(1, len(categories))
                           if categories[i] != categories[i-1])
            category_switches.append({
                'user_id': user_id,
                'session_id': session_id,
                'switches': switches
            })

    # Convert to DataFrame and calculate average per user
    if category_switches:  # Only process if we found any switches
        switches_df = pd.DataFrame(category_switches)
        avg_switches_per_session = switches_df.groupby('user_id')[
            'switches'].mean()
    else:
        avg_switches_per_session = pd.Series(dtype=float)

    # Add the calculated metrics to the user table
    user_table['avg_categories_per_session'] = user_table['user_id'].map(
        avg_categories_per_session)
    user_table['avg_session_duration'] = user_table['user_id'].map(
        avg_session_duration)
    user_table['proportion_article_time'] = user_table['user_id'].map(
        proportion_article_time)
    user_table['avg_reading_time_homepage'] = user_table['user_id'].map(
        avg_reading_time_homepage)
    user_table['avg_reading_time_articles'] = user_table['user_id'].map(
        avg_reading_time_articles)
    user_table['avg_category_switches'] = user_table['user_id'].map(
        avg_switches_per_session)

    # Add subscription status to the user table
    user_table['is_subscriber'] = user_table['user_id'].map(
        subscription_status)

    # Save the user table to a csv file
    user_table.to_csv(
        f'results/user_clusters/{result_folder}/user_table.csv', index=False)

    return user_table

# Clustering analysis: can distinct user clusters be identified based on the user tables?
# Look at the user table and try to find distinct clusters by trying different combinations of features


def prepare_clustering_data(user_table: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Prepare data for clustering by selecting and scaling relevant features."""
    # Select features for clustering
    features = [
        'count_sessions',
        'count_total_impressions',
        'count_total_homepage_impressions',
        'count_total_article_impressions',
        'count_total_unique_categories',
        'count_total_unique_articles',
        'avg_reading_time',
        'proportion_article_time',
        'avg_session_length',
        'avg_categories_per_session',
        'avg_category_switches',
        'avg_session_duration',
        # 'percentage_morning',
        # 'percentage_afternoon',
        # 'percentage_evening',
        # 'percentage_night'
    ]

    # Create feature matrix
    X = user_table[features].copy()

    # Handle missing values - fill with 0 for these specific columns
    X['avg_categories_per_session'] = X['avg_categories_per_session'].fillna(0)
    X['avg_category_switches'] = X['avg_category_switches'].fillna(0)
    X['proportion_article_time'] = X['proportion_article_time'].fillna(0)

    # Handle remaining missing values with mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, scaler


def find_optimal_clusters(X: pd.DataFrame, max_clusters: int = 10) -> int:
    """Find optimal number of clusters using elbow method."""
    distortions = []
    K = range(1, max_clusters + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(f'results/user_clusters/{result_folder}/elbow_plot.png')
    plt.close()

    # Find elbow point using kneed library
    kneedle = KneeLocator(K, distortions, curve='convex',
                          direction='decreasing')
    optimal_k = kneedle.elbow if kneedle.elbow else 5  # Default to 5 if no clear elbow

    return optimal_k


def perform_clustering(X: pd.DataFrame, n_clusters: int) -> Tuple[KMeans, pd.DataFrame]:
    """Perform K-means clustering and return cluster assignments."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster labels to original data
    X_with_clusters = X.copy()
    X_with_clusters['cluster'] = cluster_labels

    return kmeans, X_with_clusters


def calculate_category_switches(merged_df: pd.DataFrame) -> pd.Series:
    """Calculate average number of category switches per session for each user.
    A category switch occurs when a user reads articles from different categories within the same session."""
    # Sort by user_id, session_id, and impression_time to ensure correct order
    sorted_df = merged_df.sort_values(
        ['user_id', 'session_id', 'impression_time'])

    # Calculate category switches per session
    category_switches = []
    for (user_id, session_id), group in sorted_df.groupby(['user_id', 'session_id']):
        # Get categories in order of reading, excluding empty strings
        categories = group['category_str'][group['category_str'] != ''].values
        # Count switches (when category changes)
        switches = sum(1 for i in range(1, len(categories))
                       if categories[i] != categories[i-1])
        category_switches.append({
            'user_id': user_id,
            'session_id': session_id,
            'category_switches': switches
        })

    # Convert to DataFrame and calculate average per user
    switches_df = pd.DataFrame(category_switches)
    avg_switches = switches_df.groupby('user_id')['category_switches'].mean()

    return avg_switches


def create_cluster_summary(user_table: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    """Create a detailed summary of cluster characteristics."""
    # Calculate cluster sizes and percentages
    cluster_sizes = user_table['cluster'].value_counts().sort_index()
    total_users = len(user_table)
    cluster_percentages = (cluster_sizes / total_users * 100).round(2)

    # Calculate subscription status per cluster
    subscription_counts = user_table.groupby(
        ['cluster', 'is_subscriber']).size().unstack(fill_value=0)

    # Calculate metrics per cluster
    cluster_metrics = user_table.groupby('cluster').agg({
        'avg_reading_time': 'mean',
        'proportion_article_time': 'mean',
        'avg_session_length': 'mean',
        'count_total_unique_categories': 'mean',
        'avg_session_duration': 'mean',
        'avg_reading_time_homepage': 'mean',
        'avg_reading_time_articles': 'mean',
        'percentage_morning': 'mean',
        'percentage_evening': 'mean',
        'percentage_afternoon': 'mean',
        'percentage_night': 'mean'
    }).round(2)

    # Calculate category switches
    category_switches = calculate_category_switches(merged_df)
    user_table['avg_category_switches'] = user_table['user_id'].map(
        category_switches)
    avg_switches_per_cluster = user_table.groupby(
        'cluster')['avg_category_switches'].mean().round(2)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Number of Users': cluster_sizes,
        'Percentage of Users': cluster_percentages,
        'Avg Reading Time (s)': cluster_metrics['avg_reading_time'],
        'Proportion of Time on Articles': cluster_metrics['proportion_article_time'],
        'Avg Reading Time Homepage (s)': cluster_metrics['avg_reading_time_homepage'],
        'Avg Reading Time Articles (s)': cluster_metrics['avg_reading_time_articles'],
        'Avg Articles per Session': cluster_metrics['avg_session_length'],
        'Avg Categories Read': cluster_metrics['count_total_unique_categories'],
        'Avg Session Duration (s)': cluster_metrics['avg_session_duration'],
        'Avg Category Switches per Session': avg_switches_per_cluster,
        'Avg Reading Time Morning (%)': cluster_metrics['percentage_morning'],
        'Avg Reading Time Evening (%)': cluster_metrics['percentage_evening'],
        'Avg Reading Time Afternoon (%)': cluster_metrics['percentage_afternoon'],
        'Avg Reading Time Night (%)': cluster_metrics['percentage_night'],
    })

    # Save summary to CSV
    summary_df.to_csv(
        f'results/user_clusters/{result_folder}/cluster_summary.csv')

    # Create a stacked bar plot for cluster sizes with subscription status
    plt.figure(figsize=(12, 6))

    # Plot stacked bars
    bottom = np.zeros(len(cluster_sizes))
    # Yellow for non-subscribers, Green for subscribers
    colors = ['#FFD700', '#2E8B57']

    for i, (is_subscriber, counts) in enumerate(subscription_counts.items()):
        plt.bar(summary_df.index, counts, bottom=bottom,
                label='Subscribers' if is_subscriber else 'Non-subscribers',
                color=colors[i])
        bottom += counts

    plt.title('Cluster Sizes by Subscription Status')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.legend()

    # Add percentage labels on top of bars
    for i in range(len(cluster_sizes)):
        total = subscription_counts.iloc[i].sum()
        for j, count in enumerate(subscription_counts.iloc[i]):
            percentage = (count / total * 100).round(1)
            plt.text(i, bottom[i] - count/2,
                     f'{percentage}%',
                     ha='center', va='center',
                     color='black' if j == 0 else 'white')

    plt.tight_layout()
    plt.savefig(f'results/user_clusters/{result_folder}/cluster_sizes.png')
    plt.close()

    # Create a comparison plot for key metrics
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Reading time
    axes[0, 0].bar(summary_df.index, summary_df['Avg Reading Time (s)'])
    axes[0, 0].set_title('Average Reading Time')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Seconds')

    # Reading time homepage
    axes[0, 1].bar(summary_df.index,
                   summary_df['Avg Reading Time Homepage (s)'])
    axes[0, 1].set_title('Average Reading Time on Homepage')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Seconds')

    # Proportion of time on articles
    axes[0, 2].bar(summary_df.index,
                   summary_df['Proportion of Time on Articles'])
    axes[0, 2].set_title('Proportion of Time on Articles')
    axes[0, 2].set_xlabel('Cluster')
    axes[0, 2].set_ylabel('Proportion')

    # Session duration
    axes[0, 3].bar(summary_df.index, summary_df['Avg Session Duration (s)'])
    axes[0, 3].set_title('Average Session Duration')
    axes[0, 3].set_xlabel('Cluster')
    axes[0, 3].set_ylabel('Seconds')

    # Articles per session
    axes[1, 0].bar(summary_df.index, summary_df['Avg Articles per Session'])
    axes[1, 0].set_title('Average Articles per Session')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of Articles')

    # Categories read
    axes[1, 1].bar(summary_df.index, summary_df['Avg Categories Read'])
    axes[1, 1].set_title('Average Categories Read')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Categories')

    # Category switches
    axes[1, 2].bar(summary_df.index,
                   summary_df['Avg Category Switches per Session'])
    axes[1, 2].set_title('Average Category Switches per Session')
    axes[1, 2].set_xlabel('Cluster')
    axes[1, 2].set_ylabel('Number of Switches')

    # Hide the last subplot
    axes[1, 3].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f'results/user_clusters/{result_folder}/cluster_metrics_comparison.png')
    plt.close()

    # Print summary to console
    print("\nCluster Summary:")
    print("=" * 80)
    print(summary_df.to_string())
    print(
        f"\nDetailed metrics saved to 'results/user_clusters/{result_folder}/cluster_summary.csv'")


def analyze_category_popularity(merged_df: pd.DataFrame, user_table: pd.DataFrame) -> None:
    """Analyze and visualize category popularity per cluster."""
    # Merge cluster information with the merged_df
    merged_with_clusters = merged_df.merge(
        user_table[['user_id', 'cluster']],
        on='user_id',
        how='left'
    )

    # Filter out empty category strings
    merged_with_clusters = merged_with_clusters[merged_with_clusters['category_str'] != '']

    # Calculate category counts per cluster
    category_counts = merged_with_clusters.groupby(
        ['cluster', 'category_str']).size().reset_index(name='count')

    # Calculate total articles per cluster
    cluster_totals = category_counts.groupby(
        'cluster')['count'].sum().reset_index()

    # Calculate relative popularity (percentage) of each category within each cluster
    category_popularity = category_counts.merge(
        cluster_totals,
        on='cluster',
        suffixes=('', '_total')
    )
    category_popularity['percentage'] = (
        category_popularity['count'] / category_popularity['count_total'] * 100).round(2)

    # Get top 5 categories for each cluster
    top_categories = category_popularity.sort_values(
        ['cluster', 'percentage'], ascending=[True, False])
    top_categories = top_categories.groupby('cluster').head(5)

    # Create a heatmap of top categories per cluster
    pivot_data = top_categories.pivot(
        index='cluster',
        columns='category_str',
        values='percentage'
    )

    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Top 5 Categories per Cluster (Percentage)')
    plt.xlabel('Category')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(
        f'results/user_clusters/{result_folder}/category_popularity_heatmap.png')
    plt.close()

    # Save detailed category popularity data
    category_popularity.to_csv(
        f'results/user_clusters/{result_folder}/category_popularity.csv', index=False)

    # Print top categories for each cluster
    print("\nTop Categories per Cluster:")
    print("=" * 80)
    for cluster in sorted(category_popularity['cluster'].unique()):
        cluster_data = category_popularity[category_popularity['cluster'] == cluster]
        top_5 = cluster_data.nlargest(5, 'percentage')
        print(f"\nCluster {cluster}:")
        for _, row in top_5.iterrows():
            print(f"  {row['category_str']}: {row['percentage']:.1f}%")


def create_cluster_files(merged_df: pd.DataFrame, user_table: pd.DataFrame) -> None:
    """Create separate files for each cluster's data and summary statistics.

    For each cluster, creates:
    1. A CSV file with all merged data for that cluster (cluster_{index}_merged.csv)
    2. A summary text file with key statistics (cluster_{index}_summary.txt)
    """
    # Create directory if it doesn't exist
    os.makedirs(
        f'results/user_clusters/{result_folder}/cluster_data', exist_ok=True)

    total_users = len(user_table['user_id'].unique())

    # Process each cluster
    for cluster_id in sorted(user_table['cluster'].unique()):
        # Get users in this cluster
        cluster_users = user_table[user_table['cluster'] == cluster_id]

        # Get all behaviors for users in this cluster
        cluster_df = merged_df[merged_df['user_id'].isin(
            cluster_users['user_id'])]

        # Save merged data for this cluster
        cluster_df.to_csv(
            f'results/user_clusters/{result_folder}/cluster_data/cluster_{cluster_id}_merged.csv', index=False)

        # Calculate summary statistics
        num_users = len(cluster_users)
        pct_users = (num_users / total_users) * 100
        avg_reading_time = cluster_users['avg_reading_time'].mean()
        avg_articles_per_session = cluster_users['avg_session_length'].mean()
        avg_categories = cluster_users['count_total_unique_categories'].mean()
        avg_category_switches = cluster_users['avg_category_switches'].mean()

        avg_articles_per_user = cluster_users['count_total_article_impressions'].mean(
        )
        total_reading_time = cluster_df['read_time'].sum()
        avg_total_reading_time = total_reading_time / num_users

        # Calculate top categories
        valid_categories = cluster_df[cluster_df['category_str'] != '']
        category_counts = valid_categories.groupby('category_str').size()
        category_percentages = (
            category_counts / len(valid_categories) * 100).round(2)
        top_categories = category_percentages.nlargest(5)

        # Create summary text file
        with open(f'results/user_clusters/{result_folder}/cluster_data/cluster_{cluster_id}_summary.txt', 'w') as f:
            f.write(f"Cluster {cluster_id} Summary\n")
            f.write("=" * 20 + "\n\n")

            f.write("General Statistics:\n")
            f.write(f"- Number of users: {num_users:.1f}\n")
            f.write(f"- Percentage of total users: {pct_users:.2f}%\n")
            f.write(
                f"- Average reading time: {avg_reading_time:.2f} seconds\n")
            f.write(
                f"- Average articles per session: {avg_articles_per_session:.2f}\n")
            f.write(f"- Average categories read: {avg_categories:.2f}\n")
            f.write(
                f"- Average category switches per session: {avg_category_switches:.2f}\n\n")

            f.write("Additional Metrics:\n")
            f.write(
                f"- Average articles per user: {avg_articles_per_user:.2f}\n")
            f.write(
                f"- Total reading time for cluster: {total_reading_time:.2f} seconds\n")
            f.write(
                f"- Average total reading time per user: {avg_total_reading_time:.2f} seconds\n\n")

            f.write("Top 5 Categories:\n")
            for category, percentage in top_categories.items():
                f.write(f"- {category}: {percentage:.2f}%\n")

    # Create a combined results file
    with open(f'results/user_clusters/{result_folder}/results.txt', 'w') as f:
        f.write("Clustering Results Summary\n")
        f.write("=" * 25 + "\n\n")
        f.write(f"Total number of users: {total_users}\n")
        f.write(
            f"Number of clusters: {len(user_table['cluster'].unique())}\n\n")

        for cluster_id in sorted(user_table['cluster'].unique()):
            cluster_users = user_table[user_table['cluster'] == cluster_id]
            num_users = len(cluster_users)
            pct_users = (num_users / total_users) * 100
            f.write(
                f"Cluster {cluster_id}: {num_users} users ({pct_users:.2f}%)\n")


def analyze_clusters(X_with_clusters: pd.DataFrame, user_table: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    """Analyze and visualize cluster characteristics."""
    # Add cluster labels to user table
    user_table['cluster'] = X_with_clusters['cluster']

    # Calculate cluster sizes
    cluster_sizes = user_table['cluster'].value_counts().sort_index()

    # Calculate mean values for each feature by cluster, excluding user_id
    numeric_columns = user_table.select_dtypes(include=[np.number]).columns
    cluster_means = user_table[numeric_columns].groupby('cluster').mean()

    # Create heatmap of feature means by cluster
    plt.figure(figsize=(15, 10))
    sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Feature Means by Cluster')
    plt.tight_layout()
    plt.savefig(f'results/user_clusters/{result_folder}/cluster_heatmap.png')
    plt.close()

    # Create radar plot for cluster characteristics
    features_for_radar = [
        'count_sessions',
        'avg_category_switches',
        'avg_reading_time',
        'proportion_article_time',
        'avg_session_length',
        'avg_session_duration',
        'percentage_morning',
        'percentage_afternoon',
        'percentage_evening',
        'percentage_night'
    ]

    # Normalize features for radar plot
    radar_data = cluster_means[features_for_radar].copy()
    radar_data = (radar_data - radar_data.min()) / \
        (radar_data.max() - radar_data.min())

    # Create radar plot
    angles = np.linspace(0, 2*np.pi, len(features_for_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the plot

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for cluster in range(len(cluster_means)):
        values = radar_data.iloc[cluster].values
        values = np.concatenate((values, [values[0]]))  # close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_for_radar)
    plt.title('Cluster Characteristics (Radar Plot)')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(f'results/user_clusters/{result_folder}/cluster_radar.png')
    plt.close()

    # Save cluster analysis results
    results = {
        'cluster_sizes': cluster_sizes.to_dict(),
        'cluster_means': cluster_means.to_dict(),
        'feature_importance': dict(zip(X_with_clusters.columns[:-1],
                                       np.abs(X_with_clusters.corr()['cluster']).sort_values(ascending=False)))
    }

    with open(f'results/user_clusters/{result_folder}/cluster_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Create detailed cluster summary
    create_cluster_summary(user_table, merged_df)

    # Create cluster files and summaries
    create_cluster_files(merged_df, user_table)

    # Analyze category popularity
    analyze_category_popularity(merged_df, user_table)


def perform_clustering_analysis(user_table: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    """Main function to perform clustering analysis."""
    # Prepare data
    X_scaled, scaler = prepare_clustering_data(user_table)

    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled)
    print(f"Optimal number of clusters: {optimal_k}")

    # Perform clustering
    kmeans, X_with_clusters = perform_clustering(X_scaled, optimal_k)

    # Analyze clusters
    analyze_clusters(X_with_clusters, user_table, merged_df)

# Main function that loads the data, creates the tables and saves them to csv files


def main():
    merged_df, articles_df = load_data()
    # article_table = create_article_table(merged_df, articles_df)
    user_table = create_user_table(merged_df)

    # Perform clustering analysis
    perform_clustering_analysis(user_table, merged_df)


if __name__ == "__main__":
    main()
