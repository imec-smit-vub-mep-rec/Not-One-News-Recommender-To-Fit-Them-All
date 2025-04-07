import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.dataset import load_datasets

# Load datasets
folder = "datasets/ekstra"
[behaviors, history, articles] = load_datasets(
    folder, ["behaviors", "history", "articles"], "parquet")


def create_and_visualize_clusters(features, feature_names, cluster_type, n_clusters=4):
    """Helper function to create, evaluate and visualize clusters"""
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Evaluate
    sil_score = silhouette_score(scaled_features, clusters)

    # Create DataFrame with results
    results_df = pd.DataFrame(features, columns=feature_names)
    results_df['Cluster'] = clusters

    # Plotting
    plt.figure(figsize=(15, 5))

    # 1. Scatter plot with PCA
    plt.subplot(1, 2, 1)

    # Check if we have enough features for PCA
    n_components = min(2, features.shape[1], features.shape[0])
    if n_components >= 2:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(scaled_features)
        reduced_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
        reduced_df['Cluster'] = clusters
        sns.scatterplot(data=reduced_df, x='PC1', y='PC2',
                        hue='Cluster', palette='viridis')
    else:
        # If we can't do PCA, create a simple 1D scatter plot
        plt.scatter(scaled_features[:, 0], [0] * len(scaled_features),
                    c=clusters, cmap='viridis')

    plt.title(f'{cluster_type} Clusters\nSilhouette Score: {sil_score:.3f}')

    # 2. Cluster characteristics
    plt.subplot(1, 2, 2)
    cluster_means = results_df.groupby('Cluster')[feature_names].mean()
    sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title(f'{cluster_type} Cluster Characteristics')

    plt.tight_layout()
    plt.show()

    # Log cluster statistics
    print(f"\n=== {cluster_type} Clustering Results ===")
    print(f"Silhouette Score: {sil_score:.3f}")
    print("\nCluster Sizes:")
    print(pd.Series(clusters).value_counts().sort_index())
    print("\nCluster Characteristics (Mean Values):")
    print(cluster_means)

    return clusters, sil_score


def cluster_users(behaviors: pd.DataFrame, history: pd.DataFrame, articles: pd.DataFrame):
    # Get user_ids that appear more than once
    frequent_users = behaviors['user_id'].value_counts()
    frequent_users = frequent_users[frequent_users > 1].index

    # Filter behaviors to only include frequent users and merge with articles
    behaviors_expanded = behaviors[behaviors['article_id'].notna()].merge(
        articles[["article_id", "sentiment_label", "category_str", "premium"]],
        on="article_id",
        how="left"
    )
    behaviors_expanded = behaviors_expanded[behaviors_expanded['user_id'].isin(
        frequent_users)]

    # 1. Time-based Clustering
    time_features = behaviors_expanded.groupby("user_id").agg({
        "impression_time": [
            lambda x: pd.to_datetime(x).dt.hour.mean(),  # Average reading hour
            lambda x: pd.to_datetime(x).dt.dayofweek.mean(),  # Avg day of week
            lambda x: pd.to_datetime(x).diff().mean(
            ).total_seconds() / 3600,  # Hours between reads
        ]
    }).fillna(0)
    time_features.columns = ['Avg Hour', 'Avg Day', 'Hours Between Reads']

    time_clusters, time_score = create_and_visualize_clusters(
        time_features,
        time_features.columns,
        'Temporal'
    )

    # 2. Content Preference Clustering
    content_features = behaviors_expanded.groupby("user_id").agg({
        "category_str": lambda x: x.nunique(),  # Category diversity
        "premium": "mean",  # Premium content ratio
        "sentiment_label": lambda x: x.nunique(),  # Sentiment diversity
        # Average read time
        "read_time": lambda x: np.mean([t for t in x if t is not None])
    }).fillna(0)

    content_clusters, content_score = create_and_visualize_clusters(
        content_features,
        ['Category Diversity', 'Premium Ratio',
            'Sentiment Diversity', 'Avg Read Time'],
        'Content Preference'
    )

    # 3. Engagement Depth Clustering
    engagement_features = behaviors_expanded.groupby("user_id").agg({
        "read_time": [
            # Average read time
            lambda x: np.mean([t for t in x if t is not None]),
            lambda x: np.std([t for t in x if t is not None]
                             )    # Read time variation
        ],
        "scroll_percentage": [
            # Average scroll
            lambda x: np.mean([s for s in x if s is not None]),
            lambda x: np.std([s for s in x if s is not None]
                             )    # Scroll variation
        ]
    }).fillna(0)
    engagement_features.columns = ['Avg Read Time',
                                   'Read Time Std', 'Avg Scroll', 'Scroll Std']

    engagement_clusters, engagement_score = create_and_visualize_clusters(
        engagement_features,
        engagement_features.columns,
        'Engagement Depth'
    )

    # 4. Session Behavior Clustering
    session_features = behaviors.groupby("user_id").agg({
        "session_id": "nunique",  # Number of unique sessions
        "device_type": "nunique",  # Number of different devices used
        "is_subscriber": "first",  # Subscriber status
        "article_id": lambda x: x.notna().sum()  # Number of articles read
    }).fillna(0)

    session_clusters, session_score = create_and_visualize_clusters(
        session_features,
        ['Session Count', 'Device Diversity', 'Sub Status', 'Articles Read'],
        'Session Behavior'
    )

    # Compare clustering methods
    clustering_comparison = pd.DataFrame({
        'Clustering Method': ['Temporal', 'Content', 'Engagement', 'Session'],
        'Silhouette Score': [time_score, content_score, engagement_score, session_score]
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(data=clustering_comparison,
                x='Clustering Method', y='Silhouette Score')
    plt.title('Clustering Method Comparison')
    plt.ylim(0, 1)
    plt.show()

    return {
        'temporal': time_clusters,
        'content': content_clusters,
        'engagement': engagement_clusters,
        'session': session_clusters,
        'comparison': clustering_comparison
    }


def analyze_temporal_patterns(behaviors: pd.DataFrame, articles: pd.DataFrame):
    """
    Analyzes temporal patterns in user engagement across different categories and times of day.
    Includes analysis of device types, subscription status, and demographics.
    """
    # Convert timestamp to datetime and extract time components
    behaviors['hour'] = pd.to_datetime(behaviors['impression_time']).dt.hour
    behaviors['day_of_week'] = pd.to_datetime(
        behaviors['impression_time']).dt.day_name()

    # Define time periods
    behaviors['time_period'] = pd.cut(
        behaviors['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)',
                'Afternoon (12-18)', 'Evening (18-24)']
    )

    # Merge with articles to get category information
    engagement_data = behaviors.merge(
        articles[['article_id', 'category_str',
                  'premium', 'sentiment_label']],
        left_on='article_id',
        right_on='article_id',
        how='left'
    )

    # Analysis components
    analyses = {
        'category_time': pd.crosstab(
            engagement_data['time_period'],
            engagement_data['category_str'],
            values=engagement_data['read_time'],
            aggfunc='mean'
        ),

        'device_distribution': pd.crosstab(
            engagement_data['time_period'],
            engagement_data['device_type'],
            normalize='index'
        ).rename(columns={0: 'Unknown', 1: 'Desktop', 2: 'Mobile', 3: 'Tablet'}),

        'subscription_engagement': pd.pivot_table(
            engagement_data,
            values=['read_time', 'scroll_percentage'],
            index='time_period',
            columns='is_subscriber',
            aggfunc='mean'
        ),

        'sentiment_distribution': pd.crosstab(
            engagement_data['time_period'],
            engagement_data['sentiment_label'],
            normalize='index'
        )
    }

    # Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Category engagement by time
    analyses['category_time'].plot(kind='bar', ax=ax1)
    ax1.set_title('Average Read Time by Category and Time Period')
    ax1.set_ylabel('Average Read Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Device type distribution
    analyses['device_distribution'].plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Device Type Distribution by Time Period')
    ax2.set_ylabel('Proportion')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Subscriber vs Non-subscriber engagement
    analyses['subscription_engagement']['read_time'].plot(kind='bar', ax=ax3)
    ax3.set_title('Read Time: Subscribers vs Non-subscribers')
    ax3.set_ylabel('Average Read Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Sentiment distribution
    analyses['sentiment_distribution'].plot(kind='bar', stacked=True, ax=ax4)
    ax4.set_title('Content Sentiment Distribution by Time Period')
    ax4.set_ylabel('Proportion')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Print additional insights
    print("\nKey Metrics by Time Period:")
    for period in behaviors['time_period'].unique():
        period_data = engagement_data[engagement_data['time_period'] == period]
        print(f"\n{period}:")
        print(
            f"Average Read Time: {period_data['read_time'].mean():.2f} seconds")
        print(
            f"Average Scroll Percentage: {period_data['scroll_percentage'].mean():.2f}%")
        print(
            f"Most Popular Category: {period_data['category_string'].mode().iloc[0]}")

    return analyses


def analyze_advanced_patterns(behaviors: pd.DataFrame, articles: pd.DataFrame):
    """
    Analyzes additional clustering patterns in user behavior:
    - Most read topics by weekday
    - Read time patterns by user frequency
    - Gender-based reading patterns
    - Topic preferences by user engagement level
    """
    # Merge behaviors with articles to get topic information
    behaviors_expanded = behaviors[behaviors['article_id'].notna()].merge(
        articles[["article_id", "category_str", "sentiment_label", "premium"]],
        on="article_id",
        how="left"
    )

    # Add datetime components
    behaviors_expanded['datetime'] = pd.to_datetime(
        behaviors_expanded['impression_time'])
    behaviors_expanded['weekday'] = behaviors_expanded['datetime'].dt.day_name()

    # Classify users by reading frequency
    user_frequency = behaviors_expanded.groupby('user_id').size()
    frequency_labels = pd.qcut(user_frequency, q=4, labels=[
        'Irregular', 'Sometimes', 'Regular', 'Frequent'])
    user_frequency_df = pd.DataFrame({
        'user_id': user_frequency.index,
        'frequency_group': frequency_labels
    }).reset_index(drop=True)  # Reset the index to avoid the ambiguity

    behaviors_expanded = behaviors_expanded.merge(
        user_frequency_df, on='user_id', how='left')

    # 1. Most read topics by weekday
    topic_weekday = pd.crosstab(
        behaviors_expanded['weekday'],
        behaviors_expanded['category_str'],
        values=behaviors_expanded['read_time'],
        aggfunc='count'
    ).fillna(0)

    # 2. Average read time by user frequency
    read_time_by_frequency = behaviors_expanded.groupby(
        'frequency_group')['read_time'].agg(['mean', 'std']).round(2)

    # 3. Gender-based reading patterns
    gender_patterns = behaviors_expanded.groupby(['gender', 'category_str']).agg({
        'read_time': ['mean', 'count'],
        'scroll_percentage': 'mean'
    }).round(2)

    # 4. Topic preferences by user engagement level
    topic_by_frequency = pd.crosstab(
        [behaviors_expanded['frequency_group'],
            behaviors_expanded['category_str']],
        columns='count',
        values=behaviors_expanded['read_time'],
        aggfunc='mean'
    ).round(2)

    # Visualizations
    plt.figure(figsize=(20, 15))

    # 1. Topic heatmap by weekday
    plt.subplot(2, 2, 1)
    sns.heatmap(topic_weekday, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Most Read Topics by Weekday')
    plt.xticks(rotation=45)

    # 2. Read time by user frequency
    plt.subplot(2, 2, 2)
    read_time_by_frequency['mean'].plot(
        kind='bar', yerr=read_time_by_frequency['std'])
    plt.title('Average Read Time by User Frequency')
    plt.ylabel('Seconds')
    plt.xticks(rotation=45)

    # 3. Gender reading patterns
    plt.subplot(2, 2, 3)
    # Create a mapping for gender labels
    gender_map = {0: 'Female', 1: 'Male'}
    # Create a copy of the data with mapped gender labels
    gender_patterns_mapped = gender_patterns.copy()
    gender_patterns_mapped.index = gender_patterns_mapped.index.set_levels(
        [gender_patterns_mapped.index.levels[0].map(lambda x: gender_map.get(x, 'Unknown')),
         gender_patterns_mapped.index.levels[1]]
    )
    gender_patterns_pivot = gender_patterns_mapped['read_time']['mean'].unstack(
    )
    sns.heatmap(gender_patterns_pivot, cmap='YlOrRd', annot=True, fmt='.1f')
    plt.title('Average Read Time by Gender and Category')
    plt.xticks(rotation=45)

    # 4. Topic preferences by user frequency
    plt.subplot(2, 2, 4)
    topic_by_frequency_pivot = topic_by_frequency.unstack(level=0)
    sns.heatmap(topic_by_frequency_pivot, cmap='YlOrRd', annot=True, fmt='.1f')
    plt.title('Average Read Time by User Frequency and Category')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Print additional insights
    print("\nKey Insights:")
    print("\n1. Most Active Reading Days:")
    print(topic_weekday.sum().sort_values(ascending=False).head())

    print("\n2. User Frequency Patterns:")
    print(read_time_by_frequency)

    print("\n3. Gender Reading Patterns:")
    print("\nMost Popular Categories by Gender:")
    for gender in gender_patterns.index.get_level_values(0).unique():
        if pd.notna(gender):  # Skip NaN gender
            gender_label = gender_map.get(gender, 'Unknown')
            print(f"\n{gender_label}:")
            gender_data = gender_patterns.loc[gender]
            print(gender_data['read_time']['mean'].sort_values(
                ascending=False).head())

    print("\n4. User Frequency Topic Preferences:")
    for freq in topic_by_frequency.index.get_level_values(0).unique():
        print(f"\n{freq} readers - Top categories by read time:")
        print(topic_by_frequency.loc[freq].sort_values(
            'count', ascending=False).head())

    return {
        'topic_weekday': topic_weekday,
        'read_time_by_frequency': read_time_by_frequency,
        'gender_patterns': gender_patterns,
        'topic_by_frequency': topic_by_frequency
    }


def analyze_subscription_time_patterns(behaviors: pd.DataFrame, articles: pd.DataFrame):
    """
    Analyzes reading time patterns for subscribers vs non-subscribers across different
    time periods and topics, and examines article publication patterns.
    """
    # Create a copy of the dataframes to avoid modifying the originals
    behaviors = behaviors.copy()
    articles = articles.copy()

    # Convert timestamps to datetime
    behaviors['datetime'] = pd.to_datetime(behaviors['impression_time'])
    articles['datetime'] = pd.to_datetime(articles['published_time'])

    # Extract hours from datetime
    behaviors['hour'] = behaviors['datetime'].dt.hour
    articles['hour'] = articles['datetime'].dt.hour

    def assign_time_period(hour):
        if 0 <= hour < 6:
            return 'Night (0-6)'
        elif 6 <= hour < 12:
            return 'Morning (6-12)'
        elif 12 <= hour < 18:
            return 'Afternoon (12-18)'
        else:
            return 'Evening (18-24)'

    # Merge behaviors with articles
    reading_patterns = behaviors.merge(
        articles[['article_id', 'category_str']],  # Remove datetime from here
        on='article_id',
        how='left'
    )

    # Use impression_time from behaviors for time period calculation
    # datetime was already created from impression_time
    reading_patterns['hour'] = reading_patterns['datetime'].dt.hour
    reading_patterns['time_period'] = reading_patterns['hour'].map(
        assign_time_period)

    # Calculate average reading time by subscription status, time period, and category
    avg_reading_time = reading_patterns.groupby(
        ['time_period', 'category_str', 'is_subscriber']
    )['read_time'].agg(['mean', 'count']).round(2).reset_index()

    # Calculate article publication patterns
    articles['time_period'] = articles['hour'].map(assign_time_period)
    publication_patterns = articles.groupby(
        ['time_period', 'category_str']
    ).size().reset_index(name='article_count')

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 1. Subscriber reading time by category and time period
    sub_data = avg_reading_time[avg_reading_time['is_subscriber'] == 1]
    sns.barplot(
        data=sub_data,
        x='category_str',
        y='mean',
        hue='time_period',
        ax=ax1,
        palette='YlOrRd'
    )
    ax1.set_title('Average Reading Time by Category and Time Period - Subscribers')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Reading Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Non-subscriber reading time by category and time period
    non_sub_data = avg_reading_time[avg_reading_time['is_subscriber'] == 0]
    sns.barplot(
        data=non_sub_data,
        x='category_str',
        y='mean',
        hue='time_period',
        ax=ax2,
        palette='YlOrRd'
    )
    ax2.set_title('Average Reading Time by Category and Time Period - Non-Subscribers')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Reading Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Print insights
    print("\nKey Insights:")
    
    print("\n1. Average Reading Times by Time Period and Subscription Status:")
    for is_sub in [0, 1]:
        status = "Subscribers" if is_sub == 1 else "Non-subscribers"
        print(f"\n{status}:")
        for period in avg_reading_time['time_period'].unique():
            period_data = avg_reading_time[
                (avg_reading_time['time_period'] == period) & 
                (avg_reading_time['is_subscriber'] == is_sub)
            ]
            print(f"\n{period}:")
            print(period_data.groupby('category_str')['mean'].mean().sort_values(ascending=False).head())

    print("\n2. Biggest Differences in Reading Time (Subscribers vs Non-subscribers):")
    diff_summary = (avg_reading_time[avg_reading_time['is_subscriber'] == 1]['mean'] - avg_reading_time[avg_reading_time['is_subscriber'] == 0]['mean']).sort_values(ascending=False)
    print("\nCategories where subscribers spend more time:")
    print(diff_summary[diff_summary > 0].head())
    print("\nCategories where non-subscribers spend more time:")
    print(diff_summary[diff_summary < 0].head())

    return {
        'reading_patterns': avg_reading_time,
        'publication_patterns': publication_patterns,
        'subscriber_patterns': sub_data,
        'non_subscriber_patterns': non_sub_data,
        'difference_patterns': diff_summary
    }


# cluster_users(behaviors, history, articles)
# analyze_temporal_patterns(behaviors, articles)
# analyze_advanced_patterns(behaviors, articles)
# analyze_subscription_time_patterns(behaviors, articles)
analyze_subscription_time_patterns(behaviors, articles)
