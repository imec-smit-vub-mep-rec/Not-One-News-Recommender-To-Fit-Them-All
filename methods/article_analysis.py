import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the articles and behaviors data."""
    # Load the parquet files
    articles_df = pd.read_parquet('datasets/ekstra/articles.parquet')
    behaviors_df = pd.read_parquet('datasets/ekstra/behaviors.parquet')

    # Drop rows where article_id is NaN in behaviors (homepage views)
    behaviors_df = behaviors_df.dropna(subset=['article_id'])
    
    return articles_df, behaviors_df

def analyze_readtime_vs_pageviews(articles_df: pd.DataFrame) -> None:
    """Create scatter plot of article read time vs total page views."""
    # Drop articles with null values in relevant columns
    filtered_df = articles_df.dropna(subset=['total_read_time', 'total_pageviews'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='total_read_time', y='total_pageviews', alpha=0.5)
    plt.title('Article Read Time vs Total Page Views')
    plt.xlabel('Total Read Time (seconds)')
    plt.ylabel('Total Page Views')
    plt.savefig('readtime_vs_pageviews.png')
    plt.close()

def analyze_user_metrics(behaviors_df: pd.DataFrame) -> None:
    """Create scatter plot of average read time vs average page views by user."""
    # Calculate average metrics per user
    user_metrics = behaviors_df.groupby('user_id').agg({
        'read_time': 'mean',
        'article_ids_clicked': lambda x: sum(len(clicks) if isinstance(clicks, list) else 0 for clicks in x) / len(x)
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=user_metrics, x='read_time', y='article_ids_clicked', alpha=0.5)
    plt.title('Average Read Time vs Average Page Views by User')
    plt.xlabel('Average Read Time (seconds)')
    plt.ylabel('Average Clicks per Session')
    plt.savefig('user_metrics.png')
    plt.close()

def analyze_sentiment_metrics(articles_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> None:
    """Create box plots of read time distribution by sentiment."""
    # Merge behaviors with articles to get sentiment information
    merged_df = behaviors_df.merge(
        articles_df[['article_id', 'sentiment_label']], 
        left_on='article_id', 
        right_on='article_id'
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged_df, x='sentiment_label', y='read_time')
    plt.title('Read Time Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Read Time (seconds)')
    plt.savefig('sentiment_metrics.png')
    plt.close()

def analyze_article_type_metrics(articles_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> None:
    """Create box plots of normalized read time by article type."""
    # Merge behaviors with articles
    merged_df = behaviors_df.merge(
        articles_df[['article_id', 'article_type', 'body']], 
        left_on='article_id', 
        right_on='article_id'
    )
    
    # Calculate article length (character count)
    merged_df['article_length'] = merged_df['body'].str.len()
    
    # Normalize read time by article length
    merged_df['normalized_read_time'] = merged_df['read_time'] / merged_df['article_length']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df, x='article_type', y='normalized_read_time')
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Read Time by Article Type')
    plt.xlabel('Article Type')
    plt.ylabel('Normalized Read Time (seconds/character)')
    plt.tight_layout()
    plt.savefig('article_type_metrics.png')
    plt.close()

def main():
    # Set the style for all plots
    sns.set_style("whitegrid")
    
    # Load and preprocess data
    articles_df, behaviors_df = load_and_preprocess_data()
    
    # Generate all visualizations
    analyze_readtime_vs_pageviews(articles_df)
    analyze_user_metrics(behaviors_df)
    analyze_sentiment_metrics(articles_df, behaviors_df)
    analyze_article_type_metrics(articles_df, behaviors_df)

if __name__ == "__main__":
    main()
