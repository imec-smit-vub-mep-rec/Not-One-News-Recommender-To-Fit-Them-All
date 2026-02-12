"""
Data transformation utilities for the RICON analysis pipeline.

Provides functions to transform data between different formats
required by various pipeline stages.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

from ..config.schema import ColumnNames
from ..utils.logging import get_logger
from ..utils.io import save_dataframe


logger = get_logger("preprocessing.transformers")


def behaviors_to_interactions(
    behaviors_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'article_id',
    time_col: str = 'impression_time',
) -> pd.DataFrame:
    """Convert behaviors/impressions to RecPack interactions format.
    
    Removes homepage views and invalid entries, keeping only
    (user_id, article_id, impression_time) tuples.
    
    Args:
        behaviors_df: Behaviors/impressions DataFrame
        user_col: Name of user ID column
        item_col: Name of article/item ID column
        time_col: Name of timestamp column
        
    Returns:
        Interactions DataFrame ready for RecPack
    """
    logger.info("Converting behaviors to interactions format...")
    
    initial_count = len(behaviors_df)
    
    # Filter out invalid rows
    df = behaviors_df.copy()
    
    # Remove null article IDs
    df = df[df[item_col].notna()]
    
    # Remove homepage and empty articles
    df = df[~df[item_col].isin(['homepage', 'empty', '', '-1'])]
    
    # Remove null user IDs
    df = df[df[user_col].notna()]
    df = df[df[user_col] != '']
    
    # Remove null timestamps
    df = df[df[time_col].notna()]
    
    # Keep only required columns
    df = df[[user_col, item_col, time_col]].copy()
    
    # Ensure correct data types
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    
    # Handle timestamp conversion
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        # Convert datetime to Unix timestamp (seconds)
        df[time_col] = df[time_col].astype('int64') // 10**9
    else:
        # Normalize timestamp to seconds if in milliseconds
        max_time = df[time_col].max()
        if pd.notna(max_time) and max_time > 10**12:
            df[time_col] = df[time_col] // 1000
        df[time_col] = df[time_col].astype(np.int64)
    
    # Log statistics
    removed = initial_count - len(df)
    logger.info(f"Removed {removed} invalid rows ({removed/initial_count*100:.1f}%)")
    logger.info(f"Final interactions: {len(df)}")
    logger.info(f"Unique users: {df[user_col].nunique()}")
    logger.info(f"Unique items: {df[item_col].nunique()}")
    
    # Calculate per-user and per-item statistics
    user_counts = df.groupby(user_col).size()
    item_counts = df.groupby(item_col).size()
    
    logger.info(f"Interactions per user: mean={user_counts.mean():.1f}, min={user_counts.min()}, max={user_counts.max()}")
    logger.info(f"Interactions per item: mean={item_counts.mean():.1f}, min={item_counts.min()}, max={item_counts.max()}")
    
    return df


def articles_to_content(
    articles_df: pd.DataFrame,
    article_col: str = 'article_id',
    category_col: str = 'category_str',
    title_col: str = 'title',
    body_col: str = 'body',
    template: str = 'query: {category}: {title}',
    full_content: bool = False,
) -> pd.DataFrame:
    """Create content strings for content-based recommendations.
    
    By default, combines category and title into a single content string
    suitable for sentence transformer embedding (matching legacy behavior).
    
    With full_content=True, includes the article body for richer embeddings.
    
    Args:
        articles_df: Articles DataFrame
        article_col: Name of article ID column
        category_col: Name of category column
        title_col: Name of title column
        body_col: Name of body/text column (used only if full_content=True)
        template: Template for combining fields. Use {category}, {title}, {body} placeholders.
                  Default: 'query: {category}: {title}' (legacy format)
        full_content: If True, use full content template with body.
                      If False (default), use legacy template (category + title only).
        
    Returns:
        DataFrame with article_id and content columns
    """
    logger.info("Creating article content strings...")
    
    df = articles_df.copy()
    
    # Override template if full_content is requested
    if full_content:
        template = 'query: {category}: {title}. {body}'
        logger.info("Using FULL CONTENT mode (category + title + body)")
    else:
        logger.info("Using LEGACY mode (category + title only)")
    
    # Ensure columns exist
    if category_col not in df.columns:
        logger.warning(f"Column '{category_col}' not found, using empty string")
        df[category_col] = ''
    
    if title_col not in df.columns:
        logger.warning(f"Column '{title_col}' not found, using empty string")
        df[title_col] = ''
    
    if body_col not in df.columns:
        if full_content:
            logger.warning(f"Column '{body_col}' not found, using empty string for body")
        df[body_col] = ''
    
    # Fill NA values
    df[category_col] = df[category_col].fillna('').astype(str)
    df[title_col] = df[title_col].fillna('').astype(str)
    df[body_col] = df[body_col].fillna('').astype(str)
    
    # Truncate body to avoid overly long content (limit to ~500 chars like legacy)
    if full_content:
        df[body_col] = df[body_col].str[:500]
    
    # Create content strings with vectorized operations for large datasets.
    if full_content:
        content = (
            'query: '
            + df[category_col]
            + ': '
            + df[title_col]
            + '. '
            + df[body_col]
        )
    elif template == 'query: {category}: {title}':
        content = 'query: ' + df[category_col] + ': ' + df[title_col]
    else:
        # Fallback for custom templates.
        content = df.apply(
            lambda row: template.format(
                category=row[category_col],
                title=row[title_col],
                body=row[body_col] if full_content else ''
            ),
            axis=1
        )
    
    result = pd.DataFrame({
        'article_id': df[article_col].astype(str),
        'content': content,
    })
    
    logger.info(f"Created content for {len(result)} articles")
    
    return result


def add_time_features(
    df: pd.DataFrame,
    time_col: str = 'impression_time',
    prefix: str = '',
) -> pd.DataFrame:
    """Add time-of-day features to a DataFrame.
    
    Adds columns for time period (morning, afternoon, evening, night)
    and day of week.
    
    Args:
        df: DataFrame with timestamp column
        time_col: Name of timestamp column
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Convert timestamp to datetime if needed
    if df[time_col].dtype in ['int64', 'float64']:
        # Assume milliseconds if large values
        divisor = 1000 if df[time_col].max() > 10**12 else 1
        dt = pd.to_datetime(df[time_col] // divisor, unit='s')
    else:
        dt = pd.to_datetime(df[time_col])
    
    # Extract hour
    hour = dt.dt.hour
    
    # Time period features
    df[f'{prefix}is_morning'] = (hour >= 6) & (hour < 12)
    df[f'{prefix}is_afternoon'] = (hour >= 12) & (hour < 18)
    df[f'{prefix}is_evening'] = (hour >= 18) & (hour < 24)
    df[f'{prefix}is_night'] = (hour >= 0) & (hour < 6)
    
    # Day of week (0 = Monday, 6 = Sunday)
    df[f'{prefix}day_of_week'] = dt.dt.dayofweek
    df[f'{prefix}is_weekend'] = dt.dt.dayofweek >= 5
    
    return df


def merge_with_articles(
    behaviors_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    article_col: str = 'article_id',
    article_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge behaviors with article information.
    
    Args:
        behaviors_df: Behaviors DataFrame
        articles_df: Articles DataFrame
        article_col: Name of article ID column
        article_columns: Specific columns to include from articles (None = category_str only)
        
    Returns:
        Merged DataFrame
    """
    if article_columns is None:
        article_columns = ['category_str']
    
    # Ensure article_id is in the columns
    columns_to_merge = [article_col] + [c for c in article_columns if c != article_col]
    columns_to_merge = [c for c in columns_to_merge if c in articles_df.columns]
    
    # Select only needed columns from articles
    articles_subset = articles_df[columns_to_merge].copy()
    
    # Merge
    merged = behaviors_df.merge(
        articles_subset,
        on=article_col,
        how='left'
    )
    
    logger.info(f"Merged {len(merged)} behaviors with article info")
    
    return merged


class DataTransformer:
    """Comprehensive data transformer for the pipeline.
    
    Combines multiple transformation steps into a single interface.
    """
    
    def __init__(
        self,
        content_template: str = 'query: {category}: {title}',
        min_items_per_user: int = 5,
        min_users_per_item: int = 1,
    ):
        """Initialize the transformer.
        
        Args:
            content_template: Template for article content strings
            min_items_per_user: Minimum items for RecPack filtering
            min_users_per_item: Minimum users for RecPack filtering
        """
        self.content_template = content_template
        self.min_items_per_user = min_items_per_user
        self.min_users_per_item = min_users_per_item
    
    def create_recpack_data(
        self,
        behaviors_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> tuple:
        """Create all data files needed for RecPack evaluation.
        
        Args:
            behaviors_df: Behaviors DataFrame
            articles_df: Articles DataFrame
            output_dir: Optional directory to save files
            
        Returns:
            Tuple of (interactions_df, articles_content_df)
        """
        # Create interactions
        interactions_df = behaviors_to_interactions(behaviors_df)
        
        # Create article content
        articles_content_df = articles_to_content(
            articles_df,
            template=self.content_template
        )
        
        # Save if output directory provided
        if output_dir:
            from pathlib import Path
            from ..utils.io import ensure_dir
            
            ensure_dir(output_dir)
            
            interactions_path = Path(output_dir) / 'interactions.csv'
            content_path = Path(output_dir) / 'articles_content.csv'
            
            save_dataframe(interactions_df, str(interactions_path), format='csv')
            save_dataframe(articles_content_df, str(content_path), format='csv')
            
            logger.info(f"Saved interactions to {interactions_path}")
            logger.info(f"Saved article content to {content_path}")
        
        return interactions_df, articles_content_df
    
    def prepare_clustering_data(
        self,
        behaviors_df: pd.DataFrame,
        articles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare data for clustering by merging behaviors with articles.
        
        Args:
            behaviors_df: Behaviors DataFrame
            articles_df: Articles DataFrame
            
        Returns:
            Merged DataFrame with time features
        """
        # Merge with article info
        merged = merge_with_articles(
            behaviors_df,
            articles_df,
            article_columns=['category_str']
        )
        
        # Add time features
        merged = add_time_features(merged)
        
        return merged
