"""
Data cleaning utilities for the RICON analysis pipeline.

Provides functions to clean and filter data according to various criteria.
"""

from typing import Optional, List, Set
import pandas as pd
import numpy as np

from ..utils.logging import get_logger


logger = get_logger("preprocessing.cleaners")


def remove_empty_articles(
    impressions_df: pd.DataFrame,
    articles_df: Optional[pd.DataFrame] = None,
    article_col: str = 'article_id',
) -> pd.DataFrame:
    """Remove impressions with empty or invalid article IDs.
    
    Args:
        impressions_df: Impressions DataFrame
        articles_df: Optional articles DataFrame to validate against
        article_col: Name of the article ID column
        
    Returns:
        Filtered impressions DataFrame
    """
    initial_count = len(impressions_df)
    
    # Remove null article IDs
    df = impressions_df[impressions_df[article_col].notna()].copy()
    
    # Convert to string (handling float -> int -> str to avoid ".0" suffix)
    if pd.api.types.is_float_dtype(df[article_col]):
        # Convert float to int first (for non-null values), then to string
        df[article_col] = df[article_col].astype('Int64').astype(str)
    else:
        df[article_col] = df[article_col].astype(str)
    
    # Remove empty strings
    df = df[df[article_col] != '']
    df = df[df[article_col] != '<NA>']
    
    # Remove 'homepage' entries if present
    df = df[df[article_col] != 'homepage']
    
    # Remove 'empty' entries if present
    df = df[df[article_col] != 'empty']
    
    # If articles DataFrame provided, only keep valid article IDs
    if articles_df is not None:
        # Ensure same type for comparison (convert both to string)
        if pd.api.types.is_float_dtype(articles_df[article_col]):
            valid_articles = set(articles_df[article_col].astype('Int64').astype(str).unique())
        else:
            valid_articles = set(articles_df[article_col].astype(str).unique())
        df = df[df[article_col].isin(valid_articles)]
    
    removed_count = initial_count - len(df)
    logger.info(f"Removed {removed_count} impressions with empty/invalid articles ({removed_count/initial_count*100:.1f}%)")
    
    return df


def remove_invalid_sessions(
    df: pd.DataFrame,
    min_impressions: int = 2,
    session_col: str = 'session_id',
) -> pd.DataFrame:
    """Remove sessions with too few impressions.
    
    Args:
        df: Impressions DataFrame
        min_impressions: Minimum impressions per session
        session_col: Name of the session ID column
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    initial_sessions = df[session_col].nunique()
    
    # Count impressions per session
    session_counts = df.groupby(session_col).size()
    
    # Find valid sessions
    valid_sessions = session_counts[session_counts >= min_impressions].index
    
    # Filter
    df = df[df[session_col].isin(valid_sessions)].copy()
    
    removed_sessions = initial_sessions - df[session_col].nunique()
    removed_impressions = initial_count - len(df)
    
    logger.info(f"Removed {removed_sessions} sessions with < {min_impressions} impressions")
    logger.info(f"Removed {removed_impressions} impressions total")
    
    return df


def remove_outlier_users(
    df: pd.DataFrame,
    min_impressions: int = 5,
    max_impressions: Optional[int] = None,
    user_col: str = 'user_id',
) -> pd.DataFrame:
    """Remove users with too few or too many impressions.
    
    Args:
        df: Impressions DataFrame
        min_impressions: Minimum impressions per user
        max_impressions: Maximum impressions per user (None = no limit)
        user_col: Name of the user ID column
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    initial_users = df[user_col].nunique()
    
    # Count impressions per user
    user_counts = df.groupby(user_col).size()
    
    # Find valid users
    mask = user_counts >= min_impressions
    if max_impressions is not None:
        mask &= user_counts <= max_impressions
    
    valid_users = user_counts[mask].index
    
    # Filter
    df = df[df[user_col].isin(valid_users)].copy()
    
    removed_users = initial_users - df[user_col].nunique()
    removed_impressions = initial_count - len(df)
    
    logger.info(f"Removed {removed_users} users with impressions not in [{min_impressions}, {max_impressions or 'inf'}]")
    logger.info(f"Removed {removed_impressions} impressions total")
    
    return df


def clean_categories(
    df: pd.DataFrame,
    category_col: str = 'category_str',
    lowercase: bool = True,
    remove_empty: bool = False,
    valid_categories: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Clean and normalize category strings.
    
    Args:
        df: DataFrame with category column
        category_col: Name of the category column
        lowercase: Whether to lowercase categories
        remove_empty: Whether to remove rows with empty categories
        valid_categories: Optional set of valid category values
        
    Returns:
        DataFrame with cleaned categories
    """
    df = df.copy()
    
    if category_col not in df.columns:
        logger.warning(f"Category column '{category_col}' not found")
        return df
    
    # Fill NA with empty string
    df[category_col] = df[category_col].fillna('')
    
    # Strip whitespace
    df[category_col] = df[category_col].str.strip()
    
    # Lowercase if requested
    if lowercase:
        df[category_col] = df[category_col].str.lower()
    
    # Remove empty if requested
    if remove_empty:
        initial_count = len(df)
        df = df[df[category_col] != '']
        logger.info(f"Removed {initial_count - len(df)} rows with empty categories")
    
    # Filter to valid categories if provided
    if valid_categories:
        initial_count = len(df)
        df = df[df[category_col].isin(valid_categories)]
        logger.info(f"Removed {initial_count - len(df)} rows with invalid categories")
    
    return df


def remove_duplicate_impressions(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
) -> pd.DataFrame:
    """Remove duplicate impressions.
    
    Args:
        df: Impressions DataFrame
        subset: Columns to consider for duplicates (None = all columns)
        keep: Which duplicate to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    
    # Default subset for impressions
    if subset is None:
        subset = ['user_id', 'article_id', 'impression_time']
        subset = [col for col in subset if col in df.columns]
    
    df = df.drop_duplicates(subset=subset, keep=keep).copy()
    
    removed = initial_count - len(df)
    logger.info(f"Removed {removed} duplicate impressions")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    numeric_only: bool = True,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """Handle missing values in a DataFrame.
    
    Args:
        df: DataFrame to process
        strategy: Imputation strategy ('mean', 'median', 'zero', 'drop', 'fill')
        numeric_only: Whether to only impute numeric columns
        fill_value: Value to use for 'fill' strategy
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_count - len(df)} rows with missing values")
        return df
    
    # Get columns to impute
    if numeric_only:
        cols = df.select_dtypes(include=[np.number]).columns
    else:
        cols = df.columns
    
    for col in cols:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'zero':
            df[col] = df[col].fillna(0)
        elif strategy == 'fill' and fill_value is not None:
            df[col] = df[col].fillna(fill_value)
        
        logger.debug(f"Imputed {null_count} missing values in '{col}' using {strategy}")
    
    return df


class DataCleaner:
    """Comprehensive data cleaner that applies multiple cleaning steps.
    
    Provides a pipeline-style interface for data cleaning.
    
    IMPORTANT: User filtering (min_impressions_per_user) should typically be
    disabled for clustering (filter_users=False) and only applied for evaluation.
    This ensures clustering happens on ALL users, matching the legacy behavior.
    """
    
    def __init__(
        self,
        min_impressions_per_user: int = 5,
        max_impressions_per_user: Optional[int] = None,
        min_impressions_per_session: int = 1,
        remove_empty_articles: bool = True,
        clean_categories: bool = True,
        remove_duplicates: bool = True,
        filter_users: bool = True,
    ):
        """Initialize the cleaner.
        
        Args:
            min_impressions_per_user: Minimum impressions per user (only applied if filter_users=True)
            max_impressions_per_user: Maximum impressions per user (only applied if filter_users=True)
            min_impressions_per_session: Minimum impressions per session
            remove_empty_articles: Whether to remove empty article entries
            clean_categories: Whether to clean category strings
            remove_duplicates: Whether to remove duplicate impressions
            filter_users: Whether to filter users by impression count.
                          Set to False for clustering (cluster ALL users),
                          set to True for evaluation (filter for RecPack).
        """
        self.min_impressions_per_user = min_impressions_per_user
        self.max_impressions_per_user = max_impressions_per_user
        self.min_impressions_per_session = min_impressions_per_session
        self.remove_empty_articles = remove_empty_articles
        self.clean_categories_flag = clean_categories
        self.remove_duplicates = remove_duplicates
        self.filter_users = filter_users
        
        self.stats = {}
    
    def clean_impressions(
        self,
        df: pd.DataFrame,
        articles_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Clean impressions DataFrame.
        
        Args:
            df: Impressions DataFrame
            articles_df: Optional articles DataFrame for validation
            
        Returns:
            Cleaned DataFrame
        """
        self.stats['initial_rows'] = len(df)
        self.stats['initial_users'] = df['user_id'].nunique()
        
        # Remove empty articles
        if self.remove_empty_articles and 'article_id' in df.columns:
            df = remove_empty_articles(df, articles_df)
        
        # Remove duplicates
        if self.remove_duplicates:
            df = remove_duplicate_impressions(df)
        
        # Remove invalid sessions
        if self.min_impressions_per_session > 1 and 'session_id' in df.columns:
            df = remove_invalid_sessions(df, self.min_impressions_per_session)
        
        # Remove outlier users (only if filter_users is enabled)
        # NOTE: For clustering, this should be DISABLED to cluster ALL users.
        # Filtering should only happen for RecPack evaluation.
        if self.filter_users:
            df = remove_outlier_users(
                df,
                self.min_impressions_per_user,
                self.max_impressions_per_user
            )
        else:
            logger.info("User filtering disabled - keeping ALL users for clustering")
        
        self.stats['final_rows'] = len(df)
        self.stats['final_users'] = df['user_id'].nunique()
        self.stats['retention_rate'] = self.stats['final_rows'] / self.stats['initial_rows']
        
        return df
    
    def clean_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean articles DataFrame.
        
        Args:
            df: Articles DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if self.clean_categories_flag and 'category_str' in df.columns:
            df = clean_categories(df)
        
        # Remove duplicates
        if self.remove_duplicates and 'article_id' in df.columns:
            initial = len(df)
            df = df.drop_duplicates(subset=['article_id'], keep='first')
            logger.info(f"Removed {initial - len(df)} duplicate articles")
        
        return df
    
    def get_stats(self) -> dict:
        """Get cleaning statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
