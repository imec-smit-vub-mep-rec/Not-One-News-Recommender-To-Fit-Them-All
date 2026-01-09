"""
Feature engineering for user clustering.

Extracts user behavior features from impressions data for clustering.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger


logger = get_logger("clustering.feature_engineering")


def create_category_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    category_col: str = 'category_str',
) -> pd.DataFrame:
    """Create category preference features per user.
    
    Computes the proportion of each category in user's history.
    
    Args:
        df: Impressions DataFrame with category information
        user_col: Name of user ID column
        category_col: Name of category column
        
    Returns:
        DataFrame with user_id and category proportion columns
    """
    logger.info("Creating category features...")
    
    # Fill missing categories
    df = df.copy()
    df[category_col] = df[category_col].fillna('unknown')
    
    # Get all unique categories
    categories = df[category_col].unique()
    logger.info(f"Found {len(categories)} unique categories")
    
    # Count impressions per user per category
    category_counts = df.groupby([user_col, category_col]).size().unstack(fill_value=0)
    
    # Normalize to proportions
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    
    # Rename columns with prefix
    category_proportions.columns = [f'cat_{c}' for c in category_proportions.columns]
    
    # Reset index
    category_proportions = category_proportions.reset_index()
    
    logger.info(f"Created {len(category_proportions.columns) - 1} category features for {len(category_proportions)} users")
    
    return category_proportions


def create_time_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    time_col: str = 'impression_time',
) -> pd.DataFrame:
    """Create time-of-day preference features per user.
    
    Computes when users are most active (morning, afternoon, evening, night).
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        time_col: Name of timestamp column
        
    Returns:
        DataFrame with user_id and time preference columns
    """
    logger.info("Creating time features...")
    
    df = df.copy()
    
    # Convert timestamp to datetime
    if df[time_col].dtype in ['int64', 'float64']:
        # Assume milliseconds if large values
        divisor = 1000 if df[time_col].max() > 10**12 else 1
        df['_datetime'] = pd.to_datetime(df[time_col] // divisor, unit='s')
    else:
        df['_datetime'] = pd.to_datetime(df[time_col])
    
    # Extract hour
    df['_hour'] = df['_datetime'].dt.hour
    
    # Create time period columns
    df['is_morning'] = (df['_hour'] >= 6) & (df['_hour'] < 12)
    df['is_afternoon'] = (df['_hour'] >= 12) & (df['_hour'] < 18)
    df['is_evening'] = (df['_hour'] >= 18) & (df['_hour'] < 24)
    df['is_night'] = (df['_hour'] >= 0) & (df['_hour'] < 6)
    
    # Day of week
    df['_dow'] = df['_datetime'].dt.dayofweek
    df['is_weekend'] = df['_dow'] >= 5
    
    # Aggregate per user
    time_features = df.groupby(user_col).agg({
        'is_morning': 'mean',
        'is_afternoon': 'mean',
        'is_evening': 'mean',
        'is_night': 'mean',
        'is_weekend': 'mean',
    }).reset_index()
    
    # Rename columns
    time_features.columns = [user_col, 'time_morning', 'time_afternoon', 
                             'time_evening', 'time_night', 'time_weekend']
    
    logger.info(f"Created 5 time features for {len(time_features)} users")
    
    return time_features


def create_activity_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    session_col: str = 'session_id',
    time_col: str = 'impression_time',
) -> pd.DataFrame:
    """Create user activity features.
    
    Computes total impressions, unique sessions, avg session length, etc.
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        session_col: Name of session ID column
        time_col: Name of timestamp column
        
    Returns:
        DataFrame with user_id and activity columns
    """
    logger.info("Creating activity features...")
    
    # Basic counts
    activity = df.groupby(user_col).agg({
        'article_id': 'nunique',  # Unique articles viewed
    }).rename(columns={'article_id': 'unique_articles'})
    
    # Total impressions
    activity['total_impressions'] = df.groupby(user_col).size()
    
    # Session-based features if session column exists
    if session_col in df.columns:
        session_counts = df.groupby(user_col)[session_col].nunique()
        activity['num_sessions'] = session_counts
        activity['impressions_per_session'] = activity['total_impressions'] / activity['num_sessions']
    
    # Time-based features
    if time_col in df.columns:
        # Engagement span (days)
        time_agg = df.groupby(user_col)[time_col].agg(['min', 'max'])
        
        # Handle datetime vs numeric timestamps
        if pd.api.types.is_datetime64_any_dtype(time_agg['max']):
            # Convert datetime to seconds for calculation
            activity['engagement_span_days'] = (
                (time_agg['max'] - time_agg['min']).dt.total_seconds() / (24 * 3600)
            ).values
        else:
            # Handle milliseconds for numeric timestamps
            if time_agg['max'].max() > 10**12:
                time_agg = time_agg // 1000
            activity['engagement_span_days'] = (time_agg['max'] - time_agg['min']) / (24 * 3600)
    
    activity = activity.reset_index()
    
    logger.info(f"Created {len(activity.columns) - 1} activity features for {len(activity)} users")
    
    return activity


def create_homepage_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    article_col: str = 'article_id',
    read_time_col: str = 'read_time',
) -> pd.DataFrame:
    """Create homepage behavior features per user.
    
    These features capture the distinction between users who primarily
    browse the homepage vs those who read articles. This is an important
    clustering signal from the legacy code.
    
    Args:
        df: Impressions DataFrame (must include homepage views where article_id is null)
        user_col: Name of user ID column
        article_col: Name of article ID column
        read_time_col: Name of read time column
        
    Returns:
        DataFrame with user_id and homepage behavior columns
    """
    logger.info("Creating homepage features...")
    
    homepage_features = []
    
    for user, group in df.groupby(user_col):
        # Count homepage vs article impressions
        homepage_impressions = group[article_col].isna().sum()
        article_impressions = group[article_col].notna().sum()
        total_impressions = len(group)
        
        # Proportion of homepage impressions
        homepage_ratio = homepage_impressions / total_impressions if total_impressions > 0 else 0
        
        # Reading time features (if available)
        if read_time_col in group.columns:
            homepage_rows = group[group[article_col].isna()]
            article_rows = group[group[article_col].notna()]
            
            # Average reading time on homepage
            avg_reading_time_homepage = homepage_rows[read_time_col].mean() if len(homepage_rows) > 0 else 0
            
            # Average reading time on articles
            avg_reading_time_articles = article_rows[read_time_col].mean() if len(article_rows) > 0 else 0
            
            # Total reading time
            total_reading_time = group[read_time_col].sum()
            article_reading_time = article_rows[read_time_col].sum()
            
            # Proportion of time on articles vs homepage
            proportion_article_time = article_reading_time / total_reading_time if total_reading_time > 0 else 0
        else:
            avg_reading_time_homepage = 0
            avg_reading_time_articles = 0
            proportion_article_time = 0
        
        homepage_features.append({
            user_col: user,
            'homepage_impressions': homepage_impressions,
            'article_impressions': article_impressions,
            'homepage_ratio': homepage_ratio,
            'avg_reading_time_homepage': avg_reading_time_homepage,
            'avg_reading_time_articles': avg_reading_time_articles,
            'proportion_article_time': proportion_article_time,
        })
    
    result = pd.DataFrame(homepage_features)
    
    # Fill NaN values
    result = result.fillna(0)
    
    logger.info(f"Created {len(result.columns) - 1} homepage features for {len(result)} users")
    
    return result


def create_diversity_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    category_col: str = 'category_str',
    article_col: str = 'article_id',
) -> pd.DataFrame:
    """Create content diversity features per user.
    
    Computes entropy and diversity metrics of user preferences.
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        category_col: Name of category column
        article_col: Name of article ID column
        
    Returns:
        DataFrame with user_id and diversity columns
    """
    logger.info("Creating diversity features...")
    
    df = df.copy()
    df[category_col] = df[category_col].fillna('unknown')
    
    def category_entropy(cats):
        """Compute entropy of category distribution."""
        value_counts = cats.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
        return entropy
    
    diversity_features = []
    
    for user, group in df.groupby(user_col):
        # Category entropy
        cat_entropy = category_entropy(group[category_col])
        
        # Number of unique categories
        n_categories = group[category_col].nunique()
        
        # Gini coefficient of category distribution
        cat_counts = group[category_col].value_counts().values
        if len(cat_counts) > 0:
            sorted_counts = np.sort(cat_counts)
            n = len(sorted_counts)
            index = np.arange(1, n + 1)
            gini = (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts) + 1e-10)
        else:
            gini = 0
        
        diversity_features.append({
            user_col: user,
            'category_entropy': cat_entropy,
            'num_categories': n_categories,
            'category_gini': gini,
        })
    
    result = pd.DataFrame(diversity_features)
    
    logger.info(f"Created 3 diversity features for {len(result)} users")
    
    return result


def scale_features(
    features_df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    method: str = 'standard',
) -> Tuple[pd.DataFrame, Any]:
    """Scale feature values.
    
    Args:
        features_df: DataFrame with features
        exclude_cols: Columns to exclude from scaling (e.g., user_id)
        method: Scaling method ('standard' or 'minmax')
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    if exclude_cols is None:
        exclude_cols = ['user_id']
    
    # Separate excluded columns
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Get feature values
    X = features_df[feature_cols].values
    
    # Scale
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # Create scaled DataFrame
    scaled_df = features_df.copy()
    scaled_df[feature_cols] = X_scaled
    
    logger.info(f"Scaled {len(feature_cols)} features using {method} scaling")
    
    return scaled_df, scaler


def create_user_features(
    impressions_df: pd.DataFrame,
    articles_df: Optional[pd.DataFrame] = None,
    include_categories: bool = True,
    include_time: bool = True,
    include_activity: bool = True,
    include_diversity: bool = True,
    include_homepage: bool = True,
    scale: bool = True,
    user_col: str = 'user_id',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create all user features for clustering.
    
    Main entry point for feature engineering.
    
    Args:
        impressions_df: Impressions DataFrame (should include homepage views for homepage features)
        articles_df: Optional articles DataFrame (for category info)
        include_categories: Whether to include category features
        include_time: Whether to include time features
        include_activity: Whether to include activity features
        include_diversity: Whether to include diversity features
        include_homepage: Whether to include homepage behavior features.
                          These capture the distinction between homepage browsers
                          vs article readers, matching the legacy clustering behavior.
        scale: Whether to scale features
        user_col: Name of user ID column
        
    Returns:
        Tuple of (features DataFrame, metadata dict)
    """
    logger.info("Creating user features...")
    
    # Merge with articles if needed for category
    df = impressions_df.copy()
    if articles_df is not None and 'category_str' not in df.columns:
        df = df.merge(
            articles_df[['article_id', 'category_str']].drop_duplicates(),
            on='article_id',
            how='left'
        )
    
    # Get unique users as base
    features = df[[user_col]].drop_duplicates().reset_index(drop=True)
    
    metadata = {'feature_groups': {}}
    
    # Category features
    if include_categories and 'category_str' in df.columns:
        cat_features = create_category_features(df, user_col)
        features = features.merge(cat_features, on=user_col, how='left')
        metadata['feature_groups']['category'] = [c for c in cat_features.columns if c != user_col]
    
    # Time features
    if include_time and 'impression_time' in df.columns:
        time_features = create_time_features(df, user_col)
        features = features.merge(time_features, on=user_col, how='left')
        metadata['feature_groups']['time'] = [c for c in time_features.columns if c != user_col]
    
    # Activity features
    if include_activity:
        activity_features = create_activity_features(df, user_col)
        features = features.merge(activity_features, on=user_col, how='left')
        metadata['feature_groups']['activity'] = [c for c in activity_features.columns if c != user_col]
    
    # Homepage behavior features (legacy clustering signal)
    if include_homepage:
        homepage_features = create_homepage_features(df, user_col)
        features = features.merge(homepage_features, on=user_col, how='left')
        metadata['feature_groups']['homepage'] = [c for c in homepage_features.columns if c != user_col]
    
    # Diversity features
    if include_diversity and 'category_str' in df.columns:
        diversity_features = create_diversity_features(df, user_col)
        features = features.merge(diversity_features, on=user_col, how='left')
        metadata['feature_groups']['diversity'] = [c for c in diversity_features.columns if c != user_col]
    
    # Fill NaN values
    features = features.fillna(0)
    
    # Scale features
    if scale:
        features, scaler = scale_features(features, exclude_cols=[user_col])
        metadata['scaler'] = scaler
    
    # Store feature columns
    metadata['feature_columns'] = [c for c in features.columns if c != user_col]
    metadata['n_features'] = len(metadata['feature_columns'])
    metadata['n_users'] = len(features)
    
    logger.info(f"Created {metadata['n_features']} features for {metadata['n_users']} users")
    
    return features, metadata


class UserFeatureExtractor:
    """Class-based interface for user feature extraction.
    
    Provides a stateful interface that remembers configuration
    and can be reused across datasets.
    """
    
    def __init__(
        self,
        include_categories: bool = True,
        include_time: bool = True,
        include_activity: bool = True,
        include_diversity: bool = True,
        include_homepage: bool = True,
        scale: bool = True,
        user_col: str = 'user_id',
    ):
        """Initialize the feature extractor.
        
        Args:
            include_categories: Whether to include category features
            include_time: Whether to include time features
            include_activity: Whether to include activity features
            include_diversity: Whether to include diversity features
            include_homepage: Whether to include homepage behavior features.
                              These capture the distinction between homepage browsers
                              vs article readers, matching legacy clustering behavior.
            scale: Whether to scale features
            user_col: Name of user ID column
        """
        self.include_categories = include_categories
        self.include_time = include_time
        self.include_activity = include_activity
        self.include_diversity = include_diversity
        self.include_homepage = include_homepage
        self.scale = scale
        self.user_col = user_col
        
        self.metadata: Dict[str, Any] = {}
        self.is_fitted = False
    
    def fit_transform(
        self,
        impressions_df: pd.DataFrame,
        articles_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract features and fit scaler.
        
        Args:
            impressions_df: Impressions DataFrame (should include homepage views)
            articles_df: Optional articles DataFrame
            
        Returns:
            DataFrame with user features
        """
        features, self.metadata = create_user_features(
            impressions_df,
            articles_df,
            include_categories=self.include_categories,
            include_time=self.include_time,
            include_activity=self.include_activity,
            include_diversity=self.include_diversity,
            include_homepage=self.include_homepage,
            scale=self.scale,
            user_col=self.user_col,
        )
        
        self.is_fitted = True
        return features
    
    def get_feature_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Get feature matrix without user_id column.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Numpy array of features
        """
        feature_cols = [c for c in features_df.columns if c != self.user_col]
        return features_df[feature_cols].values
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.metadata.get('feature_columns', [])
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get extraction metadata.
        
        Returns:
            Metadata dictionary
        """
        return self.metadata.copy()
