"""
Feature engineering for user clustering.

Extracts user behavior features from impressions data for clustering.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    
    # Fill missing categories without copying the full frame.
    categories_series = df[category_col].fillna('unknown')
    
    # Get all unique categories
    categories = categories_series.unique()
    logger.info(f"Found {len(categories)} unique categories")
    
    # Count impressions per user per category
    category_counts = (
        df.assign(_category=categories_series)
        .groupby([user_col, '_category'])
        .size()
        .unstack(fill_value=0)
    )
    
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
    
    # Convert timestamp to datetime
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        dt = pd.to_datetime(df[time_col], errors="coerce")
    elif pd.api.types.is_numeric_dtype(df[time_col]):
        # Handle nullable pandas integer dtype (e.g., Int64) and infer unit by magnitude.
        # AD/HLN impressions store Unix milliseconds as Int64; parsing those as ns would
        # collapse hours to ~00:xx and yield "all night" features.
        ts = pd.to_numeric(df[time_col], errors="coerce")
        max_val = ts.max(skipna=True)
        if pd.isna(max_val):
            dt = pd.to_datetime(df[time_col], errors="coerce")
        else:
            if max_val > 1e17:
                unit = "ns"
            elif max_val > 1e14:
                unit = "us"
            elif max_val > 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(ts, unit=unit, errors="coerce")
    else:
        dt = pd.to_datetime(df[time_col], errors="coerce")
    
    # Extract hour
    hour = dt.dt.hour
    
    # Create time period columns
    time_df = pd.DataFrame({
        user_col: df[user_col].values,
        'is_morning': (hour >= 6) & (hour < 12),
        'is_afternoon': (hour >= 12) & (hour < 18),
        'is_evening': (hour >= 18) & (hour < 24),
        'is_night': (hour >= 0) & (hour < 6),
        'is_weekend': dt.dt.dayofweek >= 5,
    })
    
    # Aggregate per user
    time_features = time_df.groupby(user_col).agg({
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
    
    # Support both normalized null homepage IDs and legacy string values.
    article_ids = df[article_col]
    # Vectorized implementation for performance (~100x faster than row-by-row)
    is_homepage = article_ids.isna() | article_ids.eq('homepage')
    
    # Total impressions per user
    total_counts = df.groupby(user_col).size()
    
    # Homepage impressions per user
    homepage_counts = df[is_homepage].groupby(user_col).size().reindex(total_counts.index, fill_value=0)
    
    # Article impressions = total - homepage
    article_counts = total_counts - homepage_counts
    
    # Homepage ratio
    homepage_ratio = homepage_counts / total_counts
    
    # Build result DataFrame
    result = pd.DataFrame({
        user_col: total_counts.index,
        'homepage_impressions': homepage_counts.values,
        'article_impressions': article_counts.values,
        'homepage_ratio': homepage_ratio.values,
    })
    
    # Reading time features (if available)
    if read_time_col in df.columns:
        # Average reading time on homepage per user
        avg_reading_time_homepage = (
            df[is_homepage]
            .groupby(user_col)[read_time_col]
            .mean()
            .reindex(total_counts.index, fill_value=0)
        )
        
        # Average reading time on articles per user
        avg_reading_time_articles = (
            df[~is_homepage]
            .groupby(user_col)[read_time_col]
            .mean()
            .reindex(total_counts.index, fill_value=0)
        )
        
        # Total reading time per user (for proportion calculation)
        total_reading_time = (
            df.groupby(user_col)[read_time_col]
            .sum()
            .reindex(total_counts.index, fill_value=0)
        )
        
        # Article reading time per user
        article_reading_time = (
            df[~is_homepage]
            .groupby(user_col)[read_time_col]
            .sum()
            .reindex(total_counts.index, fill_value=0)
        )
        
        # Proportion of time on articles (handle division by zero)
        proportion_article_time = (article_reading_time / total_reading_time).fillna(0)
        
        result['avg_reading_time_homepage'] = avg_reading_time_homepage.values
        result['avg_reading_time_articles'] = avg_reading_time_articles.values
        result['proportion_article_time'] = proportion_article_time.values
    else:
        result['avg_reading_time_homepage'] = 0
        result['avg_reading_time_articles'] = 0
        result['proportion_article_time'] = 0
    
    # Fill NaN values
    result = result.fillna(0)
    
    logger.info(f"Created {len(result.columns) - 1} homepage features for {len(result)} users")
    
    return result


def create_diversity_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    category_col: str = 'category_str',
    article_col: str = 'article_id',
    legacy_mode: bool = False,
) -> pd.DataFrame:
    """Create content diversity features per user.
    
    Computes entropy and diversity metrics of user preferences.
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        category_col: Name of category column
        article_col: Name of article ID column
        legacy_mode: If True, only return num_categories (skip entropy/gini)
                     to match the legacy clustering feature set
        
    Returns:
        DataFrame with user_id and diversity columns
    """
    logger.info("Creating diversity features...")
    
    if legacy_mode:
        # Legacy mode: count only valid categories (exclude NaN/empty), matching legacy pipeline.
        all_users = df[user_col].drop_duplicates()
        valid_df = df[(df[category_col].notna()) & (df[category_col] != '')]
        num_categories = valid_df.groupby(user_col)[category_col].nunique().reindex(all_users, fill_value=0)
        result = pd.DataFrame({
            user_col: all_users.values,
            'num_categories': num_categories.values.astype(float),
        })
        logger.info(f"Created 1 diversity feature (legacy mode) for {len(result)} users")
        return result

    categories_series = df[category_col].fillna('unknown')
    
    # Vectorized implementation for performance (~5x faster than row-by-row)
    
    # Count impressions per user per category
    user_cat_counts = (
        df.assign(_category=categories_series)
        .groupby([user_col, '_category'])
        .size()
        .unstack(fill_value=0)
    )
    
    # Number of unique categories per user
    num_categories = (user_cat_counts > 0).sum(axis=1)
    
    # Full mode: include entropy and gini
    # Convert to proportions for entropy calculation
    user_cat_props = user_cat_counts.div(user_cat_counts.sum(axis=1), axis=0)
    
    # Entropy: -sum(p * log(p)) - handle zeros with where
    log_props = np.where(user_cat_props > 0, np.log(user_cat_props + 1e-10), 0)
    category_entropy = -(user_cat_props * log_props).sum(axis=1)
    
    # Gini coefficient per user (vectorized over all users).
    counts_arr = user_cat_counts.to_numpy(dtype=np.float64)
    n_categories = counts_arr.shape[1]
    sorted_counts = np.sort(counts_arr, axis=1)
    row_sums = sorted_counts.sum(axis=1)
    idx = np.arange(1, n_categories + 1, dtype=np.float64)
    numer = ((2.0 * idx - n_categories - 1.0) * sorted_counts).sum(axis=1)
    denom = n_categories * row_sums + 1e-10
    category_gini = np.where(row_sums > 0, numer / denom, 0.0)
    
    result = pd.DataFrame({
        user_col: user_cat_counts.index,
        'category_entropy': category_entropy.values,
        'num_categories': num_categories.values,
        'category_gini': category_gini,
    })
    
    logger.info(f"Created 3 diversity features for {len(result)} users")
    
    return result


def create_subscriber_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    subscriber_col: str = 'is_subscriber',
) -> pd.DataFrame:
    """Create subscriber status feature per user.
    
    Maps the per-impression subscriber flag to a single boolean per user.
    In the legacy pipeline this was a clustering feature.
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        subscriber_col: Name of subscriber column
        
    Returns:
        DataFrame with user_id and is_subscriber column (1/0)
    """
    logger.info("Creating subscriber features...")
    
    if subscriber_col not in df.columns:
        logger.info(f"Column '{subscriber_col}' not found — skipping subscriber feature")
        return None
    
    # A user is a subscriber if *any* of their impressions have is_subscriber=True
    subscriber_status = df.groupby(user_col)[subscriber_col].any().astype(int)
    
    result = subscriber_status.reset_index()
    result.columns = [user_col, 'is_subscriber']
    
    n_subscribers = result['is_subscriber'].sum()
    logger.info(f"Created subscriber feature for {len(result)} users ({n_subscribers} subscribers, {len(result) - n_subscribers} non-subscribers)")
    
    return result


def create_session_behavior_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    session_col: str = 'session_id',
    time_col: str = 'impression_time',
    category_col: str = 'category_str',
    read_time_col: str = 'read_time',
    article_col: str = 'article_id',
) -> pd.DataFrame:
    """Create legacy session behavior features per user.
    
    These features match the legacy clustering pipeline and capture
    session-level behavior patterns.
    
    Args:
        df: Impressions DataFrame
        user_col: Name of user ID column
        session_col: Name of session ID column
        time_col: Name of timestamp column
        category_col: Name of category column
        read_time_col: Name of read time column
        article_col: Name of article ID column
        
    Returns:
        DataFrame with user_id and session behavior columns:
        - avg_reading_time: overall average reading time per user
        - avg_session_length: average impressions per session (includes homepage views in legacy mode)
        - avg_categories_per_session: unique categories per session, averaged
        - avg_category_switches: category switches within sessions, averaged
        - avg_session_duration: average total read_time per session (sum of read_time across
          impressions in each session). Differs from legacy which used timestamp span
          (max - min impression_time); this definition includes single-impression sessions
          and avoids unit/timestamp issues.
    """
    logger.info("Creating session behavior features...")
    
    # Get unique users
    users = df[user_col].unique()
    
    # 1. avg_reading_time: overall average reading time per user
    if read_time_col in df.columns:
        avg_reading_time = df.groupby(user_col)[read_time_col].mean()
    else:
        avg_reading_time = pd.Series(0, index=users)
    
    # Initialize result with users
    result = pd.DataFrame({user_col: users})
    result = result.set_index(user_col)
    result['avg_reading_time'] = avg_reading_time.reindex(result.index, fill_value=0)
    
    # Session-based features (require session_col)
    if session_col in df.columns:
        # 2. avg_session_length: average impressions per session (legacy behavior includes homepage views)
        impressions_per_session = df.groupby([user_col, session_col]).size()
        avg_session_length = impressions_per_session.groupby(user_col).mean()
        result['avg_session_length'] = avg_session_length.reindex(result.index, fill_value=0)
        
        # 3. avg_categories_per_session: unique categories per session, averaged
        #    Legacy alignment: exclude sessions with 0 valid categories from the mean.
        if category_col in df.columns:
            # Filter for valid categories (non-empty, non-null)
            valid_cat_mask = (df[category_col].notna()) & (df[category_col] != '')
            valid_df = df[valid_cat_mask]
            
            if len(valid_df) > 0:
                cats_per_session = valid_df.groupby([user_col, session_col])[category_col].nunique()
                cats_per_session = cats_per_session[cats_per_session > 0]  # exclude 0-category sessions
                avg_cats_per_session = cats_per_session.groupby(level=0).mean()
                result['avg_categories_per_session'] = avg_cats_per_session.reindex(result.index, fill_value=0)
            else:
                result['avg_categories_per_session'] = 0
            
            # 4. avg_category_switches: category switches within sessions, averaged
            # Legacy alignment: include all sessions in valid_df (single-impression sessions
            # contribute 0 switches to the average).
            sorted_df = valid_df.sort_values([user_col, session_col, time_col])

            if len(sorted_df) > 0:
                session_keys = [user_col, session_col]
                shifted = sorted_df.groupby(session_keys)[category_col].shift(1)
                switches = ((sorted_df[category_col] != shifted) & shifted.notna()).astype(int)
                switches_per_session = switches.groupby(
                    [sorted_df[user_col], sorted_df[session_col]]
                ).sum()
                avg_switches = switches_per_session.groupby(level=0).mean()
                result['avg_category_switches'] = avg_switches.reindex(result.index, fill_value=0)
            else:
                result['avg_category_switches'] = 0
        else:
            result['avg_categories_per_session'] = 0
            result['avg_category_switches'] = 0
        
        # 5. avg_session_duration: average summed read_time per session
        if read_time_col in df.columns:
            session_read_time = (
                df.groupby([user_col, session_col])[read_time_col]
                .sum()
            )
            avg_session_duration = session_read_time.groupby(user_col).mean()
            result['avg_session_duration'] = avg_session_duration.reindex(result.index, fill_value=0)
        else:
            result['avg_session_duration'] = 0
    else:
        # No session column - fill with zeros
        result['avg_session_length'] = 0
        result['avg_categories_per_session'] = 0
        result['avg_category_switches'] = 0
        result['avg_session_duration'] = 0
    
    result = result.reset_index()
    result = result.fillna(0)
    
    logger.info(f"Created 5 session behavior features for {len(result)} users")
    
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
    include_session_behavior: bool = False,
    include_subscriber: bool = False,
    legacy_mode: bool = False,
    scale: bool = True,
    user_col: str = 'user_id',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create all user features for clustering.
    
    Main entry point for feature engineering.
    
    Args:
        impressions_df: Impressions DataFrame (should include homepage views for homepage features)
        articles_df: Optional articles DataFrame (for category info)
        include_categories: Whether to include category features (per-category proportions)
        include_time: Whether to include time features (time-of-day preferences)
        include_activity: Whether to include activity features
        include_diversity: Whether to include diversity features (entropy, gini)
        include_homepage: Whether to include homepage behavior features.
                          These capture the distinction between homepage browsers
                          vs article readers, matching the legacy clustering behavior.
        include_session_behavior: Whether to include session behavior features
                                  (avg_reading_time, avg_session_length, avg_categories_per_session,
                                  avg_category_switches, avg_session_duration)
        include_subscriber: Whether to include subscriber status feature
        legacy_mode: If True, automatically configure for legacy feature set:
                     - Disable category proportions (include_categories=False)
                     - Disable time features (include_time=False)
                     - Enable session behavior (include_session_behavior=True)
                     - Keep subscriber as post-hoc reporting only (include_subscriber=False)
                     - Use legacy diversity mode (only num_categories)
        scale: Whether to scale features
        user_col: Name of user ID column
        
    Returns:
        Tuple of (features DataFrame, metadata dict)
    """
    # Apply legacy mode overrides
    if legacy_mode:
        logger.info("Legacy mode enabled - using legacy feature set")
        include_categories = False
        include_time = False
        include_session_behavior = True
        include_subscriber = False  # Subscriber is post-hoc reporting only, not a clustering feature
        # legacy_mode will also be passed to diversity features
    
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
    
    metadata = {'feature_groups': {}, 'legacy_mode': legacy_mode}
    
    # Category features (per-category proportions - NOT in legacy)
    if include_categories and 'category_str' in df.columns:
        cat_features = create_category_features(df, user_col)
        features = features.merge(cat_features, on=user_col, how='left')
        metadata['feature_groups']['category'] = [c for c in cat_features.columns if c != user_col]
    
    # Time features (time-of-day preferences - NOT in legacy)
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
    
    # Session behavior features (legacy features)
    if include_session_behavior:
        session_features = create_session_behavior_features(df, user_col)
        features = features.merge(session_features, on=user_col, how='left')
        metadata['feature_groups']['session_behavior'] = [c for c in session_features.columns if c != user_col]
    
    # Subscriber feature (legacy clustering signal)
    if include_subscriber:
        subscriber_features = create_subscriber_features(df, user_col)
        if subscriber_features is not None:
            features = features.merge(subscriber_features, on=user_col, how='left')
            metadata['feature_groups']['subscriber'] = [c for c in subscriber_features.columns if c != user_col]
    
    # Diversity features
    if include_diversity and 'category_str' in df.columns:
        diversity_features = create_diversity_features(df, user_col, legacy_mode=legacy_mode)
        features = features.merge(diversity_features, on=user_col, how='left')
        metadata['feature_groups']['diversity'] = [c for c in diversity_features.columns if c != user_col]
    
    if legacy_mode:
        # Keep exact legacy clustering feature set.
        legacy_feature_cols = [
            'num_sessions',
            'total_impressions',
            'homepage_impressions',
            'article_impressions',
            'num_categories',
            'unique_articles',
            'avg_reading_time',
            'proportion_article_time',
            'avg_session_length',
            'avg_categories_per_session',
            'avg_category_switches',
            'avg_session_duration',
        ]
        existing_legacy_cols = [c for c in legacy_feature_cols if c in features.columns]
        features = features[[user_col] + existing_legacy_cols].copy()

    # Legacy-style targeted zero fill before mean imputation.
    for col in ('avg_categories_per_session', 'avg_category_switches', 'proportion_article_time'):
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Mean imputation fallback for any remaining NaNs.
    feature_cols = [c for c in features.columns if c != user_col]
    if feature_cols and features[feature_cols].isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        features[feature_cols] = imputer.fit_transform(features[feature_cols])

    # Final fallback.
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
        include_session_behavior: bool = False,
        include_subscriber: bool = False,
        legacy_mode: bool = False,
        scale: bool = True,
        user_col: str = 'user_id',
    ):
        """Initialize the feature extractor.
        
        Args:
            include_categories: Whether to include category features (per-category proportions)
            include_time: Whether to include time features (time-of-day preferences)
            include_activity: Whether to include activity features
            include_diversity: Whether to include diversity features (entropy, gini)
            include_homepage: Whether to include homepage behavior features.
                              These capture the distinction between homepage browsers
                              vs article readers, matching legacy clustering behavior.
            include_session_behavior: Whether to include session behavior features
                                      (avg_reading_time, avg_session_length, etc.)
            include_subscriber: Whether to include subscriber status feature
            legacy_mode: If True, automatically configure for legacy feature set:
                         - Disable category proportions
                         - Disable time features
                         - Enable session behavior
                         - Keep subscriber as post-hoc reporting only
                         - Use legacy diversity mode (only num_categories)
            scale: Whether to scale features
            user_col: Name of user ID column
        """
        self.include_categories = include_categories
        self.include_time = include_time
        self.include_activity = include_activity
        self.include_diversity = include_diversity
        self.include_homepage = include_homepage
        self.include_session_behavior = include_session_behavior
        self.include_subscriber = include_subscriber
        self.legacy_mode = legacy_mode
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
            include_session_behavior=self.include_session_behavior,
            include_subscriber=self.include_subscriber,
            legacy_mode=self.legacy_mode,
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
