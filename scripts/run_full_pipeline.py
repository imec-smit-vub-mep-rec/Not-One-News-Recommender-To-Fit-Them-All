#!/usr/bin/env python3
"""
Full pipeline script for RICON analysis.

This script runs the complete pipeline:
1. Convert raw data to standard format
2. Preprocess and clean data
3. Cluster users
4. Evaluate recommendation algorithms per cluster

Usage:
    python run_full_pipeline.py --config config.json
    python run_full_pipeline.py --dataset adressa --input-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import ast
import json
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, load_config, save_config, PRESET_CONFIGS
from src.utils import Session, setup_logging, get_logger, load_dataframe, save_dataframe, log_memory
from src.utils.datetime import parse_timestamp_series
from src.converters import ADConverter, AdressaConverter, EBNeRDConverter, GenericConverter
from src.preprocessing import DataCleaner, DataValidator, behaviors_to_interactions, articles_to_content
from src.clustering import UserFeatureExtractor, KMeansClusterer, ClusterVisualizer
from src.clustering.clustering import get_cluster_statistics


logger = get_logger("pipeline")


def _parse_embedding(raw_value):
    """Parse embedding values from article metadata into list[float]."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    # JSON-style arrays
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            return [float(x) for x in parsed]
    except Exception:
        pass

    # Python literal arrays/tuples
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and parsed:
            return [float(x) for x in parsed]
    except Exception:
        pass

    # Space-separated values in bracketed strings
    cleaned = text.strip("[]")
    values = []
    for token in cleaned.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values if values else None


def _save_session_embeddings_if_available(articles_df, config: PipelineConfig, session: Session):
    """
    Save session-local title_category_embeddings.parquet if AD raw embeddings exist.

    This keeps evaluation logic local-file based and avoids direct S3 Path.exists checks.
    """
    if config.dataset.name not in ("ad", "hln", "vk"):
        return
    if "bert_embedding" not in articles_df.columns:
        logger.info("AD dataset detected but no 'bert_embedding' column found in articles")
        return

    parsed = articles_df["bert_embedding"].map(_parse_embedding)
    valid_mask = parsed.notna()
    if not valid_mask.any():
        logger.warning("No valid embeddings parsed from 'bert_embedding' column")
        return

    embeddings_df = articles_df.loc[valid_mask, ["article_id"]].copy()
    embeddings_df["article_id"] = embeddings_df["article_id"].astype(str)
    embeddings_df["embedding"] = parsed.loc[valid_mask]

    # Keep only rows with consistent embedding length
    lengths = embeddings_df["embedding"].map(len)
    target_dim = int(lengths.mode().iloc[0])
    consistent_mask = lengths.eq(target_dim)
    dropped = int((~consistent_mask).sum())
    if dropped:
        logger.warning(
            f"Dropping {dropped} embedding rows with non-standard dimension (target={target_dim})"
        )
        embeddings_df = embeddings_df.loc[consistent_mask].copy()

    embedding_path = session.get_path("title_category_embeddings.parquet")
    save_dataframe(embeddings_df, embedding_path)
    logger.info(
        f"Saved {len(embeddings_df)} session embeddings to {embedding_path} "
        f"(dimension={target_dim})"
    )


def _derive_subscriber_from_paywall_metered(
    impressions_df,
    articles_df,
    min_paywall_reads: int = 2,
    min_read_time_seconds: float = 30.0,
):
    """Derive user-level subscriber flags from metered paywall behavior.

    A user is marked subscriber when they have at least ``min_paywall_reads``
    impressions on paywalled articles with ``read_time`` strictly greater than
    ``min_read_time_seconds``.
    """
    import pandas as pd

    if articles_df is None or impressions_df is None:
        return impressions_df
    if 'article_id' not in impressions_df.columns or 'user_id' not in impressions_df.columns:
        return impressions_df
    if 'article_id' not in articles_df.columns or 'is_paywall' not in articles_df.columns:
        return impressions_df

    df = impressions_df.copy()
    lookup = articles_df[['article_id', 'is_paywall']].drop_duplicates(subset='article_id').copy()
    lookup['is_paywall'] = lookup['is_paywall'].fillna(False).astype(bool)

    merged = df[['user_id', 'article_id', 'read_time']].merge(
        lookup, on='article_id', how='left'
    )
    read_time = pd.to_numeric(merged['read_time'], errors='coerce').fillna(0.0)
    qualifying = merged['is_paywall'].fillna(False) & (read_time > float(min_read_time_seconds))

    qualifying_counts = merged.loc[qualifying].groupby('user_id').size()
    derived = qualifying_counts >= int(min_paywall_reads)
    derived = derived.astype(bool)

    user_ids = df['user_id'].dropna().astype(str).unique()
    derived = derived.reindex(user_ids, fill_value=False)

    if 'is_subscriber' in df.columns:
        existing = df.groupby('user_id')['is_subscriber'].any().reindex(user_ids, fill_value=False).astype(bool)
        user_subscriber = existing | derived
    else:
        user_subscriber = derived

    df['is_subscriber'] = df['user_id'].map(user_subscriber).fillna(False).astype(bool)
    logger.info(
        "Derived subscriber status from metered paywall: "
        f"{int(user_subscriber.sum())}/{len(user_subscriber)} users"
    )
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full RICON analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ad", "adressa", "ebnerd", "hln", "vk", "custom"],
        help="Dataset type (uses preset config)",
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory with raw data",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: runs/)",
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID / session directory name (e.g. hln_20260216_130931). Use when --skip-clustering to load clusters from an existing run.",
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Number of clusters (default: auto-detect)",
    )
    
    parser.add_argument(
        "--legacy-features",
        action="store_true",
        help="Use legacy feature set for clustering (no per-category proportions, no time-of-day)",
    )
    
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip data conversion (use existing converted data)",
    )
    
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering (use existing cluster assignments)",
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip RecPack evaluation",
    )
    
    parser.add_argument(
        "--train-per-cluster",
        action="store_true",
        help="Train a separate model per cluster (default: train on full dataset, aggregate per cluster)",
    )
    
    parser.add_argument(
        "--content-mode",
        type=str,
        choices=["legacy", "full", "embeddings"],
        help="Content mode for CB-ST: legacy (category+title), full (category+title+body), or embeddings (pre-calculated only)",
    )

    parser.add_argument(
        "--full-content",
        action="store_true",
        help="DEPRECATED: equivalent to --content-mode full",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def load_or_create_config(args) -> PipelineConfig:
    """Load config from file or create from arguments.
    
    Supports three modes:
    1. --config alone: Load full config from JSON file
    2. --dataset alone: Use preset config for known datasets
    3. --dataset + --config: Use preset for dataset, merge clustering/evaluation from config file
    """
    from dataclasses import replace
    import json
    
    config = None
    config_overrides = {}
    
    # Load config file overrides if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_overrides = json.load(f)
        logger.info(f"Loaded config overrides from {args.config}")
    
    # Create base config from dataset preset or full config file
    if args.dataset:
        if args.dataset in PRESET_CONFIGS:
            # PRESET_CONFIGS contains DatasetConfig objects, not full PipelineConfig
            # Use replace() to create a copy - never mutate the shared preset
            preset = PRESET_CONFIGS[args.dataset]
            if args.input_dir:
                preset = replace(preset, input_path=args.input_dir)
            config = PipelineConfig(dataset=preset)
            logger.info(f"Using preset config for {args.dataset}")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    elif args.config and 'dataset' in config_overrides:
        # Full config from file
        config = load_config(args.config)
        config_overrides = {}  # Already loaded
    else:
        # Create default config
        from src.config import DatasetConfig, SessionConfig
        config = PipelineConfig(
            dataset=DatasetConfig(name="custom", input_path=args.input_dir or ""),
        )
    
    # Apply config file overrides (for --dataset + --config case)
    if config_overrides:
        if 'dataset' in config_overrides:
            for key, value in config_overrides['dataset'].items():
                # Accept legacy alias in config files.
                target_key = "input_path" if key == "input_dir" else key
                if hasattr(config.dataset, target_key):
                    setattr(config.dataset, target_key, value)
                    logger.info(f"Config override: dataset.{target_key} = {value}")
            # Keep session run_id in sync when dataset name is overridden
            if 'name' in config_overrides['dataset']:
                config.session.dataset_name = config.dataset.name
                config.session.run_id = f"{config.dataset.name}_{config.session.timestamp}"
        if 'clustering' in config_overrides:
            for key, value in config_overrides['clustering'].items():
                if hasattr(config.clustering, key):
                    setattr(config.clustering, key, value)
                    logger.info(f"Config override: clustering.{key} = {value}")
        if 'evaluation' in config_overrides:
            for key, value in config_overrides['evaluation'].items():
                if hasattr(config.evaluation, key):
                    setattr(config.evaluation, key, value)
                    logger.info(f"Config override: evaluation.{key} = {value}")
        if 'session' in config_overrides:
            for key, value in config_overrides['session'].items():
                if hasattr(config.session, key):
                    setattr(config.session, key, value)
                    logger.info(f"Config override: session.{key} = {value}")
    
    # Override with command line args (highest priority)
    if args.input_dir:
        config.dataset.input_path = args.input_dir
    
    if args.output_dir:
        config.session.base_output_dir = args.output_dir
    
    if args.run_id:
        config.session.run_id = args.run_id
        logger.info(f"CLI override: session.run_id = {args.run_id}")
    
    if args.n_clusters:
        config.clustering.n_clusters = args.n_clusters
    
    # --legacy-features flag takes highest priority
    if args.legacy_features:
        config.clustering.legacy_features = True
        logger.info("CLI override: clustering.legacy_features = True")

    if args.content_mode:
        config.evaluation.content_mode = args.content_mode
        logger.info(f"CLI override: evaluation.content_mode = {args.content_mode}")

    if args.train_per_cluster:
        config.evaluation.train_on_full_dataset = False
        logger.info("CLI override: evaluation.train_on_full_dataset = False (train per cluster)")
    
    return config


def run_conversion(config: PipelineConfig, session: Session) -> tuple:
    """Run data conversion step.
    
    Returns:
        Tuple of (articles_df, impressions_df)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Data Conversion")
    logger.info("=" * 60)
    
    dataset_format = config.dataset.format
    input_path = config.dataset.input_path
    
    # Select converter
    # ad, hln, vk share the same S3 Spark CSV structure (article_metadata.csv + impressions/)
    if config.dataset.name in ("ad", "hln", "vk") or dataset_format == "spark_csv":
        converter = ADConverter(config=config.dataset)
    elif dataset_format == "jsonl" or config.dataset.name == "adressa":
        converter = AdressaConverter(config=config.dataset)
    elif dataset_format == "parquet" or config.dataset.name == "ebnerd":
        converter = EBNeRDConverter(config=config.dataset)
    else:
        converter = GenericConverter(
            config=config.dataset,
        )
    
    if dataset_format == "jsonl" or config.dataset.name == "adressa":
        # Adressa articles are extracted during impression processing.
        logger.info("Converting impressions (required before articles for Adressa)...")
        impressions_df = converter.convert_impressions()
        
        logger.info("Converting articles...")
        articles_df = converter.convert_articles()
    else:
        # Convert articles
        logger.info("Converting articles...")
        articles_df = converter.convert_articles()
        
        # Convert impressions
        logger.info("Converting impressions...")
        impressions_df = converter.convert_impressions()

    # Keep logged-in and subscriber concepts disentangled:
    # - is_logged_in comes from impression auth state
    # - is_subscriber is derived from metered paywall behavior when available
    if config.dataset.name in ("ad", "hln", "vk"):
        min_paywall_reads = 1 if config.dataset.name == "vk" else 2
        impressions_df = _derive_subscriber_from_paywall_metered(
            impressions_df,
            articles_df,
            min_paywall_reads=min_paywall_reads,
            min_read_time_seconds=30.0,
        )
    
    articles_path = session.get_path("articles.parquet")
    save_dataframe(articles_df, articles_path)
    logger.info(f"Saved {len(articles_df)} articles to {articles_path}")
    
    impressions_path = session.get_path("impressions.parquet")
    save_dataframe(impressions_df, impressions_path)
    logger.info(f"Saved {len(impressions_df)} impressions to {impressions_path}")

    _save_session_embeddings_if_available(articles_df, config, session)
    
    return articles_df, impressions_df


def run_preprocessing(
    articles_df,
    impressions_df,
    config: PipelineConfig,
    session: Session,
    content_mode: str = "legacy",
) -> tuple:
    """Run preprocessing step for CLUSTERING.
    
    IMPORTANT: This preprocessing matches the legacy behavior:
    1. NO user filtering - clustering happens on ALL users
    2. NO removal of homepage views - homepage behavior is a clustering signal!
       Users who only visit homepage are a distinct behavioral cluster.
    3. interactions.csv (for RecPack) is created separately and only includes
       rows with valid article_id. RecPack's MinItemsPerUser filter is applied there.
    
    Returns:
        Tuple of (cleaned_articles, cleaned_impressions, interactions)
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing (for clustering)")
    logger.info("=" * 60)
    
    # Validate data
    validator = DataValidator()
    is_valid = validator.validate_all(
        articles=articles_df,
        impressions=impressions_df,
    )
    validator.print_summary()
    
    if not is_valid:
        logger.warning("Validation found issues, proceeding with cleaning")
    
    # Clean data for CLUSTERING - matching legacy behavior:
    # - NO user filtering (filter_users=False)
    # - NO removal of homepage views (remove_empty_articles=False)
    # Homepage behavior is a meaningful clustering signal!
    cleaner = DataCleaner(
        min_impressions_per_user=config.clustering.min_impressions_per_user,
        remove_empty_articles=False,  # CRITICAL: Keep homepage views for clustering
        clean_categories=True,
        filter_users=False,  # CRITICAL: Do NOT filter users before clustering
    )
    
    cleaned_articles = cleaner.clean_articles(articles_df)
    cleaned_impressions = cleaner.clean_impressions(impressions_df, cleaned_articles)
    
    logger.info(f"Cleaning stats: {cleaner.get_stats()}")
    logger.info(f"NOTE: All {cleaner.get_stats().get('final_users', 'N/A')} users retained for clustering (including homepage-only users)")
    
    # Save cleaned data (includes homepage views)
    save_dataframe(cleaned_articles, session.get_path("articles_cleaned.parquet"))
    save_dataframe(cleaned_impressions, session.get_path("impressions_cleaned.parquet"))
    
    # Create interactions for RecPack - this ONLY includes article interactions
    # (behaviors_to_interactions filters out rows without valid article_id)
    # User filtering (min_items_per_user) happens later in RecPack
    interactions_df = behaviors_to_interactions(cleaned_impressions)
    
    interactions_path = session.get_path("interactions.csv")
    save_dataframe(interactions_df, interactions_path, format="csv")
    logger.info(f"Saved {len(interactions_df)} article interactions to {interactions_path}")
    logger.info(f"NOTE: interactions.csv excludes homepage views (for RecPack evaluation)")
    
    # Create article content for content-based unless embeddings-only mode is selected.
    if content_mode == "embeddings":
        logger.info("Skipping content generation (EMBEDDINGS mode uses pre-calculated vectors)")
    else:
        full_content = content_mode == "full"
        content_df = articles_to_content(cleaned_articles, full_content=full_content)
        content_path = session.get_path("articles_content.csv")
        save_dataframe(content_df, content_path, format="csv")
        logger.info(f"Saved article content to {content_path}")
    
    return cleaned_articles, cleaned_impressions, interactions_df


def _compute_cluster_summary(
    impressions_df: 'pd.DataFrame',
    users_df: 'pd.DataFrame',
    subscriber_label: str = "Number of Subscribers",
    articles_df: 'pd.DataFrame | None' = None,
) -> 'pd.DataFrame':
    """Compute the legacy-style cluster summary from raw impressions.

    Matches the legacy ``create_cluster_summary`` output with interpretable
    average values per cluster.

    Args:
        impressions_df: Raw impressions DataFrame (with homepage views).
        users_df: DataFrame with ``user_id`` and ``cluster_id`` columns.
        articles_df: Optional articles DataFrame with ``article_id`` and
                     ``category_str`` columns.  When provided and
                     ``category_str`` is not already in *impressions_df*,
                     the category information is merged in so that
                     "Avg Categories Read" and category-switch metrics can
                     be computed.

    Returns:
        DataFrame indexed by cluster_id with one row per cluster.
    """
    import numpy as np
    import pandas as pd

    users_unique = users_df[['user_id', 'cluster_id']].drop_duplicates()
    user_cluster = users_unique.set_index('user_id')['cluster_id']

    cluster_by_impression = impressions_df['user_id'].map(user_cluster)
    valid_mask = cluster_by_impression.notna()
    if not valid_mask.any():
        return pd.DataFrame()

    needed_cols = [c for c in ['user_id', 'article_id', 'read_time', 'session_id', 'category_str', 'impression_time'] if c in impressions_df.columns]
    df = impressions_df.loc[valid_mask, needed_cols].copy()
    df['cluster_id'] = cluster_by_impression.loc[valid_mask].values

    # Merge category info from articles if not already present in impressions
    if 'category_str' not in df.columns and articles_df is not None and 'category_str' in articles_df.columns:
        cat_lookup = articles_df[['article_id', 'category_str']].drop_duplicates(subset='article_id')
        df = df.merge(cat_lookup, on='article_id', how='left')

    total_users = users_unique['user_id'].nunique()

    # --- Basic cluster sizes --------------------------------------------------------
    cluster_sizes = users_unique.groupby('cluster_id')['user_id'].nunique()
    cluster_pct = (cluster_sizes / total_users * 100).round(2)

    # --- Subscriber and logged-in counts (post-hoc, not clustering features) --------
    if 'is_subscriber' in impressions_df.columns:
        sub_per_user = (
            impressions_df.loc[valid_mask, ['user_id', 'is_subscriber']]
            .groupby('user_id')['is_subscriber']
            .any()
            .astype(int)
        )
        sub_per_user = sub_per_user.reindex(user_cluster.index, fill_value=0)
        sub_counts = sub_per_user.groupby(user_cluster).sum()
    else:
        sub_per_user = pd.Series(0, index=user_cluster.index, dtype=int)
        sub_counts = pd.Series(0, index=cluster_sizes.index)

    has_logged_in = 'is_logged_in' in impressions_df.columns
    if has_logged_in:
        logged_in_per_user = (
            impressions_df.loc[valid_mask, ['user_id', 'is_logged_in']]
            .groupby('user_id')['is_logged_in']
            .any()
            .astype(int)
        )
        logged_in_per_user = logged_in_per_user.reindex(user_cluster.index, fill_value=0)
        logged_in_counts = logged_in_per_user.groupby(user_cluster).sum()
    else:
        logged_in_per_user = pd.Series(0, index=user_cluster.index, dtype=int)
        logged_in_counts = pd.Series(0, index=cluster_sizes.index)

    # --- Per-user metrics (then average per cluster) --------------------------------
    is_homepage = df['article_id'].isna() | df['article_id'].eq('homepage')
    zero_user = pd.Series(0.0, index=user_cluster.index)

    if 'read_time' in df.columns:
        avg_reading_time = df.groupby('user_id')['read_time'].mean().reindex(user_cluster.index, fill_value=0)
        article_reading_time = df.loc[~is_homepage].groupby('user_id')['read_time'].sum().reindex(user_cluster.index, fill_value=0)
        total_reading_time = df.groupby('user_id')['read_time'].sum().reindex(user_cluster.index, fill_value=0)
        proportion_article_time = (article_reading_time / total_reading_time).replace([np.inf, -np.inf], 0).fillna(0)
        avg_rt_homepage = df.loc[is_homepage].groupby('user_id')['read_time'].mean().reindex(user_cluster.index, fill_value=0)
        avg_rt_articles = df.loc[~is_homepage].groupby('user_id')['read_time'].mean().reindex(user_cluster.index, fill_value=0)
    else:
        avg_reading_time = zero_user
        proportion_article_time = zero_user
        avg_rt_homepage = zero_user
        avg_rt_articles = zero_user

    has_session = 'session_id' in df.columns
    has_category = 'category_str' in df.columns

    if has_session:
        session_counts = (
            df.groupby('user_id')['session_id']
            .nunique()
            .reindex(user_cluster.index, fill_value=0)
            .astype(float)
        )
        impressions_per_session = df.groupby(['user_id', 'session_id']).size()
        avg_impressions_per_session = impressions_per_session.groupby('user_id').mean().reindex(user_cluster.index, fill_value=0)

        has_subscriber = 'is_subscriber' in impressions_df.columns
        if has_subscriber:
            subscriber_mask = sub_per_user.astype(bool)
            avg_sessions_subscriber = (
                session_counts[subscriber_mask]
                .groupby(user_cluster[subscriber_mask])
                .mean()
                .reindex(cluster_sizes.index)
            )
            avg_sessions_non_subscriber = (
                session_counts[~subscriber_mask]
                .groupby(user_cluster[~subscriber_mask])
                .mean()
                .reindex(cluster_sizes.index)
            )
        else:
            avg_sessions_subscriber = pd.Series(np.nan, index=cluster_sizes.index)
            avg_sessions_non_subscriber = pd.Series(np.nan, index=cluster_sizes.index)

        if has_logged_in:
            logged_in_mask = logged_in_per_user.astype(bool)
            avg_sessions_logged_in = (
                session_counts[logged_in_mask]
                .groupby(user_cluster[logged_in_mask])
                .mean()
                .reindex(cluster_sizes.index)
            )
            avg_sessions_non_logged_in = (
                session_counts[~logged_in_mask]
                .groupby(user_cluster[~logged_in_mask])
                .mean()
                .reindex(cluster_sizes.index)
            )
        else:
            avg_sessions_logged_in = pd.Series(np.nan, index=cluster_sizes.index)
            avg_sessions_non_logged_in = pd.Series(np.nan, index=cluster_sizes.index)
    else:
        session_counts = zero_user
        avg_impressions_per_session = zero_user
        avg_sessions_subscriber = pd.Series(np.nan, index=cluster_sizes.index)
        avg_sessions_non_subscriber = pd.Series(np.nan, index=cluster_sizes.index)
        avg_sessions_logged_in = pd.Series(np.nan, index=cluster_sizes.index)
        avg_sessions_non_logged_in = pd.Series(np.nan, index=cluster_sizes.index)

    if has_category:
        valid_cat = df['category_str'].notna() & (df['category_str'] != '')
        num_categories = (
            df.loc[valid_cat]
            .groupby('user_id')['category_str']
            .nunique()
            .reindex(user_cluster.index, fill_value=0)
        )
    else:
        num_categories = zero_user

    time_seconds = None
    if 'impression_time' in df.columns:
        # Robust parsing for Int64 (nullable) unix-ms, unix-as-string, and ISO8601 ("...Z").
        dt = parse_timestamp_series(df['impression_time'])
        time_seconds = dt.astype('int64') / 1e9
        hour = dt.dt.hour
        time_flags = pd.DataFrame({
            'user_id': df['user_id'].values,
            'morning': ((hour >= 6) & (hour < 12)).values,
            'afternoon': ((hour >= 12) & (hour < 18)).values,
            'evening': ((hour >= 18) & (hour < 24)).values,
            'night': ((hour >= 0) & (hour < 6)).values,
            'weekend': (dt.dt.dayofweek >= 5).values,
        })
        time_means = time_flags.groupby('user_id').mean()
        pct_morning = time_means['morning'].reindex(user_cluster.index, fill_value=0)
        pct_afternoon = time_means['afternoon'].reindex(user_cluster.index, fill_value=0)
        pct_evening = time_means['evening'].reindex(user_cluster.index, fill_value=0)
        pct_night = time_means['night'].reindex(user_cluster.index, fill_value=0)
        pct_weekend = time_means['weekend'].reindex(user_cluster.index, fill_value=0)

        _time_agg = dt.groupby(df['user_id']).agg(['min', 'max'])
        _span = (_time_agg['max'] - _time_agg['min']).dt.total_seconds() / (24 * 3600)
        engagement_span_days = _span.reindex(user_cluster.index, fill_value=0).fillna(0)
    else:
        pct_morning = pct_afternoon = pct_evening = pct_night = pct_weekend = zero_user
        engagement_span_days = zero_user

    if has_session and 'read_time' in df.columns:
        # Avg Session Duration: sum of read_time per session (differs from legacy timestamp
        # span max-min; includes single-impression sessions and avoids unit issues).
        session_read_time = (
            df.groupby(['user_id', 'session_id'])['read_time']
            .sum()
        )
        avg_session_duration = session_read_time.groupby('user_id').mean().reindex(user_cluster.index, fill_value=0)
    else:
        avg_session_duration = zero_user

    if has_session and has_category:
        valid_cat_df = df.loc[df['category_str'].notna() & (df['category_str'] != ''), ['user_id', 'session_id', 'category_str']].copy()
        if not valid_cat_df.empty:
            if 'impression_time' in df.columns:
                valid_cat_df['_ts'] = time_seconds.reindex(valid_cat_df.index).values
                valid_cat_df = valid_cat_df.sort_values(['user_id', 'session_id', '_ts'])
            session_keys = ['user_id', 'session_id']
            shifted = valid_cat_df.groupby(session_keys)['category_str'].shift(1)
            switches = ((valid_cat_df['category_str'] != shifted) & shifted.notna()).astype(int)
            switches_per_session = switches.groupby([valid_cat_df['user_id'], valid_cat_df['session_id']]).sum()
            avg_cat_switches = switches_per_session.groupby(level=0).mean().reindex(user_cluster.index, fill_value=0)
        else:
            avg_cat_switches = zero_user
    else:
        avg_cat_switches = zero_user

    # --- Device breakdown (if column present) ----------------------------------------
    if 'device_type' in df.columns:
        _dev = df[['user_id', 'device_type']].copy()
        _dev['device_type'] = _dev['device_type'].fillna('unknown').str.lower()
        _dev_dummies = pd.get_dummies(_dev, columns=['device_type'], prefix='', prefix_sep='')
        _dev_cols = [c for c in _dev_dummies.columns if c != 'user_id']
        _dev_means = _dev_dummies.groupby('user_id')[_dev_cols].mean()
        device_desktop = _dev_means.get('desktop', pd.Series(0.0, index=_dev_means.index)).reindex(user_cluster.index, fill_value=0)
        device_mobile = _dev_means.get('mobile', pd.Series(0.0, index=_dev_means.index)).reindex(user_cluster.index, fill_value=0)
        device_tablet = _dev_means.get('tablet', pd.Series(0.0, index=_dev_means.index)).reindex(user_cluster.index, fill_value=0)
    else:
        device_desktop = device_mobile = device_tablet = None

    # --- Scroll depth (if column present) -------------------------------------------
    if 'scroll_depth' in df.columns:
        avg_scroll_depth = df.groupby('user_id')['scroll_depth'].mean().reindex(user_cluster.index, fill_value=0)
    else:
        avg_scroll_depth = None

    # --- Homepage / article impression counts per user ------------------------------
    hp_count_per_user = df[is_homepage].groupby('user_id').size().reindex(user_cluster.index, fill_value=0).astype(float)
    art_count_per_user = df[~is_homepage].groupby('user_id').size().reindex(user_cluster.index, fill_value=0).astype(float)
    _total_per_user = (hp_count_per_user + art_count_per_user).replace(0, np.nan)
    homepage_ratio_user = (hp_count_per_user / _total_per_user).fillna(0)

    # --- Diversity metrics (entropy & gini from raw category data) ------------------
    if has_category:
        _valid_cat = df['category_str'].notna() & (df['category_str'] != '')
        _vdf = df.loc[_valid_cat, ['user_id', 'category_str']]
        if not _vdf.empty:
            _ucc = _vdf.groupby(['user_id', 'category_str']).size().unstack(fill_value=0)
            _ucp = _ucc.div(_ucc.sum(axis=1), axis=0)
            _logp = np.where(_ucp.values > 0, np.log(_ucp.values + 1e-10), 0)
            category_entropy = pd.Series(
                -(_ucp.values * _logp).sum(axis=1), index=_ucc.index,
            ).reindex(user_cluster.index, fill_value=0)
            _arr = _ucc.reindex(user_cluster.index, fill_value=0).to_numpy(dtype=np.float64)
            _nc = _arr.shape[1]
            _sorted = np.sort(_arr, axis=1)
            _rs = _sorted.sum(axis=1)
            _idx_arr = np.arange(1, _nc + 1, dtype=np.float64)
            _num = ((2.0 * _idx_arr - _nc - 1.0) * _sorted).sum(axis=1)
            _den = _nc * _rs + 1e-10
            category_gini = pd.Series(
                np.where(_rs > 0, _num / _den, 0.0), index=user_cluster.index,
            )
        else:
            category_entropy = zero_user
            category_gini = zero_user
    else:
        category_entropy = zero_user
        category_gini = zero_user

    # --- Build per-user table, then average per cluster -----------------------------
    user_metrics = pd.DataFrame(index=user_cluster.index)
    user_metrics['cluster_id'] = user_cluster.values
    user_metrics['avg_reading_time'] = avg_reading_time
    user_metrics['proportion_article_time'] = proportion_article_time
    user_metrics['avg_reading_time_homepage'] = avg_rt_homepage
    user_metrics['avg_reading_time_articles'] = avg_rt_articles
    user_metrics['avg_impressions_per_session'] = avg_impressions_per_session
    user_metrics['avg_sessions_per_user'] = session_counts
    user_metrics['avg_categories_read'] = num_categories
    user_metrics['avg_session_duration'] = avg_session_duration
    user_metrics['avg_category_switches'] = avg_cat_switches
    user_metrics['pct_morning'] = pct_morning
    user_metrics['pct_afternoon'] = pct_afternoon
    user_metrics['pct_evening'] = pct_evening
    user_metrics['pct_night'] = pct_night
    user_metrics['pct_weekend'] = pct_weekend
    user_metrics['engagement_span_days'] = engagement_span_days
    user_metrics['avg_homepage_impressions'] = hp_count_per_user
    user_metrics['avg_article_impressions'] = art_count_per_user
    user_metrics['homepage_ratio'] = homepage_ratio_user
    user_metrics['category_entropy'] = category_entropy
    user_metrics['category_gini'] = category_gini
    if device_desktop is not None:
        user_metrics['device_desktop'] = device_desktop
        user_metrics['device_mobile'] = device_mobile
        user_metrics['device_tablet'] = device_tablet
    if avg_scroll_depth is not None:
        user_metrics['avg_scroll_depth'] = avg_scroll_depth
    user_metrics = user_metrics.fillna(0)

    cluster_means = user_metrics.groupby('cluster_id').mean().round(4)

    sub_proportion = (sub_counts / cluster_sizes.replace(0, np.nan)).fillna(0).round(4)
    logged_in_proportion = (logged_in_counts / cluster_sizes.replace(0, np.nan)).fillna(0).round(4)

    cols = {
        'Number of Users': cluster_sizes,
        'Percentage of Users (%)': cluster_pct,
        subscriber_label: sub_counts,
        'Proportion of Subscribers': sub_proportion,
    }
    if has_logged_in:
        cols['Number of Logged-in Users'] = logged_in_counts
        cols['Proportion of Logged-in Users'] = logged_in_proportion

    cols.update({
        'Avg Reading Time (s)': cluster_means['avg_reading_time'],
        'Proportion of Time on Articles': cluster_means['proportion_article_time'],
        'Avg Reading Time Homepage (s)': cluster_means['avg_reading_time_homepage'],
        'Avg Reading Time Articles (s)': cluster_means['avg_reading_time_articles'],
        'Avg Impressions per Session': cluster_means['avg_impressions_per_session'],
        'Avg Sessions per User': cluster_means['avg_sessions_per_user'],
        'Avg Sessions per Subscriber': avg_sessions_subscriber.round(4),
        'Avg Sessions per Non-subscriber': avg_sessions_non_subscriber.round(4),
    })
    if has_logged_in:
        cols['Avg Sessions per Logged-in User'] = avg_sessions_logged_in.round(4)
        cols['Avg Sessions per Non-logged-in User'] = avg_sessions_non_logged_in.round(4)

    cols.update({
        'Avg Categories Read': cluster_means['avg_categories_read'],
        'Avg Session Duration (s)': cluster_means['avg_session_duration'],
        'Avg Category Switches per Session': cluster_means['avg_category_switches'],
        'Morning (%)': (cluster_means['pct_morning'] * 100).round(2),
        'Afternoon (%)': (cluster_means['pct_afternoon'] * 100).round(2),
        'Evening (%)': (cluster_means['pct_evening'] * 100).round(2),
        'Night (%)': (cluster_means['pct_night'] * 100).round(2),
        'Weekend (%)': (cluster_means['pct_weekend'] * 100).round(2),
        'Avg Engagement Span (days)': cluster_means['engagement_span_days'].round(2),
        'Avg Homepage Impressions': cluster_means['avg_homepage_impressions'].round(2),
        'Avg Article Impressions': cluster_means['avg_article_impressions'].round(2),
        'Homepage Ratio': cluster_means['homepage_ratio'].round(4),
        'Avg Category Entropy': cluster_means['category_entropy'].round(4),
        'Avg Category Gini': cluster_means['category_gini'].round(4),
    })

    summary = pd.DataFrame(cols)

    if 'device_desktop' in cluster_means.columns:
        summary['Desktop (%)'] = (cluster_means['device_desktop'] * 100).round(2)
        summary['Mobile (%)'] = (cluster_means['device_mobile'] * 100).round(2)
        summary['Tablet (%)'] = (cluster_means['device_tablet'] * 100).round(2)
    if 'avg_scroll_depth' in cluster_means.columns:
        summary['Avg Scroll Depth'] = cluster_means['avg_scroll_depth'].round(4)

    return summary, user_metrics


def _compute_category_profiles(
    impressions_df: 'pd.DataFrame',
    users_df: 'pd.DataFrame',
    articles_df: 'pd.DataFrame | None' = None,
    top_n: int = 10,
) -> 'pd.DataFrame | None':
    """Compute top-N category preferences per cluster from raw impressions.

    Returns a DataFrame with columns: cluster_id, category, impressions,
    proportion_pct, rank.  Returns ``None`` when category data is unavailable.
    """
    import pandas as pd

    user_cluster = users_df.drop_duplicates().set_index('user_id')['cluster_id']

    needed = ['user_id', 'article_id']
    if 'category_str' in impressions_df.columns:
        needed.append('category_str')
    df = impressions_df[impressions_df['user_id'].isin(user_cluster.index)][needed].copy()
    df['cluster_id'] = df['user_id'].map(user_cluster)

    if 'category_str' not in df.columns and articles_df is not None and 'category_str' in articles_df.columns:
        cat_lookup = articles_df[['article_id', 'category_str']].drop_duplicates(subset='article_id')
        df = df.merge(cat_lookup, on='article_id', how='left')

    if 'category_str' not in df.columns:
        return None

    valid = df[df['category_str'].notna() & (df['category_str'] != '')]
    if valid.empty:
        return None

    counts = valid.groupby(['cluster_id', 'category_str']).size().reset_index(name='impressions')
    totals = counts.groupby('cluster_id')['impressions'].transform('sum')
    counts['proportion_pct'] = (counts['impressions'] / totals * 100).round(2)
    counts['rank'] = (
        counts.groupby('cluster_id')['impressions']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    counts = counts.sort_values(['cluster_id', 'rank'])
    counts = counts.rename(columns={'category_str': 'category'})
    return counts[counts['rank'] <= top_n]


def _compute_cluster_distributions(
    user_metrics: 'pd.DataFrame',
) -> 'pd.DataFrame':
    """Compute percentile distributions per cluster for key user metrics.

    Returns a DataFrame with columns: cluster_id, metric, mean, std, min,
    p25, median, p75, max.
    """
    import pandas as pd

    key_metrics = [
        'avg_reading_time', 'avg_sessions_per_user', 'avg_impressions_per_session',
        'avg_session_duration', 'avg_categories_read', 'avg_category_switches',
        'engagement_span_days', 'category_entropy', 'category_gini',
        'avg_homepage_impressions', 'avg_article_impressions', 'homepage_ratio',
    ]
    available = [m for m in key_metrics if m in user_metrics.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    for cluster_id, group in user_metrics.groupby('cluster_id'):
        for metric in available:
            vals = group[metric]
            rows.append({
                'cluster_id': cluster_id,
                'metric': metric,
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'min': round(float(vals.min()), 4),
                'p25': round(float(vals.quantile(0.25)), 4),
                'median': round(float(vals.quantile(0.50)), 4),
                'p75': round(float(vals.quantile(0.75)), 4),
                'max': round(float(vals.max()), 4),
            })

    return pd.DataFrame(rows)


def save_cluster_profiles_excel(
    features_df: 'pd.DataFrame',
    labels: 'np.ndarray',
    cluster_centers: 'pd.DataFrame',
    feature_names: list,
    eval_metrics: dict,
    session: Session,
    scaler=None,
    impressions_df: 'pd.DataFrame | None' = None,
    articles_df: 'pd.DataFrame | None' = None,
    subscriber_label: str = "Number of Subscribers",
    per_cluster_silhouette: 'dict | None' = None,
) -> Path:
    """Save cluster profiles to an Excel file in the clusters/ directory.

    Creates a multi-sheet workbook:
      - **Cluster Summary**: interpretable report with raw averages,
        subscriber counts, time-of-day/weekend breakdown, engagement span,
        diversity metrics, device breakdown, and scroll depth (computed from
        raw impressions when available).
      - **Cluster Centers (Scaled)**: centroid values per feature (scaled).
      - **Cluster Statistics**: mean and std of every feature per cluster
        plus cluster size.
      - **Summary**: metadata (n_clusters, n_users, features, eval metrics,
        per-cluster silhouette scores).
      - **Category Profiles**: top-N categories per cluster by impression
        proportion.
      - **Cluster Distributions**: percentile breakdowns (p25, median, p75)
        of key per-user metrics within each cluster.

    Args:
        features_df: DataFrame with user features *and* ``cluster_id`` column.
        labels: Cluster label array.
        cluster_centers: Cluster centers DataFrame from the clusterer.
        feature_names: Ordered list of feature column names.
        eval_metrics: Dict of evaluation metrics (silhouette etc.).
        session: Current pipeline session.
        scaler: Fitted scaler (e.g. StandardScaler) to inverse-transform
                cluster centers into interpretable values.
        impressions_df: Raw impressions DataFrame (with homepage views).
                        When provided, the Cluster Summary sheet is computed
                        directly from raw data for maximum interpretability.
        articles_df: Optional articles DataFrame with ``article_id`` and
                     ``category_str``.  Passed through to
                     ``_compute_cluster_summary`` so that category-based
                     metrics can be computed even when *impressions_df*
                     does not contain category information.
        per_cluster_silhouette: Optional dict mapping cluster_id to the
                                mean silhouette score of that cluster's users.

    Returns:
        Path to the written Excel file.
    """
    import numpy as np
    import pandas as pd

    excel_path = session.get_path("cluster_profiles.xlsx", subdir="clusters")

    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique, counts))

    # --- Sheet 1: Cluster Summary (interpretable) -----------------------------------
    user_metrics = None
    if impressions_df is not None:
        users_df = features_df[['user_id', 'cluster_id']].drop_duplicates()
        cluster_summary, user_metrics = _compute_cluster_summary(
            impressions_df, users_df,
            subscriber_label=subscriber_label, articles_df=articles_df,
        )
    elif scaler is not None:
        center_vals = cluster_centers[feature_names].values
        raw_center_vals = scaler.inverse_transform(center_vals)
        cluster_summary = pd.DataFrame(raw_center_vals, columns=feature_names)
        cluster_summary.insert(0, "cluster_id", cluster_centers["cluster_id"].values)
        cluster_summary.insert(1, "size", cluster_summary["cluster_id"].map(size_map))
    else:
        cluster_summary = None

    # --- Sheet 2: Cluster Centers (Scaled) ------------------------------------------
    centers = cluster_centers.copy()
    centers.insert(1, "size", centers["cluster_id"].map(size_map))
    pct_scaled = centers["size"] / len(labels) * 100
    centers.insert(2, "size_pct", pct_scaled.round(2))

    # --- Sheet 3: Cluster Statistics (mean + std per feature) -----------------------
    stats = get_cluster_statistics(features_df, labels, feature_cols=feature_names)

    # --- Sheet 4: Summary -----------------------------------------------------------
    summary_rows = [
        ("Number of clusters", int(len(np.unique(labels)))),
        ("Total users", int(len(labels))),
        ("Number of features", len(feature_names)),
        ("Features", ", ".join(feature_names)),
    ]
    for key, val in eval_metrics.items():
        summary_rows.append((key, val))
    if per_cluster_silhouette:
        for cid in sorted(per_cluster_silhouette):
            summary_rows.append(
                (f"silhouette_cluster_{cid}", round(per_cluster_silhouette[cid], 4))
            )

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # --- Sheet 5: Category Profiles -------------------------------------------------
    category_profiles = None
    if impressions_df is not None:
        _users = features_df[['user_id', 'cluster_id']].drop_duplicates()
        category_profiles = _compute_category_profiles(
            impressions_df, _users, articles_df=articles_df,
        )

    # --- Sheet 6: Cluster Distributions ---------------------------------------------
    distributions = None
    if user_metrics is not None:
        distributions = _compute_cluster_distributions(user_metrics)

    # --- Write workbook -------------------------------------------------------------
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if cluster_summary is not None:
            cluster_summary.to_excel(writer, sheet_name="Cluster Summary")
        centers.to_excel(writer, sheet_name="Cluster Centers (Scaled)", index=False)
        stats.to_excel(writer, sheet_name="Cluster Statistics", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if category_profiles is not None and not category_profiles.empty:
            category_profiles.to_excel(writer, sheet_name="Category Profiles", index=False)
        if distributions is not None and not distributions.empty:
            distributions.to_excel(writer, sheet_name="Cluster Distributions", index=False)

    logger.info(f"Saved cluster profiles Excel to {excel_path}")
    return excel_path


def append_recommendation_performance(
    session: Session,
    eval_results: dict,
) -> None:
    """Append a Recommendation Performance sheet to the cluster profiles Excel.

    This is called *after* evaluation completes so that per-cluster algorithm
    metrics are available.  If the Excel file does not exist or no valid
    results are provided, the call is a no-op.

    Args:
        session: Current pipeline session.
        eval_results: Dict mapping cluster_id to results DataFrame (as
                      returned by ``run_cluster_evaluation``).
    """
    import pandas as pd
    from openpyxl import load_workbook

    excel_path = session.get_path("cluster_profiles.xlsx", subdir="clusters")
    if not Path(excel_path).exists():
        logger.warning("cluster_profiles.xlsx not found; skipping recommendation performance sheet")
        return

    frames = []
    for cluster_id, results_df in eval_results.items():
        if results_df is None or (hasattr(results_df, 'empty') and results_df.empty):
            continue
        rdf = results_df.copy()
        rdf.insert(0, 'cluster_id', cluster_id)
        frames.append(rdf)

    if not frames:
        return

    perf_df = pd.concat(frames, ignore_index=True).sort_values(['cluster_id', 'algorithm'])

    wb = load_workbook(excel_path)
    if 'Recommendation Performance' in wb.sheetnames:
        del wb['Recommendation Performance']
    wb.save(excel_path)

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        perf_df.to_excel(writer, sheet_name='Recommendation Performance', index=False)

    logger.info("Appended Recommendation Performance sheet to cluster_profiles.xlsx")


def run_clustering(
    impressions_df,
    articles_df,
    config: PipelineConfig,
    session: Session,
) -> tuple:
    """Run clustering step on ALL users.
    
    IMPORTANT: This runs on ALL users including homepage-only users.
    The impressions_df should include homepage views (article_id is null)
    as this is an important clustering signal from the legacy code.
    
    Returns:
        Tuple of (features_df, labels, cluster_info)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: User Clustering (ALL users including homepage-only)")
    logger.info("=" * 60)
    
    # Check if legacy features mode is enabled
    legacy_mode = getattr(config.clustering, 'legacy_features', False)
    if legacy_mode:
        logger.info("Using LEGACY feature set (no per-category proportions, no time-of-day, with session behavior features)")
    
    # Extract features (including homepage behavior - matching legacy clustering)
    extractor = UserFeatureExtractor(
        include_categories=True,
        include_time=True,
        include_activity=True,
        include_diversity=True,
        include_homepage=True,  # Homepage behavior is a clustering signal (legacy behavior)
        legacy_mode=legacy_mode,  # Use legacy feature set if configured
        scale=True,
    )
    
    features_df = extractor.fit_transform(impressions_df, articles_df)
    logger.info(f"Extracted {len(extractor.get_feature_names())} features for {len(features_df)} users")
    
    # Save features
    save_dataframe(features_df, session.get_path("user_features.parquet"))
    
    # Cluster
    clusterer = KMeansClusterer(
        n_clusters=config.clustering.n_clusters,
        k_range=range(1, 11),
        k_selection_method=config.clustering.k_selection_method,
        random_state=config.clustering.random_state,
    )
    
    X = extractor.get_feature_matrix(features_df)
    labels = clusterer.fit_predict(X)
    
    # Add cluster labels to features
    features_df['cluster_id'] = labels
    
    # Save clustered users
    users_df = features_df[['user_id', 'cluster_id']].copy()
    save_dataframe(users_df, session.get_path("user_clusters.parquet"))
    save_dataframe(users_df, session.get_path("user_clusters.csv"), format="csv")
    
    # Evaluate clustering
    eval_metrics = clusterer.evaluate(X)
    logger.info(f"Clustering evaluation: {eval_metrics}")

    # Per-cluster silhouette scores
    per_cluster_silhouette = {}
    try:
        from sklearn.metrics import silhouette_samples
        import numpy as _np
        _sample_sil = silhouette_samples(X, labels)
        for _cid in _np.unique(labels):
            per_cluster_silhouette[int(_cid)] = float(_sample_sil[labels == _cid].mean())
        logger.info(f"Per-cluster silhouette: {per_cluster_silhouette}")
    except Exception as _e:
        logger.warning(f"Could not compute per-cluster silhouette: {_e}")
    
    # Visualize
    viz_dir = session.get_path("visualizations")
    Path(viz_dir).mkdir(exist_ok=True)
    
    visualizer = ClusterVisualizer(output_dir=viz_dir)
    
    if clusterer.metrics_:
        visualizer.plot_elbow(clusterer.metrics_)
    
    visualizer.plot_distribution(labels)
    
    cluster_centers = clusterer.get_cluster_centers(extractor.get_feature_names())
    visualizer.plot_profiles(cluster_centers)
    
    logger.info(f"Saved visualizations to {viz_dir}")
    
    # Save cluster profiles Excel to clusters/ directory
    subscriber_label = "Number of Subscribers"

    save_cluster_profiles_excel(
        features_df=features_df,
        labels=labels,
        cluster_centers=cluster_centers,
        feature_names=extractor.get_feature_names(),
        eval_metrics=eval_metrics,
        session=session,
        scaler=extractor.get_metadata().get('scaler'),
        impressions_df=impressions_df,
        articles_df=articles_df,
        subscriber_label=subscriber_label,
        per_cluster_silhouette=per_cluster_silhouette or None,
    )
    
    return features_df, labels, {
        'n_clusters': clusterer.n_clusters,
        'metrics': eval_metrics,
        'feature_names': extractor.get_feature_names(),
    }


def run_evaluation(
    interactions_df,
    users_df,
    content_df,
    config: PipelineConfig,
    session: Session,
    content_mode: str = "legacy",
) -> dict:
    """Run evaluation step.
    
    NOTE: User filtering (min_impressions_per_user) happens HERE via RecPack's
    MinItemsPerUser filter, NOT during preprocessing. This ensures clustering
    happens on ALL users, while evaluation only includes users with enough
    interactions for meaningful recommendations.
    
    IMPORTANT: Pre-calculated embeddings are REQUIRED for CB-ST algorithm.
    Generate them first with: python scripts/generate_embeddings.py --input-dir <path>
    
    Returns:
        Dictionary of results per cluster
    """
    logger.info("=" * 60)
    logger.info("STEP 4: RecPack Evaluation (user filtering applied here)")
    logger.info("=" * 60)
    
    try:
        from src.evaluation import (
            run_cluster_evaluation,
            run_cluster_evaluation_legacy_style,
            ResultsAnalyzer,
        )
    except ImportError as e:
        logger.error(f"Could not import evaluation module: {e}")
        logger.error("RecPack may not be installed. Install with: pip install recpack")
        return {}
    
    log_memory("evaluation start")

    # Check for pre-calculated embeddings file (REQUIRED for CB-ST)
    embeddings_df = None
    embedding_column = 'embedding'  # Column name from generate_embeddings.py
    
    session_embeddings_path = session.get_path("title_category_embeddings.parquet")
    input_path = Path(config.dataset.input_path)
    embeddings_path = input_path / "title_category_embeddings.parquet"
    
    # Check if CB-ST is in the enabled algorithms
    algorithm_names = [algo.name for algo in config.evaluation.algorithms if algo.enabled]
    cb_st_enabled = any(name in ('CB-ST', 'SentenceTransformerContentBased') for name in algorithm_names)
    
    if Path(session_embeddings_path).exists():
        logger.info(f"Found session embeddings at {session_embeddings_path}")
        import pandas as pd
        embeddings_df = pd.read_parquet(session_embeddings_path)
        logger.info(f"Loaded embeddings for {len(embeddings_df)} articles")
        logger.info(f"Using embedding column: '{embedding_column}'")
    elif embeddings_path.exists():
        logger.info(f"Found pre-calculated embeddings at {embeddings_path}")
        import pandas as pd
        embeddings_df = pd.read_parquet(embeddings_path)
        logger.info(f"Loaded embeddings for {len(embeddings_df)} articles")
        logger.info(f"Using embedding column: '{embedding_column}'")
    elif cb_st_enabled or content_mode == "embeddings":
        # CB-ST requires pre-calculated embeddings - throw error
        logger.error("=" * 60)
        logger.error("ERROR: Pre-calculated embeddings are REQUIRED for CB-ST")
        logger.error("=" * 60)
        logger.error(f"Expected file: {embeddings_path}")
        logger.error("")
        logger.error("Generate embeddings first with:")
        logger.error(f"  python scripts/generate_embeddings.py --input-dir {input_path}")
        logger.error("")
        logger.error("Or disable CB-ST by removing it from the algorithms list.")
        logger.error("=" * 60)
        raise FileNotFoundError(
            f"Pre-calculated embeddings required for content_mode='{content_mode}' but not found at "
            f"{embeddings_path}. Generate with: python scripts/generate_embeddings.py --input-dir {input_path}"
        )
    else:
        logger.info(f"No pre-calculated embeddings found at {embeddings_path}")
        logger.info("CB-ST is not enabled, continuing without embeddings")
    
    # Load articles for topic-level diversity metrics (CoverageK_topics, GiniK_topics)
    articles_df = None
    articles_cleaned_path = session.get_path("articles_cleaned.parquet")
    if Path(articles_cleaned_path).exists():
        articles_df = load_dataframe(articles_cleaned_path)
        logger.info(f"Loaded {len(articles_df)} articles for topic diversity metrics")
    else:
        logger.info("No articles_cleaned.parquet found; topic-level diversity metrics will be skipped")

    # Run evaluation per cluster
    results_dir = session.get_path("evaluation_results")
    
    # Extract algorithm names and params from AlgorithmConfig objects
    algorithm_names = [algo.name for algo in config.evaluation.algorithms if algo.enabled]
    algorithm_params = {
        algo.name: dict(algo.params)
        for algo in config.evaluation.algorithms
        if algo.enabled and getattr(algo, "params", None)
    }
    algorithm_grids = {
        algo.name: dict(algo.grid)
        for algo in config.evaluation.algorithms
        if algo.enabled and getattr(algo, "grid", None) and algo.grid
    }
    
    train_on_full = getattr(config.evaluation, "train_on_full_dataset", True)
    
    if train_on_full:
        # Legacy-style: train once on full dataset, aggregate metrics per cluster
        logger.info("Using LEGACY-STYLE evaluation (train on full dataset, aggregate per cluster)")
        results = run_cluster_evaluation_legacy_style(
            interactions_df=interactions_df,
            users_df=users_df,
            content_df=content_df,
            articles_df=articles_df,
            algorithms=algorithm_names,
            algorithm_params=algorithm_params,
            algorithm_grids=algorithm_grids,
            k_values=config.evaluation.k_values,
            min_items_per_user=config.clustering.min_impressions_per_user,
            output_dir=results_dir,
            embeddings_df=embeddings_df,
            embedding_column=embedding_column,
            n_most_recent_in=getattr(config.evaluation, "n_most_recent_in", 30),
            optimization_metric=getattr(config.evaluation, "optimization_metric", "NDCGK"),
            optimization_k=getattr(config.evaluation, "optimization_k", 100),
        )
    else:
        # Per-cluster training: train a separate model per cluster
        logger.info("Using PER-CLUSTER training (train separate model per cluster)")
        results = run_cluster_evaluation(
            interactions_df=interactions_df,
            users_df=users_df,
            content_df=content_df,
            articles_df=articles_df,
            algorithms=algorithm_names,
            algorithm_params=algorithm_params,
            k_values=config.evaluation.k_values,
            min_items_per_user=config.clustering.min_impressions_per_user,
            output_dir=results_dir,
            n_jobs=1,
            embeddings_df=embeddings_df,
            embedding_column=embedding_column,
        )
    
    log_memory("evaluation after cluster run")

    # Analyze results
    analyzer = ResultsAnalyzer(results)
    
    # Generate and save report
    report = analyzer.generate_report()
    
    report_path = session.get_path("evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {report_path}")
    print("\n" + report)

    # Append recommendation performance to cluster profiles Excel
    try:
        append_recommendation_performance(session, results)
    except Exception as _e:
        logger.warning(f"Could not append recommendation performance sheet: {_e}")
    
    log_memory("evaluation end")
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("Starting RICON Analysis Pipeline")
    logger.info(f"Time: {datetime.now().isoformat()}")
    
    # Load config
    try:
        config = load_or_create_config(args)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Log feature configuration
    logger.info("=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Input path: {config.dataset.input_path}")
    legacy_features = getattr(config.clustering, 'legacy_features', False)
    if legacy_features:
        logger.info("Feature mode: LEGACY (session behavior features, no per-category proportions, no time-of-day)")
    else:
        logger.info("Feature mode: MODERN (includes per-category proportions, time-of-day, entropy/gini)")
    logger.info(f"N clusters: {config.clustering.n_clusters or 'auto-detect'}")
    logger.info(f"K selection method: {config.clustering.k_selection_method}")
    content_mode = getattr(config.evaluation, "content_mode", "legacy")
    if args.full_content:
        logger.warning("--full-content is deprecated; use --content-mode full")
        content_mode = "full"
    if args.content_mode:
        content_mode = args.content_mode

    if content_mode == "full":
        logger.info("CB-ST content: FULL (category + title + body)")
    elif content_mode == "embeddings":
        logger.info("CB-ST content: EMBEDDINGS (pre-calculated bert_embedding)")
    else:
        logger.info("CB-ST content: LEGACY (category + title only)")
    train_on_full = getattr(config.evaluation, "train_on_full_dataset", True)
    logger.info(
        f"Evaluation mode: {'LEGACY (train on full, aggregate per cluster)' if train_on_full else 'PER-CLUSTER (train separate model per cluster)'}"
    )
    logger.info("=" * 60)
    
    # Create session using the config
    session = Session(config=config)
    
    logger.info(f"Session directory: {session.session_dir}")
    
    # Save config to session
    save_config(config, session.get_path("config.json"))
    
    try:
        # Step 1: Conversion
        if args.skip_conversion:
            logger.info("Skipping conversion, loading existing data...")
            articles_df = load_dataframe(session.get_path("articles.parquet"))
            impressions_df = load_dataframe(session.get_path("impressions.parquet"))
        else:
            articles_df, impressions_df = run_conversion(config, session)
        
        # Step 2: Preprocessing
        articles_df, impressions_df, interactions_df = run_preprocessing(
            articles_df, impressions_df, config, session,
            content_mode=content_mode,
        )
        gc.collect()
        
        # Step 3: Clustering
        if args.skip_clustering:
            logger.info("Skipping clustering, loading existing clusters...")
            users_df = load_dataframe(session.get_path("user_clusters.parquet"))
        else:
            features_df, labels, cluster_info = run_clustering(
                impressions_df, articles_df, config, session
            )
            users_df = features_df[['user_id', 'cluster_id']]
            del features_df, labels, cluster_info

        # Raw article/impression frames are not needed after clustering.
        del articles_df, impressions_df
        gc.collect()
        
        # Step 4: Evaluation
        if not args.skip_evaluation:
            content_df = None
            if content_mode != "embeddings":
                content_df = load_dataframe(session.get_path("articles_content.csv"))
            results = run_evaluation(
                interactions_df, users_df, content_df, config, session, content_mode=content_mode
            )
            del results
            if content_df is not None:
                del content_df
            gc.collect()

        del interactions_df, users_df
        gc.collect()
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {session.session_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
