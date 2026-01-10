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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, load_config, save_config, PRESET_CONFIGS
from src.utils import Session, setup_logging, get_logger, load_dataframe, save_dataframe
from src.converters import AdressaConverter, EBNeRDConverter, GenericConverter
from src.preprocessing import DataCleaner, DataValidator, behaviors_to_interactions, articles_to_content
from src.clustering import UserFeatureExtractor, KMeansClusterer, ClusterVisualizer


logger = get_logger("pipeline")


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
        choices=["adressa", "ebnerd", "custom"],
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
    
    if args.n_clusters:
        config.clustering.n_clusters = args.n_clusters
    
    # --legacy-features flag takes highest priority
    if args.legacy_features:
        config.clustering.legacy_features = True
        logger.info("CLI override: clustering.legacy_features = True")
    
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
    if dataset_format == "jsonl" or config.dataset.name == "adressa":
        converter = AdressaConverter(config=config.dataset)
    elif dataset_format == "parquet" or config.dataset.name == "ebnerd":
        converter = EBNeRDConverter(config=config.dataset)
    else:
        converter = GenericConverter(
            config=config.dataset,
        )
    
    # Convert articles
    logger.info("Converting articles...")
    articles_df = converter.convert_articles()
    
    articles_path = session.get_path("articles.parquet")
    save_dataframe(articles_df, articles_path)
    logger.info(f"Saved {len(articles_df)} articles to {articles_path}")
    
    # Convert impressions
    logger.info("Converting impressions...")
    impressions_df = converter.convert_impressions()
    
    impressions_path = session.get_path("impressions.parquet")
    save_dataframe(impressions_df, impressions_path)
    logger.info(f"Saved {len(impressions_df)} impressions to {impressions_path}")
    
    return articles_df, impressions_df


def run_preprocessing(
    articles_df,
    impressions_df,
    config: PipelineConfig,
    session: Session,
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
    
    # Create article content for content-based
    content_df = articles_to_content(cleaned_articles)
    
    content_path = session.get_path("articles_content.csv")
    save_dataframe(content_df, content_path, format="csv")
    logger.info(f"Saved article content to {content_path}")
    
    return cleaned_articles, cleaned_impressions, interactions_df


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
        k_range=range(2, 11),
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
) -> dict:
    """Run evaluation step.
    
    NOTE: User filtering (min_impressions_per_user) happens HERE via RecPack's
    MinItemsPerUser filter, NOT during preprocessing. This ensures clustering
    happens on ALL users, while evaluation only includes users with enough
    interactions for meaningful recommendations.
    
    Returns:
        Dictionary of results per cluster
    """
    logger.info("=" * 60)
    logger.info("STEP 4: RecPack Evaluation (user filtering applied here)")
    logger.info("=" * 60)
    
    try:
        from src.evaluation import run_cluster_evaluation, ResultsAnalyzer
    except ImportError as e:
        logger.error(f"Could not import evaluation module: {e}")
        logger.error("RecPack may not be installed. Install with: pip install recpack")
        return {}
    
    # Run evaluation per cluster
    results_dir = session.get_path("evaluation_results")
    
    # Extract algorithm names from AlgorithmConfig objects
    algorithm_names = [algo.name for algo in config.evaluation.algorithms if algo.enabled]
    
    results = run_cluster_evaluation(
        interactions_df=interactions_df,
        users_df=users_df,
        content_df=content_df,
        algorithms=algorithm_names,
        k_values=config.evaluation.k_values,
        min_items_per_user=config.clustering.min_impressions_per_user,  # Filter only at evaluation
        output_dir=results_dir,
        n_jobs=-1,  # Parallel cluster evaluation
    )
    
    # Analyze results
    analyzer = ResultsAnalyzer(results)
    
    # Generate and save report
    report = analyzer.generate_report()
    
    report_path = session.get_path("evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {report_path}")
    print("\n" + report)
    
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
            articles_df, impressions_df, config, session
        )
        
        # Step 3: Clustering
        if args.skip_clustering:
            logger.info("Skipping clustering, loading existing clusters...")
            users_df = load_dataframe(session.get_path("user_clusters.parquet"))
        else:
            features_df, labels, cluster_info = run_clustering(
                impressions_df, articles_df, config, session
            )
            users_df = features_df[['user_id', 'cluster_id']]
        
        # Step 4: Evaluation
        if not args.skip_evaluation:
            content_df = load_dataframe(session.get_path("articles_content.csv"))
            results = run_evaluation(
                interactions_df, users_df, content_df, config, session
            )
        
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
