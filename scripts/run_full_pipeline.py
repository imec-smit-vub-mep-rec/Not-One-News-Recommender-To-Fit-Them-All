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
    """Load config from file or create from arguments."""
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    elif args.dataset:
        if args.dataset in PRESET_CONFIGS:
            # PRESET_CONFIGS contains DatasetConfig objects, not full PipelineConfig
            preset = PRESET_CONFIGS[args.dataset]
            # Set input_path from command line
            if args.input_dir:
                preset.input_path = args.input_dir
            config = PipelineConfig(dataset=preset)
            logger.info(f"Using preset config for {args.dataset}")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    else:
        # Create default config
        from src.config import DatasetConfig, SessionConfig
        config = PipelineConfig(
            dataset=DatasetConfig(name="custom", input_path=args.input_dir or ""),
        )
    
    # Override with command line args
    if args.input_dir:
        config.dataset.input_path = args.input_dir
    
    if args.output_dir:
        config.session.base_output_dir = args.output_dir
    
    if args.n_clusters:
        config.clustering.n_clusters = args.n_clusters
    
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
    
    IMPORTANT: This preprocessing does NOT filter users by impression count.
    Clustering must happen on ALL users. User filtering is only applied
    later during the RecPack evaluation step.
    
    Returns:
        Tuple of (cleaned_articles, cleaned_impressions, interactions)
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing (for clustering - NO user filtering)")
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
    
    # Clean data for CLUSTERING - NO user filtering!
    # User filtering only happens in RecPack evaluation.
    cleaner = DataCleaner(
        min_impressions_per_user=config.clustering.min_impressions_per_user,
        remove_empty_articles=True,
        clean_categories=True,
        filter_users=False,  # CRITICAL: Do NOT filter users before clustering
    )
    
    cleaned_articles = cleaner.clean_articles(articles_df)
    cleaned_impressions = cleaner.clean_impressions(impressions_df, cleaned_articles)
    
    logger.info(f"Cleaning stats: {cleaner.get_stats()}")
    logger.info(f"NOTE: All {cleaner.get_stats().get('final_users', 'N/A')} users retained for clustering")
    
    # Save cleaned data
    save_dataframe(cleaned_articles, session.get_path("articles_cleaned.parquet"))
    save_dataframe(cleaned_impressions, session.get_path("impressions_cleaned.parquet"))
    
    # Create interactions for RecPack (from ALL users - filtering happens in RecPack)
    interactions_df = behaviors_to_interactions(cleaned_impressions)
    
    interactions_path = session.get_path("interactions.csv")
    save_dataframe(interactions_df, interactions_path, format="csv")
    logger.info(f"Saved {len(interactions_df)} interactions to {interactions_path}")
    
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
    """Run clustering step.
    
    Returns:
        Tuple of (features_df, labels, cluster_info)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: User Clustering")
    logger.info("=" * 60)
    
    # Extract features
    extractor = UserFeatureExtractor(
        include_categories=True,
        include_time=True,
        include_activity=True,
        include_diversity=True,
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
