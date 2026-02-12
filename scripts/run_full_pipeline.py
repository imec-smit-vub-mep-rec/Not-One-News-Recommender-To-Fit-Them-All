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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, load_config, save_config, PRESET_CONFIGS
from src.utils import Session, setup_logging, get_logger, load_dataframe, save_dataframe
from src.converters import ADConverter, AdressaConverter, EBNeRDConverter, GenericConverter
from src.preprocessing import DataCleaner, DataValidator, behaviors_to_interactions, articles_to_content
from src.clustering import UserFeatureExtractor, KMeansClusterer, ClusterVisualizer


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
    if config.dataset.name != "ad":
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
        choices=["ad", "adressa", "ebnerd", "custom"],
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
        "--full-content",
        action="store_true",
        help="Use full article content (title + body) for CB-ST instead of just category + title (legacy default)",
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
    if config.dataset.name == "ad":
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
    full_content: bool = False,
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
    # Default: category + title only (legacy behavior)
    # With full_content=True: category + title + body (richer but slower)
    content_df = articles_to_content(cleaned_articles, full_content=full_content)
    
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
    
    IMPORTANT: Pre-calculated embeddings are REQUIRED for CB-ST algorithm.
    Generate them first with: python scripts/generate_embeddings.py --input-dir <path>
    
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
    elif cb_st_enabled:
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
            f"Pre-calculated embeddings required for CB-ST but not found at {embeddings_path}. "
            f"Generate with: python scripts/generate_embeddings.py --input-dir {input_path}"
        )
    else:
        logger.info(f"No pre-calculated embeddings found at {embeddings_path}")
        logger.info("CB-ST is not enabled, continuing without embeddings")
    
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
        embeddings_df=embeddings_df,
        embedding_column=embedding_column,
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
    if args.full_content:
        logger.info("CB-ST content: FULL (category + title + body)")
    else:
        logger.info("CB-ST content: LEGACY (category + title only)")
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
            full_content=args.full_content,
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
