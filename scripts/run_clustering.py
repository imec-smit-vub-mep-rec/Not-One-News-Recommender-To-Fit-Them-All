#!/usr/bin/env python3
"""
User clustering script.

Clusters users based on their behavior features.

Usage:
    python run_clustering.py --impressions data/impressions.parquet --articles data/articles.parquet
    python run_clustering.py --impressions data/impressions.parquet --n-clusters 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, load_dataframe, save_dataframe, ensure_dir
from src.clustering import UserFeatureExtractor, KMeansClusterer, ClusterVisualizer


logger = get_logger("clustering")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster users based on behavior",
    )
    
    parser.add_argument(
        "--impressions",
        type=str,
        required=True,
        help="Path to impressions file (parquet or csv)",
    )
    
    parser.add_argument(
        "--articles",
        type=str,
        help="Path to articles file (for category features)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./clustering_results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Number of clusters (default: auto-detect)",
    )
    
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum k for auto-detection",
    )
    
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum k for auto-detection",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="elbow",
        choices=["elbow", "silhouette", "combined"],
        help="K selection method",
    )
    
    parser.add_argument(
        "--no-categories",
        action="store_true",
        help="Exclude category features",
    )
    
    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Exclude time features",
    )
    
    parser.add_argument(
        "--no-homepage",
        action="store_true",
        help="Exclude homepage behavior features (not recommended - these are important clustering signals)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("Starting user clustering")
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load data
    logger.info(f"Loading impressions from {args.impressions}")
    impressions_df = load_dataframe(args.impressions)
    logger.info(f"Loaded {len(impressions_df)} impressions")
    
    articles_df = None
    if args.articles:
        logger.info(f"Loading articles from {args.articles}")
        articles_df = load_dataframe(args.articles)
        logger.info(f"Loaded {len(articles_df)} articles")
    
    # Extract features (including homepage behavior by default - matching legacy clustering)
    logger.info("Extracting user features...")
    extractor = UserFeatureExtractor(
        include_categories=not args.no_categories,
        include_time=not args.no_time,
        include_activity=True,
        include_diversity=True,
        include_homepage=not args.no_homepage,  # Homepage behavior is a clustering signal
        scale=True,
    )
    
    features_df = extractor.fit_transform(impressions_df, articles_df)
    
    feature_names = extractor.get_feature_names()
    logger.info(f"Extracted {len(feature_names)} features for {len(features_df)} users")
    
    # Save features
    features_path = Path(args.output_dir) / "user_features.parquet"
    save_dataframe(features_df, str(features_path))
    logger.info(f"Saved features to {features_path}")
    
    # Cluster
    logger.info("Clustering users...")
    clusterer = KMeansClusterer(
        n_clusters=args.n_clusters,
        k_range=range(args.k_min, args.k_max + 1),
        k_selection_method=args.method,
        random_state=args.seed,
    )
    
    X = extractor.get_feature_matrix(features_df)
    labels = clusterer.fit_predict(X)
    
    # Add labels to dataframe
    features_df['cluster_id'] = labels
    
    # Save results
    users_path = Path(args.output_dir) / "user_clusters.parquet"
    save_dataframe(features_df[['user_id', 'cluster_id']], str(users_path))
    
    users_csv_path = Path(args.output_dir) / "user_clusters.csv"
    save_dataframe(features_df[['user_id', 'cluster_id']], str(users_csv_path), format="csv")
    
    logger.info(f"Saved cluster assignments to {users_path}")
    
    # Evaluate
    eval_metrics = clusterer.evaluate(X)
    logger.info(f"Clustering evaluation: {eval_metrics}")
    
    # Visualize
    viz_dir = Path(args.output_dir) / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    visualizer = ClusterVisualizer(output_dir=str(viz_dir))
    
    if clusterer.metrics_:
        visualizer.plot_elbow(clusterer.metrics_)
    
    visualizer.plot_distribution(labels)
    
    cluster_centers = clusterer.get_cluster_centers(feature_names)
    visualizer.plot_profiles(cluster_centers)
    
    logger.info(f"Saved visualizations to {viz_dir}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Clustering Summary")
    logger.info("=" * 60)
    logger.info(f"Number of users: {len(features_df)}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Number of clusters: {clusterer.n_clusters}")
    
    import numpy as np
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        logger.info(f"  Cluster {cluster_id}: {count} users ({pct:.1f}%)")
    
    if 'silhouette_score' in eval_metrics:
        logger.info(f"Silhouette score: {eval_metrics['silhouette_score']:.4f}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
