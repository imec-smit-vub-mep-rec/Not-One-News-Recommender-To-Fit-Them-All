#!/usr/bin/env python3
"""
RecPack evaluation script.

Evaluates recommendation algorithms on clustered user data.

Usage:
    python run_evaluation.py --interactions data/interactions.csv --clusters data/user_clusters.csv
    python run_evaluation.py --interactions data/interactions.csv --content data/articles_content.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, load_dataframe, save_dataframe, ensure_dir


logger = get_logger("evaluation")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation algorithms",
    )
    
    parser.add_argument(
        "--interactions",
        type=str,
        required=True,
        help="Path to interactions file (csv with user_id, article_id, impression_time)",
    )
    
    parser.add_argument(
        "--clusters",
        type=str,
        help="Path to user clusters file (csv with user_id, cluster_id)",
    )
    
    parser.add_argument(
        "--content",
        type=str,
        help="Path to article content file (for content-based algorithm)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["Popularity", "ItemKNN", "EASE"],
        help="Algorithms to evaluate",
    )
    
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="K values for metrics",
    )
    
    parser.add_argument(
        "--min-items-per-user",
        type=int,
        default=5,
        help="Minimum interactions per user for RecPack filter (default: 5). "
             "NOTE: Clustering should include ALL users; this filter is only "
             "applied during evaluation.",
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
    
    logger.info("Starting RecPack evaluation")
    
    # Check for RecPack
    try:
        from src.evaluation import RecPackPipeline, ResultsAnalyzer, run_cluster_evaluation
    except ImportError as e:
        logger.error(f"Failed to import evaluation module: {e}")
        logger.error("Make sure RecPack is installed: pip install recpack")
        sys.exit(1)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load data
    logger.info(f"Loading interactions from {args.interactions}")
    interactions_df = load_dataframe(args.interactions)
    logger.info(f"Loaded {len(interactions_df)} interactions")
    
    content_df = None
    if args.content:
        logger.info(f"Loading content from {args.content}")
        content_df = load_dataframe(args.content)
        logger.info(f"Loaded content for {len(content_df)} articles")
        
        # Include content-based algorithm
        if "CB-ST" not in args.algorithms:
            args.algorithms.append("CB-ST")
    
    clusters_df = None
    if args.clusters:
        logger.info(f"Loading clusters from {args.clusters}")
        clusters_df = load_dataframe(args.clusters)
        logger.info(f"Loaded {clusters_df['cluster_id'].nunique()} clusters")
    
    # Run evaluation
    if clusters_df is not None:
        # Evaluate per cluster
        logger.info("Running per-cluster evaluation...")
        
        results = run_cluster_evaluation(
            interactions_df=interactions_df,
            users_df=clusters_df,
            content_df=content_df,
            algorithms=args.algorithms,
            k_values=args.k_values,
            min_items_per_user=args.min_items_per_user,
            output_dir=args.output_dir,
        )
        
        # Analyze results
        analyzer = ResultsAnalyzer(results)
        report = analyzer.generate_report()
        
        # Save report
        report_path = Path(args.output_dir) / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("\n" + report)
        
    else:
        # Single evaluation
        logger.info("Running global evaluation...")
        
        pipeline = RecPackPipeline(
            k_values=args.k_values,
            min_items_per_user=args.min_items_per_user,
        )
        
        # Save interactions temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            interactions_df.to_csv(f.name, index=False)
            interactions_path = f.name
        
        try:
            content_path = None
            if content_df is not None:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    content_df.to_csv(f.name, index=False)
                    content_path = f.name
            
            pipeline.load_data(
                interactions_path=interactions_path,
                content_path=content_path,
            )
            
            results = pipeline.run(algorithms=args.algorithms)
            
            # Save results
            results_path = Path(args.output_dir) / "results.csv"
            pipeline.save_results(str(results_path))
            
            print("\nResults:")
            print(results)
            
        finally:
            import os
            os.unlink(interactions_path)
            if content_path:
                os.unlink(content_path)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
