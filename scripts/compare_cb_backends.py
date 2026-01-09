#!/usr/bin/env python3
"""
Compare CB-ST backends (sklearn vs annoy) to verify the Annoy fix.

This script runs the SentenceTransformerContentBased algorithm with both
sklearn and annoy backends and compares the results to verify that the
Annoy indexing fix (using sequential indices instead of sparse article IDs)
works correctly.
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Also need to add parent for the ricon package
parent_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_path)

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_backends(dataset_path: str, subset: str = 'all'):
    """Compare sklearn and annoy backends for CB-ST.
    
    Args:
        dataset_path: Path to the dataset directory
        subset: Which cluster subset to use ('all' or cluster ID)
    """
    from src.evaluation.recpack_pipeline import RecPackPipeline
    
    dataset_path = Path(dataset_path)
    
    # Find interactions and content files
    interactions_file = dataset_path / 'interactions.csv'
    content_file = dataset_path / 'articles_content.csv'
    
    if not interactions_file.exists():
        logger.error(f"Interactions file not found: {interactions_file}")
        return None, None
    
    # Initialize pipeline
    pipeline = RecPackPipeline(
        k_values=[10, 20, 50],
        min_items_per_user=2,
        min_users_per_item=1,
    )
    
    # Load data
    logger.info(f"Loading dataset from {dataset_path}")
    content_path = str(content_file) if content_file.exists() else None
    pipeline.load_data(str(interactions_file), content_path=content_path)
    
    # Test with sklearn backend
    logger.info("\n" + "="*60)
    logger.info("Testing CB-ST with SKLEARN backend")
    logger.info("="*60)
    
    sklearn_results = pipeline.run(algorithms=['CB-ST-sklearn'])
    
    # Test with annoy backend - need to reload pipeline to reset state
    pipeline2 = RecPackPipeline(
        k_values=[10, 20, 50],
        min_items_per_user=2,
        min_users_per_item=1,
    )
    pipeline2.load_data(str(interactions_file), content_path=content_path)
    
    logger.info("\n" + "="*60)
    logger.info("Testing CB-ST with ANNOY backend")
    logger.info("="*60)
    
    annoy_results = pipeline2.run(algorithms=['CB-ST-annoy'])
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    
    print("\n### sklearn backend results:")
    print(sklearn_results.to_string(index=False))
    
    print("\n### annoy backend results:")
    print(annoy_results.to_string(index=False))
    
    # Calculate differences
    if not sklearn_results.empty and not annoy_results.empty:
        print("\n### Metric differences (sklearn - annoy):")
        
        # Get metric columns
        metric_cols = [col for col in sklearn_results.columns if col != 'algorithm']
        
        for col in metric_cols:
            sklearn_val = sklearn_results[col].iloc[0]
            annoy_val = annoy_results[col].iloc[0]
            diff = sklearn_val - annoy_val
            pct_diff = (diff / sklearn_val * 100) if sklearn_val != 0 else float('inf')
            
            status = "✓" if abs(diff) < 0.01 else "✗"
            print(f"  {col}: {sklearn_val:.4f} vs {annoy_val:.4f} | diff: {diff:+.4f} ({pct_diff:+.2f}%) {status}")
        
        # Overall assessment
        avg_diff = np.mean([abs(sklearn_results[col].iloc[0] - annoy_results[col].iloc[0]) 
                          for col in metric_cols])
        
        print(f"\nAverage absolute difference: {avg_diff:.4f}")
        
        if avg_diff < 0.01:
            print("\n✓ SUCCESS: Both backends produce nearly identical results!")
            print("  The Annoy indexing fix is working correctly.")
        elif avg_diff < 0.05:
            print("\n⚠ WARNING: Small differences detected between backends.")
            print("  This could be due to approximate nature of Annoy.")
        else:
            print("\n✗ FAILURE: Significant differences between backends.")
            print("  The Annoy backend may still have issues.")
    
    return sklearn_results, annoy_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare CB-ST backends (sklearn vs annoy)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to processed dataset (or run directory)'
    )
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Path to run directory containing data/'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='all',
        help='Cluster subset to use'
    )
    
    args = parser.parse_args()
    
    # Resolve dataset path
    base_dir = Path(__file__).parent.parent
    
    if args.run:
        # Use run directory
        run_path = Path(args.run) if os.path.isabs(args.run) else base_dir / args.run
        dataset_path = run_path / 'data'
    elif args.dataset:
        dataset_path = Path(args.dataset) if os.path.isabs(args.dataset) else base_dir / args.dataset
    else:
        # Auto-detect: look for most recent run
        runs_dir = base_dir / 'runs'
        if runs_dir.exists():
            runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
            if runs:
                dataset_path = runs[0] / 'data'
                logger.info(f"Auto-detected run: {runs[0].name}")
            else:
                logger.error("No runs found")
                sys.exit(1)
        else:
            logger.error("No runs directory found")
            sys.exit(1)
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)
    
    compare_backends(str(dataset_path), args.subset)


if __name__ == '__main__':
    main()
