#!/usr/bin/env python3
"""
Benchmark script for measuring pipeline performance.

Measures execution time of key pipeline stages to track optimization improvements.

Usage:
    python benchmark_pipeline.py --n-users 5000 --n-impressions 100000
    python benchmark_pipeline.py --output benchmark_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_data(
    n_users: int = 5000,
    n_impressions: int = 100000,
    n_articles: int = 1000,
    n_categories: int = 10,
    homepage_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Generate synthetic test data for benchmarking.
    
    Args:
        n_users: Number of unique users
        n_impressions: Total number of impressions
        n_articles: Number of unique articles
        n_categories: Number of categories
        homepage_ratio: Fraction of homepage views
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (impressions_df, articles_df)
    """
    np.random.seed(seed)
    
    # Generate impressions
    user_ids = np.random.randint(0, n_users, n_impressions)
    
    # Some impressions are homepage views (null article_id)
    is_homepage = np.random.random(n_impressions) < homepage_ratio
    article_ids = np.where(
        is_homepage,
        None,
        np.random.randint(1, n_articles + 1, n_impressions).astype(str)
    )
    
    # Generate timestamps (spread over 30 days in milliseconds)
    base_time = int(datetime(2024, 1, 1).timestamp() * 1000)
    impression_times = base_time + np.random.randint(0, 30 * 24 * 3600 * 1000, n_impressions)
    impression_times = np.sort(impression_times)  # Sort chronologically
    
    # Generate read times (exponential distribution)
    read_times = np.random.exponential(30, n_impressions)
    
    # Generate session IDs (new session every ~10 impressions per user)
    session_ids = [f"s_{i // 10}_{user_ids[i]}" for i in range(n_impressions)]
    
    impressions_df = pd.DataFrame({
        'user_id': user_ids,
        'article_id': article_ids,
        'impression_time': impression_times,
        'read_time': read_times,
        'session_id': session_ids,
    })
    
    # Generate articles
    categories = [f"cat_{i}" for i in range(n_categories)]
    articles_df = pd.DataFrame({
        'article_id': [str(i) for i in range(1, n_articles + 1)],
        'category_str': np.random.choice(categories, n_articles),
        'title': [f"Article {i}" for i in range(1, n_articles + 1)],
        'content': [f"Content for article {i}. " * 10 for i in range(1, n_articles + 1)],
    })
    
    return impressions_df, articles_df


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        return False


def benchmark_feature_extraction(
    impressions_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    n_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark feature extraction."""
    from src.clustering import UserFeatureExtractor
    
    times = []
    for _ in range(n_runs):
        extractor = UserFeatureExtractor(
            include_categories=True,
            include_time=True,
            include_activity=True,
            include_diversity=True,
            include_homepage=True,
            scale=True,
        )
        
        with Timer() as t:
            features_df = extractor.fit_transform(impressions_df, articles_df)
        
        times.append(t.elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'n_users': len(features_df),
        'n_features': len(extractor.get_feature_names()),
    }


def benchmark_k_selection(
    X: np.ndarray,
    k_range: range = range(2, 11),
    n_runs: int = 2,
) -> Dict[str, float]:
    """Benchmark K-selection (elbow method)."""
    from src.clustering.clustering import find_optimal_k
    
    times = []
    optimal_k = None
    
    for _ in range(n_runs):
        with Timer() as t:
            optimal_k, metrics = find_optimal_k(
                X,
                k_range=k_range,
                method='elbow',
                random_state=42,
                n_init=10,
            )
        times.append(t.elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'optimal_k': optimal_k,
        'k_range': f"{k_range.start}-{k_range.stop - 1}",
    }


def benchmark_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    n_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark clustering."""
    from src.clustering.clustering import cluster_users
    
    times = []
    
    for _ in range(n_runs):
        with Timer() as t:
            labels, model = cluster_users(
                X,
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            )
        times.append(t.elapsed)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'n_clusters': n_clusters,
        'n_samples': len(X),
    }


def benchmark_full_pipeline(
    impressions_df: pd.DataFrame,
    articles_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Run full pipeline benchmark."""
    from src.clustering import UserFeatureExtractor, KMeansClusterer
    
    results = {}
    
    # Feature extraction
    print("Benchmarking feature extraction...")
    results['feature_extraction'] = benchmark_feature_extraction(
        impressions_df, articles_df
    )
    print(f"  Mean time: {results['feature_extraction']['mean']:.3f}s")
    
    # Get features for clustering benchmarks
    extractor = UserFeatureExtractor(scale=True)
    features_df = extractor.fit_transform(impressions_df, articles_df)
    X = extractor.get_feature_matrix(features_df)
    
    # K-selection
    print("Benchmarking K-selection...")
    results['k_selection'] = benchmark_k_selection(X, k_range=range(2, 8))
    print(f"  Mean time: {results['k_selection']['mean']:.3f}s")
    
    # Clustering
    print("Benchmarking clustering...")
    results['clustering'] = benchmark_clustering(X, n_clusters=5)
    print(f"  Mean time: {results['clustering']['mean']:.3f}s")
    
    # Calculate totals
    results['total'] = {
        'mean': sum(r['mean'] for r in [
            results['feature_extraction'],
            results['k_selection'],
            results['clustering'],
        ]),
    }
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON."""
    results['timestamp'] = datetime.now().isoformat()
    results['version'] = 'baseline'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


def compare_results(baseline_path: str, current_results: Dict[str, Any]):
    """Compare current results against baseline."""
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"No baseline found at {baseline_path}")
        return
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Stage':<25} {'Baseline':>10} {'Current':>10} {'Speedup':>10}")
    print("-" * 60)
    
    stages = ['feature_extraction', 'k_selection', 'clustering', 'total']
    
    for stage in stages:
        if stage in baseline and stage in current_results:
            baseline_time = baseline[stage].get('mean', 0)
            current_time = current_results[stage].get('mean', 0)
            
            if current_time > 0:
                speedup = baseline_time / current_time
            else:
                speedup = float('inf')
            
            print(f"{stage:<25} {baseline_time:>10.3f}s {current_time:>10.3f}s {speedup:>9.2f}x")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline performance")
    parser.add_argument("--n-users", type=int, default=5000, help="Number of users")
    parser.add_argument("--n-impressions", type=int, default=100000, help="Number of impressions")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--baseline", type=str, help="Baseline results file to compare against")
    parser.add_argument("--save-baseline", action="store_true", help="Save as baseline")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIPELINE BENCHMARK")
    print("=" * 60)
    print(f"Users: {args.n_users}, Impressions: {args.n_impressions}")
    print()
    
    # Generate test data
    print("Generating test data...")
    impressions_df, articles_df = generate_test_data(
        n_users=args.n_users,
        n_impressions=args.n_impressions,
    )
    print(f"  Generated {len(impressions_df)} impressions, {len(articles_df)} articles")
    print()
    
    # Run benchmarks
    results = benchmark_full_pipeline(impressions_df, articles_df)
    
    # Add metadata
    results['config'] = {
        'n_users': args.n_users,
        'n_impressions': args.n_impressions,
    }
    
    # Save results
    output_path = args.output
    if args.save_baseline:
        output_path = "benchmark_baseline.json"
    
    save_results(results, output_path)
    
    # Compare against baseline if provided
    if args.baseline:
        compare_results(args.baseline, results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {results['total']['mean']:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
