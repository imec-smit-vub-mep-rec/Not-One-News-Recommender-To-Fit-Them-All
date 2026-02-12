"""
Results analysis utilities.

Provides functions for analyzing and comparing evaluation results.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

from ..utils.logging import get_logger
from ..utils.io import load_dataframe, save_dataframe


logger = get_logger("evaluation.analysis")


def aggregate_results(
    results: Dict[int, pd.DataFrame],
    metric_prefix: str = 'NDCGK',
) -> pd.DataFrame:
    """Aggregate results across clusters.
    
    Args:
        results: Dictionary mapping cluster_id to results DataFrame
        metric_prefix: Prefix for metrics to aggregate
        
    Returns:
        Aggregated results DataFrame
    """
    all_results = []
    
    for cluster_id, df in results.items():
        df = df.copy()
        df['cluster_id'] = cluster_id
        all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()
    
    combined = pd.concat(all_results, ignore_index=True)
    
    return combined


def compute_significance(
    results_a: List[float],
    results_b: List[float],
    test: str = 'ttest',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute statistical significance between two result sets.
    
    Args:
        results_a: First set of results
        results_b: Second set of results
        test: Statistical test ('ttest', 'wilcoxon', 'mannwhitney')
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    if test == 'bootstrap':
        # Bootstrap CI for the mean paired difference, robust for small samples.
        if len(results_a) != len(results_b):
            raise ValueError("Bootstrap test requires paired samples of equal length")
        diffs = results_a - results_b
        if diffs.size == 0:
            raise ValueError("No data points for bootstrap test")
        rng = np.random.default_rng(42)
        n_bootstrap = 5000
        sampled_means = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            sample = rng.choice(diffs, size=diffs.size, replace=True)
            sampled_means[i] = sample.mean()
        statistic = float(diffs.mean())
        ci_low, ci_high = np.percentile(sampled_means, [2.5, 97.5])
        # Two-sided p-value approximation from bootstrap distribution.
        p_value = 2.0 * min(
            float(np.mean(sampled_means >= 0)),
            float(np.mean(sampled_means <= 0)),
        )
        p_value = float(np.clip(p_value, 0.0, 1.0))
    elif test == 'ttest':
        statistic, p_value = stats.ttest_ind(results_a, results_b)
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(results_a, results_b)
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(results_a, results_b)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    result = {
        'test': test,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_a': np.mean(results_a),
        'mean_b': np.mean(results_b),
        'std_a': np.std(results_a),
        'std_b': np.std(results_b),
        'effect_size': (np.mean(results_a) - np.mean(results_b)) / (
            np.sqrt((np.std(results_a)**2 + np.std(results_b)**2) / 2) + 1e-10
        ),
    }
    if test == 'bootstrap':
        result['ci_low'] = float(ci_low)
        result['ci_high'] = float(ci_high)
    return result


def compare_algorithms(
    results_df: pd.DataFrame,
    metric: str = 'NDCGK_10',
    algorithm_col: str = 'algorithm',
) -> pd.DataFrame:
    """Compare algorithms across all results.
    
    Args:
        results_df: Combined results DataFrame
        metric: Metric to compare
        algorithm_col: Column name for algorithm identifier
        
    Returns:
        Comparison DataFrame
    """
    if metric not in results_df.columns:
        logger.warning(f"Metric {metric} not found in results")
        return pd.DataFrame()
    
    comparison = results_df.groupby(algorithm_col)[metric].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    comparison = comparison.sort_values('mean', ascending=False)
    
    return comparison


def compare_clusters(
    results: Dict[int, pd.DataFrame],
    algorithm: str,
    metric: str = 'NDCGK_10',
) -> pd.DataFrame:
    """Compare a specific algorithm across clusters.
    
    Args:
        results: Dictionary mapping cluster_id to results
        algorithm: Algorithm to compare
        metric: Metric to use
        
    Returns:
        Comparison DataFrame
    """
    data = []
    
    for cluster_id, df in results.items():
        algo_results = df[df['algorithm'] == algorithm]
        
        if len(algo_results) == 0:
            continue
        
        if metric in algo_results.columns:
            value = algo_results[metric].values[0]
            data.append({
                'cluster_id': cluster_id,
                'algorithm': algorithm,
                metric: value,
            })
    
    return pd.DataFrame(data)


def find_best_algorithm_per_cluster(
    results: Dict[int, pd.DataFrame],
    metric: str = 'NDCGK_10',
) -> pd.DataFrame:
    """Find the best performing algorithm for each cluster.
    
    Args:
        results: Dictionary mapping cluster_id to results
        metric: Metric to optimize
        
    Returns:
        DataFrame with best algorithm per cluster
    """
    data = []
    
    for cluster_id, df in results.items():
        if metric not in df.columns:
            continue
        
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        
        data.append({
            'cluster_id': cluster_id,
            'best_algorithm': best_row.get('algorithm', 'unknown'),
            f'best_{metric}': best_row[metric],
        })
    
    return pd.DataFrame(data)


def compute_improvement(
    baseline_results: Dict[int, float],
    improved_results: Dict[int, float],
) -> Dict[str, Any]:
    """Compute improvement statistics.
    
    Args:
        baseline_results: Baseline metric values per cluster
        improved_results: Improved metric values per cluster
        
    Returns:
        Dictionary with improvement statistics
    """
    clusters = set(baseline_results.keys()) & set(improved_results.keys())
    
    improvements = []
    for cluster in clusters:
        baseline = baseline_results[cluster]
        improved = improved_results[cluster]
        
        if baseline > 0:
            rel_improvement = (improved - baseline) / baseline * 100
        else:
            rel_improvement = 0
        
        improvements.append({
            'cluster_id': cluster,
            'baseline': baseline,
            'improved': improved,
            'absolute_improvement': improved - baseline,
            'relative_improvement_pct': rel_improvement,
        })
    
    df = pd.DataFrame(improvements)
    
    return {
        'per_cluster': df,
        'mean_absolute': df['absolute_improvement'].mean(),
        'mean_relative_pct': df['relative_improvement_pct'].mean(),
        'num_improved': (df['absolute_improvement'] > 0).sum(),
        'num_degraded': (df['absolute_improvement'] < 0).sum(),
    }


class ResultsAnalyzer:
    """Class-based interface for results analysis.
    
    Provides comprehensive analysis of evaluation results.
    """
    
    def __init__(
        self,
        results: Optional[Dict[int, pd.DataFrame]] = None,
    ):
        """Initialize the analyzer.
        
        Args:
            results: Optional dictionary of results per cluster
        """
        self.results = results or {}
        self.combined_df: Optional[pd.DataFrame] = None
    
    def load_results(
        self,
        results_dir: str,
        pattern: str = 'cluster_*_results.csv',
    ) -> 'ResultsAnalyzer':
        """Load results from directory.
        
        Args:
            results_dir: Directory containing result files
            pattern: Glob pattern for result files
            
        Returns:
            Self for chaining
        """
        results_path = Path(results_dir)
        
        for file_path in results_path.glob(pattern):
            # Extract cluster ID from filename
            try:
                cluster_id = int(file_path.stem.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not extract cluster ID from {file_path}")
                continue
            
            df = load_dataframe(str(file_path))
            self.results[cluster_id] = df
            logger.info(f"Loaded results for cluster {cluster_id}")
        
        logger.info(f"Loaded results for {len(self.results)} clusters")
        
        return self
    
    def add_results(self, cluster_id: int, results_df: pd.DataFrame):
        """Add results for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            results_df: Results DataFrame
        """
        self.results[cluster_id] = results_df
    
    def aggregate(self) -> pd.DataFrame:
        """Aggregate all results.
        
        Returns:
            Combined DataFrame
        """
        self.combined_df = aggregate_results(self.results)
        return self.combined_df
    
    def compare_algorithms(
        self,
        metric: str = 'NDCGK_10',
    ) -> pd.DataFrame:
        """Compare algorithms.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Comparison DataFrame
        """
        if self.combined_df is None:
            self.aggregate()
        
        return compare_algorithms(self.combined_df, metric)
    
    def compare_clusters(
        self,
        algorithm: str,
        metric: str = 'NDCGK_10',
    ) -> pd.DataFrame:
        """Compare clusters for an algorithm.
        
        Args:
            algorithm: Algorithm to analyze
            metric: Metric to compare
            
        Returns:
            Comparison DataFrame
        """
        return compare_clusters(self.results, algorithm, metric)
    
    def find_best_per_cluster(
        self,
        metric: str = 'NDCGK_10',
    ) -> pd.DataFrame:
        """Find best algorithm per cluster.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            DataFrame with best algorithms
        """
        return find_best_algorithm_per_cluster(self.results, metric)
    
    def significance_test(
        self,
        algorithm_a: str,
        algorithm_b: str,
        metric: str = 'NDCGK_10',
        test: str = 'bootstrap',
    ) -> Dict[str, Any]:
        """Test significance between two algorithms.
        
        Args:
            algorithm_a: First algorithm
            algorithm_b: Second algorithm
            metric: Metric to test
            test: Statistical test to use
            
        Returns:
            Test results dictionary
        """
        results_a = []
        results_b = []
        
        for cluster_id, df in self.results.items():
            a_val = df[df['algorithm'] == algorithm_a][metric].values
            b_val = df[df['algorithm'] == algorithm_b][metric].values
            
            if len(a_val) > 0 and len(b_val) > 0:
                results_a.append(a_val[0])
                results_b.append(b_val[0])
        
        if len(results_a) < 2:
            return {'error': 'Not enough paired data points'}
        
        return compute_significance(results_a, results_b, test)
    
    def generate_report(
        self,
        metrics: List[str] = [
            'NDCGK_10',
            'NDCGK_20',
            'RecallK_10',
            'RecallK_20',
            'CoverageK_10',
            'GiniK_10',
        ],
    ) -> str:
        """Generate a text report of results.
        
        Args:
            metrics: Metrics to include in report
            
        Returns:
            Report string
        """
        lines = [
            "=" * 60,
            "EVALUATION RESULTS REPORT",
            "=" * 60,
            "",
            f"Number of clusters: {len(self.results)}",
            "",
        ]
        
        # Best per cluster
        lines.append("BEST ALGORITHM PER CLUSTER")
        lines.append("-" * 40)
        
        for metric in metrics[:1]:  # Just use first metric for best
            best_df = self.find_best_per_cluster(metric)
            for _, row in best_df.iterrows():
                lines.append(f"  Cluster {row['cluster_id']}: {row['best_algorithm']} "
                            f"({metric}={row[f'best_{metric}']:.4f})")
        
        lines.append("")
        
        # Algorithm comparison
        lines.append("ALGORITHM COMPARISON")
        lines.append("-" * 40)
        
        for metric in metrics:
            lines.append(f"\n{metric}:")
            comp = self.compare_algorithms(metric)
            for algo, row in comp.iterrows():
                lines.append(f"  {algo}: mean={row['mean']:.4f}, std={row['std']:.4f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: str):
        """Save report to file.
        
        Args:
            output_path: Path to save report
        """
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved report to {output_path}")
