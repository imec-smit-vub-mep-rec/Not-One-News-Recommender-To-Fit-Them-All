"""
User clustering functionality.

Provides K-Means clustering and utilities for finding optimal cluster count.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import Parallel, delayed
import warnings

from ..utils.logging import get_logger


logger = get_logger("clustering.clustering")

# Single source of truth for clustering defaults.
DEFAULT_K_RANGE = range(1, 11)
DEFAULT_K_SELECTION_METHOD = "elbow"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10
DEFAULT_N_JOBS = -1
DEFAULT_USE_MINIBATCH = False
DEFAULT_MINIBATCH_THRESHOLD = 50_000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000


def _log_resolved_clustering_params(params: Dict[str, Any]) -> None:
    """Log resolved clustering parameters in a deterministic order."""
    logger.info("-" * 60)
    logger.info("RESOLVED CLUSTERING PARAMETERS")
    logger.info("-" * 60)
    for key in sorted(params):
        logger.info(f"{key}: {params[key]}")
    logger.info("-" * 60)


def _fit_kmeans_for_k(
    X: np.ndarray,
    k: int,
    random_state: int,
    n_init: int,
    use_minibatch: bool = DEFAULT_USE_MINIBATCH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_extra_metrics: bool = True,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Fit KMeans for a single k value and compute metrics.
    
    Helper function for parallel K-selection.
    
    Args:
        X: Feature matrix
        k: Number of clusters
        random_state: Random state
        n_init: Number of initializations
        use_minibatch: Whether to use MiniBatchKMeans
        batch_size: Batch size for MiniBatchKMeans
        compute_extra_metrics: Whether to compute silhouette/CH/DB scores
            (not needed for elbow method which only uses inertia)
        silhouette_sample_size: Subsample size for silhouette_score.
            Full pairwise silhouette is O(n²) and infeasible for large datasets.
            Set to 0 or None to use all samples (WARNING: very slow for n > 50k).
        
    Returns:
        Dictionary with k value and computed metrics
    """
    # Select KMeans variant
    if use_minibatch:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            batch_size=batch_size,
        )
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    
    labels = kmeans.fit_predict(X)
    
    result = {
        'k': k,
        'inertia': kmeans.inertia_,
    }
    
    if k > 1 and compute_extra_metrics:
        # Silhouette score is O(n²) — subsample for large datasets
        sil_sample = silhouette_sample_size if (silhouette_sample_size and len(X) > silhouette_sample_size) else None
        result['silhouette_score'] = silhouette_score(
            X, labels, sample_size=sil_sample, random_state=random_state,
        )
        result['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        result['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    else:
        result['silhouette_score'] = np.nan
        result['calinski_harabasz_score'] = np.nan
        result['davies_bouldin_score'] = np.nan
    
    return result


def find_optimal_k(
    X: np.ndarray,
    k_range: range = DEFAULT_K_RANGE,
    method: str = DEFAULT_K_SELECTION_METHOD,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    n_jobs: int = DEFAULT_N_JOBS,
    use_minibatch: bool = DEFAULT_USE_MINIBATCH,
    minibatch_threshold: int = DEFAULT_MINIBATCH_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> Tuple[int, Dict[str, Any]]:
    """Find optimal number of clusters.
    
    Uses parallel processing to evaluate multiple k values simultaneously.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Range of k values to try
        method: Method for selecting optimal k ('elbow', 'silhouette', 'combined')
        random_state: Random state for reproducibility
        n_init: Number of initializations for k-means
        n_jobs: Number of parallel jobs (-1 = all cores)
        use_minibatch: Whether to use MiniBatchKMeans (None = auto based on threshold)
        minibatch_threshold: Auto-enable MiniBatch if n_samples > this threshold
        batch_size: Batch size for MiniBatchKMeans
        
    Returns:
        Tuple of (optimal_k, metrics_dict)
    """
    n_samples = len(X)
    k_list = list(k_range)
    
    # Auto-detect minibatch usage for large datasets
    if not use_minibatch and n_samples > minibatch_threshold:
        use_minibatch = True
        logger.info(f"Auto-enabled MiniBatchKMeans for K-selection ({n_samples:,} > {minibatch_threshold:,} samples)")
    
    # Elbow method only needs inertia — skip expensive O(n²) silhouette/CH/DB scores
    compute_extra = method != 'elbow'
    extra_info = ""
    if compute_extra:
        extra_info = (
            f" (with silhouette/CH/DB metrics, silhouette subsampled to "
            f"{silhouette_sample_size:,})"
        )
    else:
        extra_info = " (inertia only — skipping O(n²) silhouette for speed)"
    logger.info(f"Finding optimal k in range {k_list} using {method} method (parallel, n_jobs={n_jobs}){extra_info}...")
    
    # Run K-means fitting in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_kmeans_for_k)(
            X, k, random_state, n_init, use_minibatch, batch_size,
            compute_extra_metrics=compute_extra,
            silhouette_sample_size=silhouette_sample_size,
        )
        for k in k_list
    )
    
    # Sort results by k (parallel execution may return out of order)
    results = sorted(results, key=lambda x: x['k'])
    
    # Build metrics dict in the same format as before
    metrics = {
        'k_values': [r['k'] for r in results],
        'inertias': [r['inertia'] for r in results],
        'silhouette_scores': [r['silhouette_score'] for r in results],
        'calinski_harabasz_scores': [r['calinski_harabasz_score'] for r in results],
        'davies_bouldin_scores': [r['davies_bouldin_score'] for r in results],
    }
    
    # Find optimal k
    if method == 'silhouette':
        # Maximum silhouette score (use nanargmax to handle NaN values correctly)
        sil_scores = np.array(metrics['silhouette_scores'])
        optimal_idx = int(np.nanargmax(sil_scores))
        optimal_k = metrics['k_values'][optimal_idx]
        
    elif method == 'elbow':
        # Elbow method using kneed (Kneedle algorithm)
        from kneed import KneeLocator

        k_list = metrics['k_values']
        inertias = np.array(metrics['inertias'])
        kneedle = KneeLocator(
            k_list,
            inertias,
            curve='convex',
            direction='decreasing',
        )
        optimal_k = kneedle.elbow
        if optimal_k is None:
            optimal_k = 4 if 4 in k_list else k_list[len(k_list) // 2]
            logger.info(f"No elbow found by kneed, using fallback k={optimal_k}")
        
    elif method == 'combined':
        # Combined score: normalize and combine silhouette and elbow
        sil_scores = np.array(metrics['silhouette_scores'])
        inertias = np.array(metrics['inertias'])
        
        # Normalize silhouette scores
        sil_norm = (sil_scores - np.nanmin(sil_scores)) / (np.nanmax(sil_scores) - np.nanmin(sil_scores) + 1e-10)
        
        # Normalize inverted inertias
        inv_inertia = 1 / (inertias + 1e-10)
        inertia_norm = (inv_inertia - np.min(inv_inertia)) / (np.max(inv_inertia) - np.min(inv_inertia) + 1e-10)
        
        # Combined score
        combined = 0.5 * np.nan_to_num(sil_norm) + 0.5 * inertia_norm
        optimal_idx = np.argmax(combined)
        optimal_k = metrics['k_values'][optimal_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    metrics['optimal_k'] = optimal_k
    metrics['method'] = method
    
    logger.info(f"Optimal k={optimal_k} found using {method} method")
    
    return optimal_k, metrics


def cluster_users(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    use_minibatch: Optional[bool] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    auto_minibatch_threshold: int = DEFAULT_MINIBATCH_THRESHOLD,
) -> Tuple[np.ndarray, Any]:
    """Cluster users using K-Means.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        n_init: Number of initializations
        use_minibatch: Whether to use MiniBatchKMeans (None = auto-detect based on dataset size)
        batch_size: Batch size for MiniBatchKMeans
        auto_minibatch_threshold: Use MiniBatchKMeans if n_samples exceeds this (default 50k)
        
    Returns:
        Tuple of (cluster_labels, fitted_model)
    """
    n_samples = len(X)
    
    # Auto-detect minibatch usage for large datasets
    if use_minibatch is None:
        use_minibatch = n_samples > auto_minibatch_threshold
        if use_minibatch:
            logger.info(f"Auto-enabled MiniBatchKMeans for {n_samples:,} samples (threshold: {auto_minibatch_threshold:,})")
    
    logger.info(f"Clustering {n_samples:,} users into {n_clusters} clusters...")
    
    if use_minibatch:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init=n_init,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
    
    labels = model.fit_predict(X)
    
    # Log cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Cluster sizes:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        logger.info(f"  Cluster {cluster}: {count} users ({pct:.1f}%)")
    
    return labels, model


def assign_cluster_labels(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    user_col: str = 'user_id',
    cluster_col: str = 'cluster_id',
) -> pd.DataFrame:
    """Add cluster labels to features DataFrame.
    
    Args:
        features_df: DataFrame with user features
        labels: Cluster labels array
        user_col: Name of user ID column
        cluster_col: Name for cluster ID column
        
    Returns:
        DataFrame with cluster labels
    """
    result = features_df.copy()
    result[cluster_col] = labels
    return result


def get_cluster_statistics(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute statistics for each cluster.
    
    Args:
        features_df: DataFrame with user features
        labels: Cluster labels array
        feature_cols: Feature columns to compute statistics for
        
    Returns:
        DataFrame with cluster statistics
    """
    df = features_df.copy()
    df['cluster_id'] = labels
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['user_id', 'cluster_id']]
    
    # Compute mean and std for each cluster
    stats = []
    for cluster in sorted(df['cluster_id'].unique()):
        cluster_data = df[df['cluster_id'] == cluster][feature_cols]
        
        stat_row = {'cluster_id': cluster, 'size': len(cluster_data)}
        
        for col in feature_cols:
            stat_row[f'{col}_mean'] = cluster_data[col].mean()
            stat_row[f'{col}_std'] = cluster_data[col].std()
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats)


def get_cluster_centers(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Get cluster centers as a DataFrame.
    
    Args:
        model: Fitted KMeans model
        feature_names: Names of features
        
    Returns:
        DataFrame with cluster centers
    """
    centers = model.cluster_centers_
    
    df = pd.DataFrame(centers, columns=feature_names)
    df['cluster_id'] = range(len(centers))
    
    # Reorder columns
    cols = ['cluster_id'] + feature_names
    return df[cols]


class KMeansClusterer:
    """Class-based interface for K-Means clustering.
    
    Provides a stateful interface for clustering with automatic
    optimal k selection.
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        k_range: range = DEFAULT_K_RANGE,
        k_selection_method: str = DEFAULT_K_SELECTION_METHOD,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_N_INIT,
        use_minibatch: bool = DEFAULT_USE_MINIBATCH,
        minibatch_threshold: int = DEFAULT_MINIBATCH_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_jobs: int = DEFAULT_N_JOBS,
        silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    ):
        """Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters (None = auto-select)
            k_range: Range for auto-selection
            k_selection_method: Method for selecting k
            random_state: Random state
            n_init: Number of initializations
            use_minibatch: Whether to use MiniBatchKMeans
            minibatch_threshold: Auto-enable MiniBatch if samples exceed threshold
            batch_size: Batch size for MiniBatchKMeans
            n_jobs: Number of parallel jobs for K-selection
            silhouette_sample_size: Subsample size for silhouette metrics
        """
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.k_selection_method = k_selection_method
        self.random_state = random_state
        self.n_init = n_init
        self.use_minibatch = use_minibatch
        self.minibatch_threshold = minibatch_threshold
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.silhouette_sample_size = silhouette_sample_size
        
        self.model: Optional[Any] = None
        self.labels_: Optional[np.ndarray] = None
        self.metrics_: Optional[Dict[str, Any]] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'KMeansClusterer':
        """Fit the clusterer.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        _log_resolved_clustering_params(
            {
                "n_clusters": self.n_clusters if self.n_clusters is not None else "auto",
                "k_range": list(self.k_range),
                "k_selection_method": self.k_selection_method,
                "random_state": self.random_state,
                "n_init": self.n_init,
                "use_minibatch": self.use_minibatch,
                "minibatch_threshold": self.minibatch_threshold,
                "batch_size": self.batch_size,
                "n_jobs": self.n_jobs,
                "silhouette_sample_size": self.silhouette_sample_size,
            }
        )

        # Auto-select k if not specified
        if self.n_clusters is None:
            optimal_k, self.metrics_ = find_optimal_k(
                X,
                k_range=self.k_range,
                method=self.k_selection_method,
                random_state=self.random_state,
                n_init=self.n_init,
                n_jobs=self.n_jobs,
                use_minibatch=self.use_minibatch,
                minibatch_threshold=self.minibatch_threshold,
                batch_size=self.batch_size,
                silhouette_sample_size=self.silhouette_sample_size,
            )
            self.n_clusters = optimal_k
        
        # Cluster
        self.labels_, self.model = cluster_users(
            X,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            use_minibatch=self.use_minibatch,
            batch_size=self.batch_size,
            auto_minibatch_threshold=self.minibatch_threshold,
        )
        
        self.is_fitted = True
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_cluster_centers(self, feature_names: List[str]) -> pd.DataFrame:
        """Get cluster centers.
        
        Args:
            feature_names: Feature column names
            
        Returns:
            DataFrame with cluster centers
        """
        if not self.is_fitted:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        return get_cluster_centers(self.model, feature_names)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get clustering metrics.
        
        Returns:
            Metrics dictionary
        """
        if self.metrics_ is None:
            return {}
        return self.metrics_.copy()
    
    def evaluate(
        self,
        X: np.ndarray,
        silhouette_sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate clustering quality.
        
        Args:
            X: Feature matrix
            silhouette_sample_size: Subsample size for silhouette_score.
                Full pairwise silhouette is O(n²) and infeasible for large datasets.
                Set to 0 or None to use all samples (WARNING: very slow for n > 50k).
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        metrics = {
            'n_clusters': self.n_clusters,
            'inertia': self.model.inertia_,
        }
        
        if self.n_clusters > 1:
            if silhouette_sample_size is None:
                silhouette_sample_size = self.silhouette_sample_size
            # Silhouette score is O(n²) — subsample for large datasets
            sil_sample = silhouette_sample_size if (silhouette_sample_size and len(X) > silhouette_sample_size) else None
            if sil_sample:
                logger.info(f"Computing silhouette score on subsample of {sil_sample:,} (full dataset: {len(X):,})")
            metrics['silhouette_score'] = silhouette_score(
                X, self.labels_, sample_size=sil_sample, random_state=self.random_state,
            )
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, self.labels_)
        
        return metrics
