"""
User clustering functionality.

Provides K-Means clustering and utilities for finding optimal cluster count.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

from ..utils.logging import get_logger


logger = get_logger("clustering.clustering")


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    method: str = 'elbow',
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[int, Dict[str, Any]]:
    """Find optimal number of clusters.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Range of k values to try
        method: Method for selecting optimal k ('elbow', 'silhouette', 'combined')
        random_state: Random state for reproducibility
        n_init: Number of initializations for k-means
        
    Returns:
        Tuple of (optimal_k, metrics_dict)
    """
    logger.info(f"Finding optimal k in range {list(k_range)} using {method} method...")
    
    metrics = {
        'k_values': list(k_range),
        'inertias': [],
        'silhouette_scores': [],
        'calinski_harabasz_scores': [],
        'davies_bouldin_scores': [],
    }
    
    for k in k_range:
        logger.info(f"Evaluating k={k}...")
        
        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(X)
        
        # Compute metrics
        metrics['inertias'].append(kmeans.inertia_)
        
        if k > 1:
            # Silhouette score (requires k > 1)
            sil_score = silhouette_score(X, labels)
            metrics['silhouette_scores'].append(sil_score)
            
            # Calinski-Harabasz score
            ch_score = calinski_harabasz_score(X, labels)
            metrics['calinski_harabasz_scores'].append(ch_score)
            
            # Davies-Bouldin score (lower is better)
            db_score = davies_bouldin_score(X, labels)
            metrics['davies_bouldin_scores'].append(db_score)
        else:
            metrics['silhouette_scores'].append(np.nan)
            metrics['calinski_harabasz_scores'].append(np.nan)
            metrics['davies_bouldin_scores'].append(np.nan)
    
    # Find optimal k
    if method == 'silhouette':
        # Maximum silhouette score
        valid_scores = [s for s in metrics['silhouette_scores'] if not np.isnan(s)]
        optimal_idx = metrics['silhouette_scores'].index(max(valid_scores))
        optimal_k = metrics['k_values'][optimal_idx]
        
    elif method == 'elbow':
        # Elbow method using second derivative
        inertias = np.array(metrics['inertias'])
        
        # Calculate first and second derivatives
        first_derivative = np.diff(inertias)
        second_derivative = np.diff(first_derivative)
        
        # Find elbow point (maximum second derivative)
        optimal_idx = np.argmax(second_derivative) + 1
        optimal_k = metrics['k_values'][optimal_idx]
        
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
    random_state: int = 42,
    n_init: int = 10,
    use_minibatch: bool = False,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, Any]:
    """Cluster users using K-Means.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        n_init: Number of initializations
        use_minibatch: Whether to use MiniBatchKMeans (faster for large datasets)
        batch_size: Batch size for MiniBatchKMeans
        
    Returns:
        Tuple of (cluster_labels, fitted_model)
    """
    logger.info(f"Clustering {len(X)} users into {n_clusters} clusters...")
    
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
        k_range: range = range(2, 11),
        k_selection_method: str = 'elbow',
        random_state: int = 42,
        n_init: int = 10,
        use_minibatch: bool = False,
    ):
        """Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters (None = auto-select)
            k_range: Range for auto-selection
            k_selection_method: Method for selecting k
            random_state: Random state
            n_init: Number of initializations
            use_minibatch: Whether to use MiniBatchKMeans
        """
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.k_selection_method = k_selection_method
        self.random_state = random_state
        self.n_init = n_init
        self.use_minibatch = use_minibatch
        
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
        # Auto-select k if not specified
        if self.n_clusters is None:
            optimal_k, self.metrics_ = find_optimal_k(
                X,
                k_range=self.k_range,
                method=self.k_selection_method,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            self.n_clusters = optimal_k
        
        # Cluster
        self.labels_, self.model = cluster_users(
            X,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            use_minibatch=self.use_minibatch,
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
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality.
        
        Args:
            X: Feature matrix
            
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
            metrics['silhouette_score'] = silhouette_score(X, self.labels_)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, self.labels_)
        
        return metrics
