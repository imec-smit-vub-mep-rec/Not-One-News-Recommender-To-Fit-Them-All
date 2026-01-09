"""
Visualization utilities for clustering results.

Provides functions for plotting elbow curves, cluster distributions, etc.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from ..utils.logging import get_logger


logger = get_logger("clustering.visualization")


# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Visualization functions will not work.")


def plot_elbow(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[Any]:
    """Plot elbow curve from clustering metrics.
    
    Args:
        metrics: Metrics dict from find_optimal_k()
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None
    
    k_values = metrics['k_values']
    inertias = metrics['inertias']
    optimal_k = metrics.get('optimal_k')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1 = axes[0]
    ax1.plot(k_values, inertias, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    if optimal_k:
        optimal_idx = k_values.index(optimal_k)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax1.scatter([optimal_k], [inertias[optimal_idx]], s=200, c='red', zorder=5)
        ax1.legend()
    
    # Silhouette scores
    ax2 = axes[1]
    sil_scores = metrics.get('silhouette_scores', [])
    
    if sil_scores:
        valid_k = []
        valid_sil = []
        for k, s in zip(k_values, sil_scores):
            if not np.isnan(s):
                valid_k.append(k)
                valid_sil.append(s)
        
        ax2.plot(valid_k, valid_sil, 'g-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs. k', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        if optimal_k and optimal_k in valid_k:
            optimal_idx = valid_k.index(optimal_k)
            ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
            ax2.scatter([optimal_k], [valid_sil[optimal_idx]], s=200, c='red', zorder=5)
            ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved elbow plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = 'Cluster Distribution',
) -> Optional[Any]:
    """Plot distribution of cluster sizes.
    
    Args:
        labels: Cluster labels
        save_path: Optional path to save figure
        show: Whether to display the plot
        title: Plot title
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None
    
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    bars = ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(unique)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add total users annotation
    ax.text(0.98, 0.95, f'Total: {len(labels)} users',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster distribution plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(
    cluster_centers: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = 'Cluster Feature Profiles',
) -> Optional[Any]:
    """Plot feature importance/profiles for each cluster.
    
    Args:
        cluster_centers: DataFrame with cluster centers
        feature_cols: List of feature columns
        save_path: Optional path to save figure
        show: Whether to display the plot
        title: Plot title
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None
    
    if feature_cols is None:
        feature_cols = [c for c in cluster_centers.columns if c != 'cluster_id']
    
    n_clusters = len(cluster_centers)
    n_features = len(feature_cols)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, n_features * 0.5), max(6, n_clusters * 0.8)))
    
    data = cluster_centers[feature_cols].values
    
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Feature Value', rotation=-90, va="bottom", fontsize=10)
    
    # Set ticks
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_clusters))
    
    # Truncate long feature names
    short_names = [f[:20] + '...' if len(f) > 20 else f for f in feature_cols]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=10)
    
    ax.set_title(title, fontsize=14)
    
    # Add value annotations
    for i in range(n_clusters):
        for j in range(n_features):
            value = data[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_category_profiles(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    max_categories: int = 15,
) -> Optional[Any]:
    """Plot category preference profiles for each cluster.
    
    Args:
        features_df: DataFrame with user features including category columns
        labels: Cluster labels
        save_path: Optional path to save figure
        show: Whether to display the plot
        max_categories: Maximum number of categories to show
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return None
    
    # Find category columns
    cat_cols = [c for c in features_df.columns if c.startswith('cat_')]
    
    if not cat_cols:
        logger.warning("No category columns found")
        return None
    
    df = features_df.copy()
    df['cluster_id'] = labels
    
    # Compute mean category preferences per cluster
    cluster_means = df.groupby('cluster_id')[cat_cols].mean()
    
    # Select top categories by variance
    cat_variance = cluster_means.var()
    top_cats = cat_variance.nlargest(max_categories).index.tolist()
    
    cluster_means = cluster_means[top_cats]
    
    # Plot
    n_clusters = len(cluster_means)
    n_cats = len(top_cats)
    
    fig, ax = plt.subplots(figsize=(max(12, n_cats * 0.6), max(6, n_clusters * 0.8)))
    
    x = np.arange(n_cats)
    width = 0.8 / n_clusters
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    
    for i, cluster in enumerate(cluster_means.index):
        offset = (i - n_clusters / 2 + 0.5) * width
        values = cluster_means.loc[cluster].values
        ax.bar(x + offset, values, width, label=f'Cluster {cluster}', color=colors[i])
    
    # Clean category names
    clean_names = [c.replace('cat_', '') for c in top_cats]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Preference', fontsize=12)
    ax.set_title('Category Preferences by Cluster', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved category profiles plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


class ClusterVisualizer:
    """Class-based interface for cluster visualization.
    
    Provides methods for creating various cluster analysis plots.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        show_plots: bool = False,
        dpi: int = 150,
    ):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            show_plots: Whether to display plots interactively
            dpi: Resolution for saved figures
        """
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.dpi = dpi
        
        if output_dir:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_save_path(self, filename: str) -> Optional[str]:
        """Get full save path for a plot."""
        if self.output_dir:
            from pathlib import Path
            return str(Path(self.output_dir) / filename)
        return None
    
    def plot_elbow(self, metrics: Dict[str, Any]) -> Optional[Any]:
        """Plot elbow curve."""
        return plot_elbow(
            metrics,
            save_path=self._get_save_path('elbow_curve.png'),
            show=self.show_plots,
        )
    
    def plot_distribution(
        self,
        labels: np.ndarray,
        title: str = 'Cluster Distribution',
    ) -> Optional[Any]:
        """Plot cluster distribution."""
        return plot_cluster_distribution(
            labels,
            save_path=self._get_save_path('cluster_distribution.png'),
            show=self.show_plots,
            title=title,
        )
    
    def plot_profiles(
        self,
        cluster_centers: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Plot cluster feature profiles."""
        return plot_feature_importance(
            cluster_centers,
            feature_cols,
            save_path=self._get_save_path('cluster_profiles.png'),
            show=self.show_plots,
        )
    
    def plot_categories(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Optional[Any]:
        """Plot category preference profiles."""
        return plot_category_profiles(
            features_df,
            labels,
            save_path=self._get_save_path('category_profiles.png'),
            show=self.show_plots,
        )
    
    def create_all_plots(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        metrics: Optional[Dict[str, Any]] = None,
        cluster_centers: Optional[pd.DataFrame] = None,
    ) -> List[Any]:
        """Create all standard plots.
        
        Args:
            features_df: DataFrame with user features
            labels: Cluster labels
            metrics: Optional k-selection metrics
            cluster_centers: Optional cluster centers DataFrame
            
        Returns:
            List of figure objects
        """
        figures = []
        
        if metrics:
            fig = self.plot_elbow(metrics)
            if fig:
                figures.append(fig)
        
        fig = self.plot_distribution(labels)
        if fig:
            figures.append(fig)
        
        if cluster_centers is not None:
            fig = self.plot_profiles(cluster_centers)
            if fig:
                figures.append(fig)
        
        # Category profiles if category columns exist
        cat_cols = [c for c in features_df.columns if c.startswith('cat_')]
        if cat_cols:
            fig = self.plot_categories(features_df, labels)
            if fig:
                figures.append(fig)
        
        return figures
