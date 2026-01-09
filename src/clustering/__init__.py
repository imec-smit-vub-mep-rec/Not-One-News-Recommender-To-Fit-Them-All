"""Clustering module for user segmentation."""

from .feature_engineering import (
    UserFeatureExtractor,
    create_user_features,
    create_category_features,
    create_time_features,
    scale_features,
)
from .clustering import (
    KMeansClusterer,
    find_optimal_k,
    cluster_users,
    assign_cluster_labels,
)
from .visualization import (
    plot_elbow,
    plot_cluster_distribution,
    plot_feature_importance,
    ClusterVisualizer,
)

__all__ = [
    # Feature engineering
    "UserFeatureExtractor",
    "create_user_features",
    "create_category_features",
    "create_time_features",
    "scale_features",
    # Clustering
    "KMeansClusterer",
    "find_optimal_k",
    "cluster_users",
    "assign_cluster_labels",
    # Visualization
    "plot_elbow",
    "plot_cluster_distribution",
    "plot_feature_importance",
    "ClusterVisualizer",
]
