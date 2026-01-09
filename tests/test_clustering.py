"""Tests for the clustering module."""

import pytest
import pandas as pd
import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering.feature_engineering import (
    create_category_features,
    create_time_features,
    create_activity_features,
    create_diversity_features,
    scale_features,
    create_user_features,
    UserFeatureExtractor,
)
from src.clustering.clustering import (
    find_optimal_k,
    cluster_users,
    assign_cluster_labels,
    get_cluster_statistics,
    KMeansClusterer,
)


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    @pytest.fixture
    def sample_impressions(self):
        """Create sample impressions DataFrame."""
        return pd.DataFrame({
            'user_id': ['u1'] * 5 + ['u2'] * 3 + ['u3'] * 7,
            'article_id': [f'a{i%5}' for i in range(15)],
            'category_str': ['news'] * 5 + ['sport'] * 3 + ['tech'] * 7,
            'impression_time': np.random.randint(1000000000, 1700000000, 15),
            'session_id': ['s1'] * 5 + ['s2'] * 3 + ['s3'] * 7,
        })
    
    def test_create_category_features(self, sample_impressions):
        """Test category feature creation."""
        result = create_category_features(sample_impressions)
        
        assert 'user_id' in result.columns
        assert any(col.startswith('cat_') for col in result.columns)
        assert len(result) == 3  # 3 unique users
    
    def test_create_time_features(self, sample_impressions):
        """Test time feature creation."""
        result = create_time_features(sample_impressions)
        
        assert 'user_id' in result.columns
        assert 'time_morning' in result.columns
        assert 'time_afternoon' in result.columns
        assert 'time_weekend' in result.columns
        assert len(result) == 3  # 3 unique users
    
    def test_create_activity_features(self, sample_impressions):
        """Test activity feature creation."""
        result = create_activity_features(sample_impressions)
        
        assert 'user_id' in result.columns
        assert 'total_impressions' in result.columns
        assert 'unique_articles' in result.columns
        
        # Check counts are correct
        u1_row = result[result['user_id'] == 'u1'].iloc[0]
        assert u1_row['total_impressions'] == 5
    
    def test_create_diversity_features(self, sample_impressions):
        """Test diversity feature creation."""
        result = create_diversity_features(sample_impressions)
        
        assert 'user_id' in result.columns
        assert 'category_entropy' in result.columns
        assert 'num_categories' in result.columns
    
    def test_scale_features(self, sample_impressions):
        """Test feature scaling."""
        features = create_activity_features(sample_impressions)
        scaled, scaler = scale_features(features)
        
        # Check that features are scaled (zero mean, unit variance)
        feature_cols = [c for c in scaled.columns if c != 'user_id']
        for col in feature_cols:
            assert abs(scaled[col].mean()) < 0.1  # Close to zero
    
    def test_create_user_features_all(self, sample_impressions):
        """Test full user feature creation."""
        features, metadata = create_user_features(sample_impressions)
        
        assert len(features) == 3  # 3 users
        assert 'feature_columns' in metadata
        assert 'n_features' in metadata
        assert metadata['n_features'] > 0
    
    def test_user_feature_extractor(self, sample_impressions):
        """Test UserFeatureExtractor class."""
        extractor = UserFeatureExtractor()
        features = extractor.fit_transform(sample_impressions)
        
        assert extractor.is_fitted
        assert len(features) == 3
        
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 0
        
        X = extractor.get_feature_matrix(features)
        assert X.shape[0] == 3
        assert X.shape[1] == len(feature_names)


class TestClustering:
    """Test clustering functions."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature matrix."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 5) + [0, 0, 0, 0, 0]
        cluster2 = np.random.randn(20, 5) + [3, 3, 0, 0, 0]
        cluster3 = np.random.randn(20, 5) + [0, 0, 3, 3, 0]
        
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_find_optimal_k_elbow(self, sample_features):
        """Test optimal k finding with elbow method."""
        optimal_k, metrics = find_optimal_k(
            sample_features,
            k_range=range(2, 6),
            method='elbow',
        )
        
        assert 2 <= optimal_k <= 5
        assert 'inertias' in metrics
        assert 'silhouette_scores' in metrics
        assert len(metrics['inertias']) == 4  # k=2,3,4,5
    
    def test_find_optimal_k_silhouette(self, sample_features):
        """Test optimal k finding with silhouette method."""
        optimal_k, metrics = find_optimal_k(
            sample_features,
            k_range=range(2, 6),
            method='silhouette',
        )
        
        assert 2 <= optimal_k <= 5
        # With 3 distinct clusters, silhouette should prefer k=3
        assert optimal_k == 3 or abs(optimal_k - 3) <= 1
    
    def test_cluster_users(self, sample_features):
        """Test user clustering."""
        labels, model = cluster_users(sample_features, n_clusters=3)
        
        assert len(labels) == len(sample_features)
        assert len(np.unique(labels)) == 3
        assert model is not None
    
    def test_assign_cluster_labels(self, sample_features):
        """Test cluster label assignment."""
        features_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in range(60)],
            'f1': sample_features[:, 0],
            'f2': sample_features[:, 1],
        })
        
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        result = assign_cluster_labels(features_df, labels)
        
        assert 'cluster_id' in result.columns
        assert len(result) == 60
    
    def test_get_cluster_statistics(self, sample_features):
        """Test cluster statistics computation."""
        features_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in range(60)],
            'f1': sample_features[:, 0],
            'f2': sample_features[:, 1],
        })
        
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        stats = get_cluster_statistics(features_df, labels, ['f1', 'f2'])
        
        assert len(stats) == 3  # 3 clusters
        assert 'cluster_id' in stats.columns
        assert 'size' in stats.columns
        assert 'f1_mean' in stats.columns
    
    def test_kmeans_clusterer_class(self, sample_features):
        """Test KMeansClusterer class."""
        clusterer = KMeansClusterer(n_clusters=3)
        labels = clusterer.fit_predict(sample_features)
        
        assert clusterer.is_fitted
        assert len(labels) == 60
        assert clusterer.n_clusters == 3
        
        # Test evaluation
        metrics = clusterer.evaluate(sample_features)
        assert 'silhouette_score' in metrics
    
    def test_kmeans_clusterer_auto_k(self, sample_features):
        """Test KMeansClusterer with automatic k selection."""
        clusterer = KMeansClusterer(
            n_clusters=None,
            k_range=range(2, 6),
            k_selection_method='silhouette',
        )
        
        labels = clusterer.fit_predict(sample_features)
        
        assert clusterer.is_fitted
        assert 2 <= clusterer.n_clusters <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
