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
    create_session_behavior_features,
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
        np.random.seed(42)
        return pd.DataFrame({
            'user_id': ['u1'] * 5 + ['u2'] * 3 + ['u3'] * 7,
            'article_id': [f'a{i%5}' for i in range(15)],
            'category_str': ['news', 'news', 'sport', 'news', 'sport'] + ['sport'] * 3 + ['tech'] * 7,
            'impression_time': [1000000000 + i * 100 for i in range(15)],  # Sequential times
            'session_id': ['s1'] * 5 + ['s2'] * 3 + ['s3'] * 7,
            'read_time': np.random.randint(10, 300, 15).astype(float),
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
    
    def test_create_session_behavior_features(self, sample_impressions):
        """Test session behavior feature creation."""
        result = create_session_behavior_features(sample_impressions)
        
        assert 'user_id' in result.columns
        assert 'avg_reading_time' in result.columns
        assert 'avg_session_length' in result.columns
        assert 'avg_categories_per_session' in result.columns
        assert 'avg_category_switches' in result.columns
        assert 'avg_session_duration' in result.columns
        assert len(result) == 3  # 3 unique users
        
        # Check that values are non-negative
        for col in ['avg_reading_time', 'avg_session_length', 'avg_session_duration']:
            assert (result[col] >= 0).all()
    
    def test_create_session_behavior_features_category_switches(self, sample_impressions):
        """Test that category switches are computed correctly."""
        result = create_session_behavior_features(sample_impressions)
        
        # User u1 has categories: news, news, sport, news, sport -> 3 switches
        u1_switches = result[result['user_id'] == 'u1']['avg_category_switches'].values[0]
        assert u1_switches == 3.0  # 3 switches in one session, avg = 3
        
        # User u2 has all sport -> 0 switches
        u2_switches = result[result['user_id'] == 'u2']['avg_category_switches'].values[0]
        assert u2_switches == 0.0
        
        # User u3 has all tech -> 0 switches
        u3_switches = result[result['user_id'] == 'u3']['avg_category_switches'].values[0]
        assert u3_switches == 0.0
    
    def test_create_diversity_features_legacy_mode(self, sample_impressions):
        """Test diversity features in legacy mode (only num_categories)."""
        result = create_diversity_features(sample_impressions, legacy_mode=True)
        
        assert 'user_id' in result.columns
        assert 'num_categories' in result.columns
        # Legacy mode should NOT include entropy and gini
        assert 'category_entropy' not in result.columns
        assert 'category_gini' not in result.columns
        assert len(result.columns) == 2  # user_id + num_categories
    
    def test_create_diversity_features_full_mode(self, sample_impressions):
        """Test diversity features in full mode (includes entropy and gini)."""
        result = create_diversity_features(sample_impressions, legacy_mode=False)
        
        assert 'user_id' in result.columns
        assert 'num_categories' in result.columns
        assert 'category_entropy' in result.columns
        assert 'category_gini' in result.columns
        assert len(result.columns) == 4  # user_id + 3 features
    
    def test_create_user_features_legacy_mode(self, sample_impressions):
        """Test user feature creation in legacy mode."""
        features, metadata = create_user_features(
            sample_impressions,
            legacy_mode=True,
        )
        
        assert len(features) == 3  # 3 users
        assert metadata['legacy_mode'] == True
        
        # Legacy mode should NOT have per-category proportions
        feature_names = metadata['feature_columns']
        cat_features = [f for f in feature_names if f.startswith('cat_')]
        assert len(cat_features) == 0, "Legacy mode should not have per-category features"
        
        # Legacy mode should NOT have time-of-day features
        time_features = [f for f in feature_names if f.startswith('time_')]
        assert len(time_features) == 0, "Legacy mode should not have time-of-day features"
        
        # Legacy mode SHOULD have session behavior features
        assert 'avg_reading_time' in feature_names
        assert 'avg_session_length' in feature_names
        assert 'avg_category_switches' in feature_names
        assert 'avg_session_duration' in feature_names
        
        # Legacy mode should have num_categories but NOT entropy/gini
        assert 'num_categories' in feature_names
        assert 'category_entropy' not in feature_names
        assert 'category_gini' not in feature_names
    
    def test_user_feature_extractor_legacy_mode(self, sample_impressions):
        """Test UserFeatureExtractor class in legacy mode."""
        extractor = UserFeatureExtractor(legacy_mode=True)
        features = extractor.fit_transform(sample_impressions)
        
        assert extractor.is_fitted
        assert len(features) == 3
        
        feature_names = extractor.get_feature_names()
        
        # Check legacy mode configuration
        cat_features = [f for f in feature_names if f.startswith('cat_')]
        assert len(cat_features) == 0, "Legacy mode should not have per-category features"
        
        time_features = [f for f in feature_names if f.startswith('time_')]
        assert len(time_features) == 0, "Legacy mode should not have time-of-day features"
        
        # Legacy features should be present
        assert 'avg_category_switches' in feature_names
        assert 'avg_session_duration' in feature_names


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
