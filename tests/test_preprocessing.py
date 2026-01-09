"""Tests for the preprocessing module."""

import pytest
import pandas as pd
import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.validators import (
    validate_articles,
    validate_impressions,
    validate_interactions,
    DataValidator,
)
from src.preprocessing.cleaners import (
    remove_empty_articles,
    remove_invalid_sessions,
    remove_outlier_users,
    DataCleaner,
)
from src.preprocessing.transformers import (
    behaviors_to_interactions,
    articles_to_content,
    add_time_features,
)


class TestValidators:
    """Test validation functions."""
    
    def test_validate_articles_valid(self):
        """Test validation of valid articles DataFrame."""
        df = pd.DataFrame({
            'article_id': ['a1', 'a2', 'a3'],
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'category_str': ['news', 'sport', 'news'],
        })
        
        result = validate_articles(df)
        assert result.is_valid
    
    def test_validate_articles_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
        })
        
        result = validate_articles(df)
        # Should have warnings about missing columns
        assert len(result.warnings) > 0 or not result.is_valid
    
    def test_validate_impressions_valid(self):
        """Test validation of valid impressions DataFrame."""
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'article_id': ['a1', 'a2', 'a1'],
            'impression_time': [1000, 1001, 1002],
            'session_id': ['s1', 's1', 's2'],
        })
        
        result = validate_impressions(df)
        assert result.is_valid
    
    def test_validate_interactions_valid(self):
        """Test validation of valid interactions DataFrame."""
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'article_id': ['a1', 'a2', 'a1'],
            'impression_time': [1000, 1001, 1002],
        })
        
        result = validate_interactions(df)
        assert result.is_valid
    
    def test_validate_interactions_null_values(self):
        """Test validation with null values."""
        df = pd.DataFrame({
            'user_id': ['u1', None, 'u2'],
            'article_id': ['a1', 'a2', 'a1'],
            'impression_time': [1000, 1001, 1002],
        })
        
        result = validate_interactions(df)
        assert not result.is_valid


class TestCleaners:
    """Test cleaning functions."""
    
    def test_remove_empty_articles(self):
        """Test removal of empty article entries."""
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'article_id': ['a1', '', 'homepage', 'a2'],
        })
        
        result = remove_empty_articles(df)
        
        assert len(result) == 2
        assert 'a1' in result['article_id'].values
        assert 'a2' in result['article_id'].values
    
    def test_remove_invalid_sessions(self):
        """Test removal of sessions with too few impressions."""
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2'],
            'article_id': ['a1', 'a2', 'a3', 'a1'],
            'session_id': ['s1', 's1', 's1', 's2'],
        })
        
        result = remove_invalid_sessions(df, min_impressions=2)
        
        assert len(result) == 3  # Only s1 has >= 2 impressions
        assert 's2' not in result['session_id'].values
    
    def test_remove_outlier_users(self):
        """Test removal of users with too few/many impressions."""
        df = pd.DataFrame({
            'user_id': ['u1'] * 10 + ['u2'] * 2 + ['u3'] * 5,
            'article_id': [f'a{i}' for i in range(17)],
        })
        
        result = remove_outlier_users(df, min_impressions=3, max_impressions=8)
        
        # u1 has 10 (too many), u2 has 2 (too few), u3 has 5 (ok)
        assert len(result) == 5
        assert result['user_id'].unique().tolist() == ['u3']
    
    def test_data_cleaner_pipeline(self):
        """Test the DataCleaner class."""
        df = pd.DataFrame({
            'user_id': ['u1'] * 10 + ['u2'] * 3,
            'article_id': [f'a{i%5}' for i in range(13)],
            'session_id': ['s1'] * 10 + ['s2'] * 3,
        })
        
        cleaner = DataCleaner(min_impressions_per_user=3)
        result = cleaner.clean_impressions(df)
        
        assert len(result) == 13  # All users have >= 3 impressions
        
        stats = cleaner.get_stats()
        assert 'initial_rows' in stats
        assert 'final_rows' in stats


class TestTransformers:
    """Test transformation functions."""
    
    def test_behaviors_to_interactions(self):
        """Test conversion of behaviors to interactions."""
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'article_id': ['a1', 'homepage', 'a2', 'a3'],
            'impression_time': [1000, 1001, 1002, 1003],
        })
        
        result = behaviors_to_interactions(df)
        
        assert len(result) == 3  # 'homepage' should be removed
        assert 'homepage' not in result['article_id'].values
        assert set(result.columns) == {'user_id', 'article_id', 'impression_time'}
    
    def test_articles_to_content(self):
        """Test creation of article content strings."""
        df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            'title': ['First article', 'Second article'],
            'category_str': ['news', 'sport'],
        })
        
        result = articles_to_content(df)
        
        assert len(result) == 2
        assert 'content' in result.columns
        assert 'news' in result[result['article_id'] == 'a1']['content'].values[0]
    
    def test_add_time_features(self):
        """Test addition of time features."""
        # Create timestamps for different times of day
        import time
        
        base_time = int(time.time())
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'impression_time': [base_time, base_time + 3600],
        })
        
        result = add_time_features(df)
        
        assert 'is_morning' in result.columns
        assert 'is_afternoon' in result.columns
        assert 'is_evening' in result.columns
        assert 'is_night' in result.columns
        assert 'is_weekend' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
