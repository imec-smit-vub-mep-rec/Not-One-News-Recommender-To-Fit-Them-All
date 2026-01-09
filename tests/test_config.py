"""Tests for the config module."""

import pytest
import json
import tempfile
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    DatasetConfig,
    ClusteringConfig,
    EvaluationConfig,
    PipelineConfig,
    load_config,
    save_config,
    PRESET_CONFIGS,
)
from src.config.schema import (
    ARTICLES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    INTERACTIONS_SCHEMA,
    validate_dataframe,
    apply_schema_defaults,
    ValidationResult,
)


class TestDataclasses:
    """Test configuration dataclasses."""
    
    def test_dataset_config_defaults(self):
        """Test DatasetConfig default values."""
        config = DatasetConfig(name="test")
        
        assert config.name == "test"
        assert config.type == "generic"
        assert config.input_dir is None
    
    def test_clustering_config_defaults(self):
        """Test ClusteringConfig default values."""
        config = ClusteringConfig()
        
        assert config.n_clusters is None  # Auto-detect
        assert config.k_selection_method == "elbow"
        assert config.random_state == 42
    
    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()
        
        assert "Popularity" in config.algorithms
        assert 10 in config.k_values
        assert 20 in config.k_values
    
    def test_pipeline_config(self):
        """Test PipelineConfig creation."""
        dataset = DatasetConfig(name="test")
        clustering = ClusteringConfig(n_clusters=5)
        evaluation = EvaluationConfig(algorithms=["Popularity"])
        
        config = PipelineConfig(
            dataset=dataset,
            clustering=clustering,
            evaluation=evaluation,
        )
        
        assert config.dataset.name == "test"
        assert config.clustering.n_clusters == 5
        assert config.evaluation.algorithms == ["Popularity"]


class TestConfigIO:
    """Test config save/load functions."""
    
    def test_save_load_config(self):
        """Test saving and loading config."""
        config = PipelineConfig(
            dataset=DatasetConfig(name="test", type="adressa"),
            clustering=ClusteringConfig(n_clusters=3),
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            
            loaded = load_config(temp_path)
            
            assert loaded.dataset.name == "test"
            assert loaded.dataset.type == "adressa"
            assert loaded.clustering.n_clusters == 3
        finally:
            Path(temp_path).unlink()
    
    def test_preset_configs_exist(self):
        """Test that preset configs are available."""
        assert "adressa" in PRESET_CONFIGS
        assert "ebnerd" in PRESET_CONFIGS
        
        adressa_config = PRESET_CONFIGS["adressa"]
        assert adressa_config.name == "adressa"


class TestSchema:
    """Test schema validation."""
    
    def test_articles_schema(self):
        """Test articles schema definition."""
        assert 'article_id' in ARTICLES_SCHEMA
        assert 'title' in ARTICLES_SCHEMA
        assert 'category_str' in ARTICLES_SCHEMA
    
    def test_impressions_schema(self):
        """Test impressions schema definition."""
        assert 'user_id' in IMPRESSIONS_SCHEMA
        assert 'article_id' in IMPRESSIONS_SCHEMA
        assert 'impression_time' in IMPRESSIONS_SCHEMA
    
    def test_interactions_schema(self):
        """Test interactions schema definition."""
        assert 'user_id' in INTERACTIONS_SCHEMA
        assert 'article_id' in INTERACTIONS_SCHEMA
        assert 'impression_time' in INTERACTIONS_SCHEMA
    
    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            'title': ['Title 1', 'Title 2'],
            'category_str': ['news', 'sport'],
        })
        
        result = validate_dataframe(df, ARTICLES_SCHEMA, "Articles")
        assert result.is_valid
    
    def test_validate_dataframe_missing_required(self):
        """Test validation with missing required column."""
        import pandas as pd
        
        df = pd.DataFrame({
            'title': ['Title 1', 'Title 2'],
            # Missing article_id
        })
        
        result = validate_dataframe(df, ARTICLES_SCHEMA, "Articles")
        assert not result.is_valid
        assert any('article_id' in error for error in result.errors)
    
    def test_validation_result_methods(self):
        """Test ValidationResult class methods."""
        result = ValidationResult()
        
        assert result.is_valid
        
        result.add_warning("Test warning")
        assert result.is_valid  # Warnings don't invalidate
        assert len(result.warnings) == 1
        
        result.add_error("Test error")
        assert not result.is_valid
        assert len(result.errors) == 1
    
    def test_apply_schema_defaults(self):
        """Test applying schema defaults."""
        import pandas as pd
        
        df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            # Missing optional columns
        })
        
        result = apply_schema_defaults(df, ARTICLES_SCHEMA)
        
        # Should have default values for optional columns
        assert 'article_id' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
