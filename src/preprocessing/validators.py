"""
Data validation utilities for the RICON analysis pipeline.

Provides functions to validate DataFrames against expected schemas
and check data quality.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from ..config.schema import (
    ARTICLES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    INTERACTIONS_SCHEMA,
    USERS_SCHEMA,
    validate_dataframe,
    ValidationResult,
)
from ..utils.logging import get_logger


logger = get_logger("preprocessing.validators")


def validate_articles(df: pd.DataFrame) -> ValidationResult:
    """Validate an articles DataFrame.
    
    Args:
        df: Articles DataFrame
        
    Returns:
        ValidationResult
    """
    result = validate_dataframe(df, ARTICLES_SCHEMA, "Articles")
    
    # Additional checks specific to articles
    if 'article_id' in df.columns:
        # Check for duplicates
        dup_count = df['article_id'].duplicated().sum()
        if dup_count > 0:
            result.add_warning(f"Found {dup_count} duplicate article_ids")
    
    if 'category_str' in df.columns:
        # Check for empty categories
        empty_count = (df['category_str'] == '').sum() + df['category_str'].isna().sum()
        if empty_count > 0:
            result.add_warning(f"Found {empty_count} articles with empty category")
    
    return result


def validate_impressions(df: pd.DataFrame) -> ValidationResult:
    """Validate an impressions/behaviors DataFrame.
    
    Args:
        df: Impressions DataFrame
        
    Returns:
        ValidationResult
    """
    result = validate_dataframe(df, IMPRESSIONS_SCHEMA, "Impressions")
    
    # Additional checks specific to impressions
    if 'user_id' in df.columns:
        # Check user count
        user_count = df['user_id'].nunique()
        if user_count == 0:
            result.add_error("No users found in impressions")
        else:
            logger.info(f"Found {user_count} unique users")
    
    if 'impression_time' in df.columns:
        # Check for valid timestamps (handle both datetime and numeric types)
        if pd.api.types.is_datetime64_any_dtype(df['impression_time']):
            # For datetime types, check for NaT (Not a Time) values
            invalid_times = df['impression_time'].isna().sum()
        else:
            # For numeric timestamps, check for invalid values
            invalid_times = (df['impression_time'] <= 0).sum()
        if invalid_times > 0:
            result.add_warning(f"Found {invalid_times} impressions with invalid timestamp")
    
    if 'session_id' in df.columns:
        # Check session distribution
        session_count = df['session_id'].nunique()
        avg_impressions_per_session = len(df) / session_count if session_count > 0 else 0
        logger.info(f"Found {session_count} sessions, avg {avg_impressions_per_session:.1f} impressions/session")
    
    return result


def validate_interactions(df: pd.DataFrame) -> ValidationResult:
    """Validate an interactions DataFrame (RecPack format).
    
    Args:
        df: Interactions DataFrame
        
    Returns:
        ValidationResult
    """
    result = validate_dataframe(df, INTERACTIONS_SCHEMA, "Interactions")
    
    # Check for RecPack requirements
    required_cols = ['user_id', 'article_id', 'impression_time']
    
    for col in required_cols:
        if col not in df.columns:
            result.add_error(f"Missing required column for RecPack: {col}")
    
    # Check for null values in required columns
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_error(f"Column {col} has {null_count} null values (not allowed for RecPack)")
    
    # Log statistics
    if result.is_valid:
        logger.info(f"Interactions: {len(df)} rows")
        logger.info(f"Unique users: {df['user_id'].nunique()}")
        logger.info(f"Unique items: {df['article_id'].nunique()}")
    
    return result


class DataValidator:
    """Comprehensive data validator for the pipeline.
    
    Validates multiple DataFrames and provides a combined report.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.results: Dict[str, ValidationResult] = {}
    
    def validate_all(
        self,
        articles: Optional[pd.DataFrame] = None,
        impressions: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None,
    ) -> bool:
        """Validate all provided DataFrames.
        
        Args:
            articles: Optional articles DataFrame
            impressions: Optional impressions DataFrame
            interactions: Optional interactions DataFrame
            
        Returns:
            True if all validations passed
        """
        if articles is not None:
            self.results['articles'] = validate_articles(articles)
        
        if impressions is not None:
            self.results['impressions'] = validate_impressions(impressions)
        
        if interactions is not None:
            self.results['interactions'] = validate_interactions(interactions)
        
        return all(r.is_valid for r in self.results.values())
    
    def get_summary(self) -> str:
        """Get a summary of all validation results.
        
        Returns:
            Human-readable summary string
        """
        lines = ["=" * 60, "Validation Summary", "=" * 60]
        
        for name, result in self.results.items():
            status = "✅ PASSED" if result.is_valid else "❌ FAILED"
            lines.append(f"\n{name}: {status}")
            
            if result.errors:
                for error in result.errors:
                    lines.append(f"  ❌ {error}")
            
            if result.warnings:
                for warning in result.warnings:
                    lines.append(f"  ⚠️ {warning}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print the validation summary."""
        print(self.get_summary())


def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
    """Check general data quality metrics.
    
    Args:
        df: DataFrame to check
        name: Name for logging
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'null_counts': {},
        'duplicate_rows': df.duplicated().sum(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    
    # Count nulls per column
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            metrics['null_counts'][col] = null_count
    
    # Log summary
    logger.info(f"Data quality check for {name}:")
    logger.info(f"  Rows: {metrics['rows']}, Columns: {metrics['columns']}")
    logger.info(f"  Memory: {metrics['memory_mb']:.2f} MB")
    logger.info(f"  Duplicate rows: {metrics['duplicate_rows']}")
    
    if metrics['null_counts']:
        logger.info(f"  Columns with nulls: {list(metrics['null_counts'].keys())}")
    
    return metrics
