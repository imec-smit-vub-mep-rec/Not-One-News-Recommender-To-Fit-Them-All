"""
Data schema definitions for the RICON analysis pipeline.

This module defines the standard column names and data types for all
data formats used in the pipeline, based on general_data_format.md.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class ColumnNames:
    """Standard column names used throughout the pipeline.
    
    These names are the internal standard format. Dataset-specific
    converters map source columns to these standard names.
    """
    # Article columns
    ARTICLE_ID = "article_id"
    TIME_PUBLISHED = "time_published"
    TITLE = "title"
    CATEGORY_STR = "category_str"
    CATEGORIES = "categories"
    ARTICLE_LENGTH = "article_length"
    SENTIMENT_SCORE = "sentiment_score"
    
    # Impression columns
    IMPRESSION_ID = "impression_id"
    START_TIME = "start_time"
    IMPRESSION_TIME = "impression_time"
    ACTIVE_SECONDS = "active_seconds"
    SESSION_ID = "session_id"
    USER_ID = "user_id"
    READ_TIME = "read_time"
    SCROLL_DEPTH = "scroll_depth"
    DEVICE_TYPE = "device_type"
    IS_SUBSCRIBER = "is_subscriber"
    DAY_OF_WEEK = "day_of_week"
    
    # Interaction columns (for RecPack)
    TIMESTAMP = "timestamp"
    
    # User feature columns (output)
    COUNT_SESSIONS = "count_sessions"
    HAS_ACCOUNT = "has_account"
    COUNT_TOTAL_IMPRESSIONS = "count_total_impressions"
    COUNT_TOTAL_HOMEPAGE_IMPRESSIONS = "count_total_homepage_impressions"
    COUNT_TOTAL_ARTICLE_IMPRESSIONS = "count_total_article_impressions"
    COUNT_TOTAL_UNIQUE_CATEGORIES = "count_total_unique_categories"
    COUNT_TOTAL_UNIQUE_ARTICLES = "count_total_unique_articles"
    TOTAL_READING_TIME = "total_reading_time"
    AVG_READING_TIME = "avg_reading_time"
    AVG_SESSION_LENGTH = "avg_session_length"
    AVG_SESSION_DURATION = "avg_session_duration"
    AVG_CATEGORIES_PER_SESSION = "avg_categories_per_session"
    AVG_CATEGORY_SWITCHES = "avg_category_switches"
    AVG_READING_TIME_HOMEPAGE = "avg_reading_time_homepage"
    AVG_READING_TIME_ARTICLES = "avg_reading_time_articles"
    PROPORTION_ARTICLE_TIME = "proportion_article_time"
    PROPORTION_MORNING_IMPRESSIONS = "proportion_morning_impressions"
    PROPORTION_AFTERNOON_IMPRESSIONS = "proportion_afternoon_impressions"
    PROPORTION_EVENING_IMPRESSIONS = "proportion_evening_impressions"
    PROPORTION_NIGHT_IMPRESSIONS = "proportion_night_impressions"
    PROPORTION_MORNING_READING_TIME = "proportion_morning_reading_time"
    PROPORTION_AFTERNOON_READING_TIME = "proportion_afternoon_reading_time"
    PROPORTION_EVENING_READING_TIME = "proportion_evening_reading_time"
    PROPORTION_NIGHT_READING_TIME = "proportion_night_reading_time"
    CLUSTER_ID = "cluster_id"


# Schema definitions with required columns and data types
# Format: {column_name: (dtype, required, default_value)}

ARTICLES_SCHEMA: Dict[str, Tuple[str, bool, Any]] = {
    ColumnNames.ARTICLE_ID: ("string", True, None),
    ColumnNames.TIME_PUBLISHED: ("datetime64[ns]", False, None),
    ColumnNames.TITLE: ("string", True, ""),
    ColumnNames.CATEGORY_STR: ("string", True, ""),
    ColumnNames.CATEGORIES: ("object", False, None),  # List of strings
    ColumnNames.ARTICLE_LENGTH: ("int32", False, 0),
    ColumnNames.SENTIMENT_SCORE: ("float32", False, 0.5),
}

IMPRESSIONS_SCHEMA: Dict[str, Tuple[str, bool, Any]] = {
    ColumnNames.IMPRESSION_ID: ("string", True, None),
    ColumnNames.IMPRESSION_TIME: ("int64", True, None),  # Unix timestamp ms
    ColumnNames.SESSION_ID: ("string", True, None),
    ColumnNames.USER_ID: ("string", True, None),
    ColumnNames.ARTICLE_ID: ("string", False, None),  # None for homepage
    ColumnNames.READ_TIME: ("float32", False, 0.0),
    ColumnNames.IS_SUBSCRIBER: ("boolean", False, False),
    ColumnNames.SCROLL_DEPTH: ("float32", False, None),
    ColumnNames.DEVICE_TYPE: ("string", False, None),
}

INTERACTIONS_SCHEMA: Dict[str, Tuple[str, bool, Any]] = {
    ColumnNames.USER_ID: ("string", True, None),
    ColumnNames.ARTICLE_ID: ("string", True, None),
    ColumnNames.IMPRESSION_TIME: ("int64", True, None),
}

USERS_SCHEMA: Dict[str, Tuple[str, bool, Any]] = {
    ColumnNames.USER_ID: ("string", True, None),
    ColumnNames.COUNT_SESSIONS: ("int32", True, 0),
    ColumnNames.COUNT_TOTAL_IMPRESSIONS: ("int32", True, 0),
    ColumnNames.COUNT_TOTAL_HOMEPAGE_IMPRESSIONS: ("int32", False, 0),
    ColumnNames.COUNT_TOTAL_ARTICLE_IMPRESSIONS: ("int32", False, 0),
    ColumnNames.COUNT_TOTAL_UNIQUE_CATEGORIES: ("int32", False, 0),
    ColumnNames.COUNT_TOTAL_UNIQUE_ARTICLES: ("int32", False, 0),
    ColumnNames.TOTAL_READING_TIME: ("float32", False, 0.0),
    ColumnNames.AVG_READING_TIME: ("float32", False, 0.0),
    ColumnNames.AVG_SESSION_LENGTH: ("float32", False, 0.0),
    ColumnNames.AVG_SESSION_DURATION: ("float32", False, 0.0),
    ColumnNames.AVG_CATEGORIES_PER_SESSION: ("float32", False, 0.0),
    ColumnNames.AVG_CATEGORY_SWITCHES: ("float32", False, 0.0),
    ColumnNames.AVG_READING_TIME_HOMEPAGE: ("float32", False, 0.0),
    ColumnNames.AVG_READING_TIME_ARTICLES: ("float32", False, 0.0),
    ColumnNames.PROPORTION_ARTICLE_TIME: ("float32", False, 0.0),
    ColumnNames.PROPORTION_MORNING_IMPRESSIONS: ("float32", False, 0.0),
    ColumnNames.PROPORTION_AFTERNOON_IMPRESSIONS: ("float32", False, 0.0),
    ColumnNames.PROPORTION_EVENING_IMPRESSIONS: ("float32", False, 0.0),
    ColumnNames.PROPORTION_NIGHT_IMPRESSIONS: ("float32", False, 0.0),
    ColumnNames.IS_SUBSCRIBER: ("boolean", False, False),
    ColumnNames.CLUSTER_ID: ("int32", False, -1),
}


class ValidationResult:
    """Result of a dataframe validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid: bool = True
    
    def add_error(self, message: str):
        """Add an error (validation failure)."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning (non-critical issue)."""
        self.warnings.append(message)
    
    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)}, warnings={len(self.warnings)})"
    
    def summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        lines = [f"Validation: {'PASSED' if self.is_valid else 'FAILED'}"]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️ {warning}")
        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    schema: Dict[str, Tuple[str, bool, Any]],
    schema_name: str = "DataFrame"
) -> ValidationResult:
    """Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Schema definition dict {column: (dtype, required, default)}
        schema_name: Name of the schema for error messages
        
    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()
    
    # Check for required columns
    for col, (dtype, required, default) in schema.items():
        if required and col not in df.columns:
            result.add_error(f"{schema_name}: Missing required column '{col}'")
    
    # Check column data types
    for col in df.columns:
        if col in schema:
            expected_dtype, required, default = schema[col]
            actual_dtype = str(df[col].dtype)
            
            # Type compatibility check
            if not _dtype_compatible(actual_dtype, expected_dtype):
                result.add_warning(
                    f"{schema_name}: Column '{col}' has dtype '{actual_dtype}', "
                    f"expected '{expected_dtype}'"
                )
    
    # Check for null values in required columns
    for col, (dtype, required, default) in schema.items():
        if required and col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_error(
                    f"{schema_name}: Required column '{col}' has {null_count} null values"
                )
    
    # Check for empty dataframe
    if len(df) == 0:
        result.add_warning(f"{schema_name}: DataFrame is empty")
    
    # Check for extra columns (not in schema)
    extra_cols = set(df.columns) - set(schema.keys())
    if extra_cols:
        result.add_warning(
            f"{schema_name}: Extra columns not in schema: {extra_cols}"
        )
    
    return result


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Check if actual dtype is compatible with expected dtype."""
    # Normalize dtype strings
    actual = actual.lower()
    expected = expected.lower()
    
    # Direct match
    if actual == expected:
        return True
    
    # String types
    if expected == "string" and actual in ("object", "string", "str"):
        return True
    
    # Integer types
    if "int" in expected and "int" in actual:
        return True
    
    # Float types
    if "float" in expected and "float" in actual:
        return True
    
    # Datetime types
    if "datetime" in expected and "datetime" in actual:
        return True
    
    # Boolean types
    if expected == "boolean" and actual in ("bool", "boolean"):
        return True
    
    return False


def apply_schema_defaults(
    df: pd.DataFrame,
    schema: Dict[str, Tuple[str, bool, Any]]
) -> pd.DataFrame:
    """Apply default values from schema to a DataFrame.
    
    Args:
        df: DataFrame to process
        schema: Schema definition dict
        
    Returns:
        DataFrame with default values applied
    """
    df = df.copy()
    
    for col, (dtype, required, default) in schema.items():
        if col in df.columns and default is not None:
            df[col] = df[col].fillna(default)
    
    return df


def coerce_dtypes(
    df: pd.DataFrame,
    schema: Dict[str, Tuple[str, bool, Any]]
) -> pd.DataFrame:
    """Coerce DataFrame columns to schema data types.
    
    Args:
        df: DataFrame to process
        schema: Schema definition dict
        
    Returns:
        DataFrame with coerced dtypes
    """
    df = df.copy()
    
    for col, (dtype, required, default) in schema.items():
        if col not in df.columns:
            continue
            
        try:
            if dtype == "string":
                df[col] = df[col].astype(str).replace('nan', '')
            elif dtype == "boolean":
                df[col] = df[col].astype(bool)
            elif "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif "int" in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif "float" in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not coerce column '{col}' to {dtype}: {e}")
    
    return df
