"""
Base converter class for transforming raw datasets to standard format.

All dataset-specific converters inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from ..config.settings import DatasetConfig
from ..config.schema import (
    ARTICLES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    validate_dataframe,
    apply_schema_defaults,
    coerce_dtypes,
    ValidationResult,
)
from ..utils.io import save_dataframe, ensure_dir
from ..utils.logging import get_logger


class BaseConverter(ABC):
    """Abstract base class for dataset converters.
    
    Converts raw dataset files to the standard format defined in schema.py.
    Subclasses must implement convert_articles() and convert_impressions().
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize the converter.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.logger = get_logger(f"converter.{config.name}")
    
    @abstractmethod
    def convert_articles(self) -> pd.DataFrame:
        """Convert raw article data to standard format.
        
        Returns:
            DataFrame with columns matching ARTICLES_SCHEMA
        """
        pass
    
    @abstractmethod
    def convert_impressions(self) -> pd.DataFrame:
        """Convert raw impression/behavior data to standard format.
        
        Returns:
            DataFrame with columns matching IMPRESSIONS_SCHEMA
        """
        pass
    
    def validate_articles(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the articles DataFrame against the schema.
        
        Args:
            df: Articles DataFrame
            
        Returns:
            ValidationResult
        """
        return validate_dataframe(df, ARTICLES_SCHEMA, "Articles")
    
    def validate_impressions(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the impressions DataFrame against the schema.
        
        Args:
            df: Impressions DataFrame
            
        Returns:
            ValidationResult
        """
        return validate_dataframe(df, IMPRESSIONS_SCHEMA, "Impressions")
    
    def apply_defaults(
        self, 
        df: pd.DataFrame, 
        schema_type: str = "articles"
    ) -> pd.DataFrame:
        """Apply default values from schema.
        
        Args:
            df: DataFrame to process
            schema_type: 'articles' or 'impressions'
            
        Returns:
            DataFrame with defaults applied
        """
        schema = ARTICLES_SCHEMA if schema_type == "articles" else IMPRESSIONS_SCHEMA
        return apply_schema_defaults(df, schema)
    
    def coerce_types(
        self,
        df: pd.DataFrame,
        schema_type: str = "articles"
    ) -> pd.DataFrame:
        """Coerce DataFrame columns to schema types.
        
        Args:
            df: DataFrame to process
            schema_type: 'articles' or 'impressions'
            
        Returns:
            DataFrame with coerced types
        """
        schema = ARTICLES_SCHEMA if schema_type == "articles" else IMPRESSIONS_SCHEMA
        return coerce_dtypes(df, schema)
    
    def save(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = "parquet"
    ) -> Path:
        """Save a DataFrame to a file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save to
            format: Output format ('parquet', 'csv')
            
        Returns:
            Path to saved file
        """
        return save_dataframe(df, output_path, format=format)
    
    def convert_all(
        self,
        output_dir: Optional[str] = None,
        validate: bool = True,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert all data (articles and impressions).
        
        Args:
            output_dir: Directory to save output files (uses config if None)
            validate: Whether to validate output
            save: Whether to save output files
            
        Returns:
            Tuple of (articles_df, impressions_df)
        """
        output_dir = output_dir or self.config.output_path
        
        self.logger.info(f"Starting conversion for {self.config.name}")
        
        # Convert articles
        self.logger.info("Converting articles...")
        articles_df = self.convert_articles()
        articles_df = self.apply_defaults(articles_df, "articles")
        articles_df = self.coerce_types(articles_df, "articles")
        
        if validate:
            result = self.validate_articles(articles_df)
            self.logger.info(f"Articles validation: {result}")
            if not result.is_valid:
                self.logger.warning(result.summary())
        
        # Convert impressions
        self.logger.info("Converting impressions...")
        impressions_df = self.convert_impressions()
        impressions_df = self.apply_defaults(impressions_df, "impressions")
        impressions_df = self.coerce_types(impressions_df, "impressions")
        
        if validate:
            result = self.validate_impressions(impressions_df)
            self.logger.info(f"Impressions validation: {result}")
            if not result.is_valid:
                self.logger.warning(result.summary())
        
        # Save if requested
        if save and output_dir:
            ensure_dir(output_dir)
            
            articles_path = Path(output_dir) / "articles.parquet"
            impressions_path = Path(output_dir) / "behaviors.parquet"
            
            self.save(articles_df, str(articles_path))
            self.save(impressions_df, str(impressions_path))
            
            self.logger.info(f"Saved articles to {articles_path}")
            self.logger.info(f"Saved impressions to {impressions_path}")
            
            # Save sample CSVs for debugging
            sample_size = min(2000, len(articles_df))
            articles_df.head(sample_size).to_csv(
                Path(output_dir) / "articles_sample.csv", index=False
            )
            impressions_df.head(sample_size).to_csv(
                Path(output_dir) / "behaviors_sample.csv", index=False
            )
        
        # Log summary statistics
        self._log_summary(articles_df, impressions_df)
        
        return articles_df, impressions_df
    
    def _log_summary(self, articles_df: pd.DataFrame, impressions_df: pd.DataFrame):
        """Log summary statistics for converted data."""
        self.logger.info("=" * 50)
        self.logger.info("Conversion Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Articles: {len(articles_df)}")
        self.logger.info(f"Impressions: {len(impressions_df)}")
        self.logger.info(f"Unique users: {impressions_df['user_id'].nunique()}")
        self.logger.info(f"Unique articles in impressions: {impressions_df['article_id'].nunique()}")
        
        if 'session_id' in impressions_df.columns:
            self.logger.info(f"Unique sessions: {impressions_df['session_id'].nunique()}")
        
        if 'category_str' in articles_df.columns:
            self.logger.info(f"Unique categories: {articles_df['category_str'].nunique()}")
    
    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to the configuration mapping.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with renamed columns
        """
        if not self.config.column_mapping:
            return df
        
        # Create reverse mapping (standard name -> source name)
        reverse_mapping = {v: k for k, v in self.config.column_mapping.items()}
        
        # Only rename columns that exist in the DataFrame
        rename_map = {}
        for source_col, target_col in self.config.column_mapping.items():
            if source_col in df.columns:
                rename_map[source_col] = target_col
        
        return df.rename(columns=rename_map)
