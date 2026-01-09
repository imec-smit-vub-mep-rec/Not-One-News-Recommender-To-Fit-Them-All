"""
EB-NeRD dataset converter.

Converts EB-NeRD parquet files to standard format.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .base import BaseConverter
from ..config.settings import DatasetConfig
from ..config.schema import ColumnNames


class EBNeRDConverter(BaseConverter):
    """Converter for EB-NeRD dataset (Parquet format).
    
    The EB-NeRD dataset is already in a structured format with
    articles.parquet and behaviors.parquet files.
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize the EB-NeRD converter.
        
        Args:
            config: Dataset configuration
        """
        super().__init__(config)
    
    def _find_file(self, patterns: list) -> Optional[Path]:
        """Find a file matching one of the patterns.
        
        Args:
            patterns: List of filename patterns to try
            
        Returns:
            Path to the found file, or None
        """
        input_path = Path(self.config.input_path)
        
        for pattern in patterns:
            # Check in input directory
            file_path = input_path / pattern
            if file_path.exists():
                return file_path
            
            # Check in subdirectories
            for subdir in input_path.iterdir():
                if subdir.is_dir():
                    file_path = subdir / pattern
                    if file_path.exists():
                        return file_path
        
        return None
    
    def convert_articles(self) -> pd.DataFrame:
        """Convert EB-NeRD articles to standard format.
        
        Returns:
            Articles DataFrame
        """
        articles_path = self._find_file([
            'articles.parquet',
            'articles.csv',
        ])
        
        if not articles_path:
            raise FileNotFoundError(
                f"Articles file not found in {self.config.input_path}"
            )
        
        self.logger.info(f"Loading articles from {articles_path}")
        
        # Load data
        if articles_path.suffix == '.parquet':
            df = pd.read_parquet(articles_path)
        else:
            df = pd.read_csv(articles_path)
        
        self.logger.info(f"Loaded {len(df)} articles")
        self.logger.info(f"Columns: {df.columns.tolist()}")
        
        # Apply column mapping
        df = self.map_columns(df)
        
        # Ensure article_id is string
        if 'article_id' in df.columns:
            df['article_id'] = df['article_id'].astype(str)
        
        # Handle time_published
        if 'published_time' in df.columns and 'time_published' not in df.columns:
            df['time_published'] = df['published_time']
        
        if 'time_published' in df.columns:
            df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
        
        # Ensure required columns exist
        required_cols = ['article_id', 'title', 'category_str']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Missing column {col}, adding empty column")
                df[col] = ''
        
        # Default sentiment score if not present
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.5
        
        return df
    
    def convert_impressions(self) -> pd.DataFrame:
        """Convert EB-NeRD behaviors to standard format.
        
        Returns:
            Impressions DataFrame
        """
        behaviors_path = self._find_file([
            'behaviors.parquet',
            'behaviors.csv',
        ])
        
        if not behaviors_path:
            raise FileNotFoundError(
                f"Behaviors file not found in {self.config.input_path}"
            )
        
        self.logger.info(f"Loading behaviors from {behaviors_path}")
        
        # Load data
        if behaviors_path.suffix == '.parquet':
            df = pd.read_parquet(behaviors_path)
        else:
            df = pd.read_csv(behaviors_path)
        
        self.logger.info(f"Loaded {len(df)} behavior records")
        self.logger.info(f"Columns: {df.columns.tolist()}")
        
        # Apply column mapping
        df = self.map_columns(df)
        
        # Ensure user_id is string
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str)
        
        # Handle article_id conversion properly (float -> int -> str)
        if 'article_id' in df.columns:
            # Keep nulls as NaN, convert non-null floats to int first then string
            mask = df['article_id'].notna()
            if pd.api.types.is_float_dtype(df['article_id']):
                # Convert non-null values: float -> int -> str
                df.loc[mask, 'article_id'] = df.loc[mask, 'article_id'].astype(int).astype(str)
            else:
                df.loc[mask, 'article_id'] = df.loc[mask, 'article_id'].astype(str)
        
        # Handle impression_id
        if 'impression_id' not in df.columns:
            df['impression_id'] = range(len(df))
        df['impression_id'] = df['impression_id'].astype(str)
        
        # Handle impression_time
        if 'impression_time' in df.columns:
            # If it's datetime, convert to milliseconds
            if df['impression_time'].dtype == 'datetime64[ns]':
                df['impression_time'] = df['impression_time'].astype(np.int64) // 10**6
            elif df['impression_time'].dtype == 'object':
                df['impression_time'] = pd.to_datetime(
                    df['impression_time'], errors='coerce'
                ).astype(np.int64) // 10**6
        
        # Handle session_id
        if 'session_id' not in df.columns:
            # Generate session IDs based on user and time gaps
            self.logger.info("Generating session IDs...")
            df = self._generate_session_ids(df)
        
        # Handle is_subscriber (may come from is_sso_user in EB-NeRD)
        if 'is_subscriber' not in df.columns:
            if 'is_sso_user' in df.columns:
                df['is_subscriber'] = df['is_sso_user']
            else:
                df['is_subscriber'] = False
        
        # Handle read_time
        if 'read_time' not in df.columns:
            df['read_time'] = 0.0
        
        return df
    
    def _generate_session_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate session IDs based on user and time gaps.
        
        Args:
            df: Behaviors DataFrame
            
        Returns:
            DataFrame with session_id column added
        """
        df = df.sort_values(['user_id', 'impression_time']).copy()
        
        # Calculate time gaps
        df['time_diff'] = df.groupby('user_id')['impression_time'].diff()
        
        # Timeout in milliseconds
        timeout_ms = self.config.session_timeout_seconds * 1000
        
        # Mark session starts
        df['new_session'] = (
            (df['time_diff'] > timeout_ms) | 
            (df['time_diff'].isna())
        )
        
        # Generate session IDs
        df['session_id'] = df.groupby('user_id')['new_session'].cumsum().astype(str)
        df['session_id'] = df['user_id'] + '_' + df['session_id']
        
        # Cleanup temporary columns
        df = df.drop(columns=['time_diff', 'new_session'])
        
        return df
