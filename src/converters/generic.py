"""
Generic dataset converter.

Converts any CSV/Parquet dataset with configurable column mapping.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import uuid

from .base import BaseConverter
from ..config.settings import DatasetConfig
from ..config.schema import ColumnNames


class GenericConverter(BaseConverter):
    """Generic converter for datasets with configurable column mapping.
    
    This converter handles datasets that already have structured data
    in CSV or Parquet format, allowing flexible column mapping.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        articles_file: str = "articles",
        impressions_file: str = "behaviors",
    ):
        """Initialize the generic converter.
        
        Args:
            config: Dataset configuration with column_mapping
            articles_file: Name of the articles file (without extension)
            impressions_file: Name of the impressions/behaviors file (without extension)
        """
        super().__init__(config)
        self.articles_file = articles_file
        self.impressions_file = impressions_file
    
    def _find_file(self, base_name: str) -> Optional[Path]:
        """Find a file with the given base name.
        
        Args:
            base_name: File name without extension
            
        Returns:
            Path to the found file, or None
        """
        input_path = Path(self.config.input_path)
        
        # Try different extensions
        extensions = ['.parquet', '.csv', '.json']
        
        for ext in extensions:
            # Direct file
            file_path = input_path / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
            
            # In input directory
            if input_path.is_dir():
                for f in input_path.iterdir():
                    if f.stem == base_name:
                        return f
        
        return None
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a data file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded DataFrame
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def convert_articles(self) -> pd.DataFrame:
        """Convert articles to standard format.
        
        Returns:
            Articles DataFrame
        """
        file_path = self._find_file(self.articles_file)
        
        if not file_path:
            raise FileNotFoundError(
                f"Articles file '{self.articles_file}' not found in {self.config.input_path}"
            )
        
        self.logger.info(f"Loading articles from {file_path}")
        
        df = self._load_file(file_path)
        
        self.logger.info(f"Loaded {len(df)} articles")
        self.logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Apply column mapping
        df = self.map_columns(df)
        
        self.logger.info(f"Mapped columns: {df.columns.tolist()}")
        
        # Ensure required columns
        if 'article_id' not in df.columns:
            raise ValueError("No article_id column found after mapping")
        
        df['article_id'] = df['article_id'].astype(str)
        
        # Add default values for missing optional columns
        if 'title' not in df.columns:
            df['title'] = ''
        if 'category_str' not in df.columns:
            df['category_str'] = ''
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.5
        
        return df
    
    def convert_impressions(self) -> pd.DataFrame:
        """Convert impressions to standard format.
        
        Returns:
            Impressions DataFrame
        """
        file_path = self._find_file(self.impressions_file)
        
        if not file_path:
            raise FileNotFoundError(
                f"Impressions file '{self.impressions_file}' not found in {self.config.input_path}"
            )
        
        self.logger.info(f"Loading impressions from {file_path}")
        
        df = self._load_file(file_path)
        
        self.logger.info(f"Loaded {len(df)} impressions")
        self.logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Apply column mapping
        df = self.map_columns(df)
        
        self.logger.info(f"Mapped columns: {df.columns.tolist()}")
        
        # Ensure required columns
        required = ['user_id', 'impression_time']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found after mapping")
        
        df['user_id'] = df['user_id'].astype(str)
        
        if 'article_id' in df.columns:
            df['article_id'] = df['article_id'].astype(str)
        
        # Generate missing columns
        if 'impression_id' not in df.columns:
            df['impression_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            df['impression_id'] = df['impression_id'].astype(str)
        
        if 'session_id' not in df.columns:
            df = self._generate_session_ids(df)
        
        if 'read_time' not in df.columns:
            df['read_time'] = 0.0
        
        if 'is_subscriber' not in df.columns:
            df['is_subscriber'] = False
        if 'is_logged_in' not in df.columns:
            df['is_logged_in'] = False
        
        # Normalize impression_time to milliseconds
        df = self._normalize_timestamp(df, 'impression_time')
        
        return df
    
    def _generate_session_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate session IDs based on user and time gaps.
        
        Args:
            df: Impressions DataFrame
            
        Returns:
            DataFrame with session_id column
        """
        df = df.sort_values(['user_id', 'impression_time']).copy()
        
        # Calculate time differences
        df['time_diff'] = df.groupby('user_id')['impression_time'].diff()
        
        # Determine timeout threshold (convert to same unit as impression_time)
        # Assuming impression_time is in milliseconds
        timeout = self.config.session_timeout_seconds * 1000
        
        # Mark session boundaries
        df['new_session'] = (df['time_diff'] > timeout) | df['time_diff'].isna()
        
        # Generate session IDs
        df['session_id'] = df.groupby('user_id')['new_session'].cumsum().astype(str)
        df['session_id'] = df['user_id'] + '_session_' + df['session_id']
        
        # Clean up
        df = df.drop(columns=['time_diff', 'new_session'])
        
        return df
    
    def _normalize_timestamp(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize timestamp column to milliseconds.
        
        Args:
            df: DataFrame
            col: Column name
            
        Returns:
            DataFrame with normalized timestamp
        """
        if col not in df.columns:
            return df
        
        # Check if already in milliseconds (large values)
        sample_value = df[col].iloc[0] if len(df) > 0 else 0
        
        if pd.isna(sample_value):
            return df
        
        # If datetime, convert to milliseconds
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(np.int64) // 10**6
        elif isinstance(sample_value, str):
            df[col] = pd.to_datetime(df[col], errors='coerce').astype(np.int64) // 10**6
        elif sample_value < 10**12:
            # Likely in seconds, convert to milliseconds
            df[col] = df[col] * 1000
        
        return df
