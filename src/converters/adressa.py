"""
Adressa dataset converter.

Converts Adressa JSONL files to standard format.
Based on the original 1.adressa_to_ekstra_format.py script.
"""

import re
import uuid
import glob
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .base import BaseConverter
from ..config.settings import DatasetConfig
from ..config.schema import ColumnNames
from ..utils.logging import get_logger, ProgressLogger


class AdressaConverter(BaseConverter):
    """Converter for Adressa dataset (JSONL format).
    
    The Adressa dataset consists of JSONL files (one per day) with user
    interaction events. This converter processes them into standard format.
    """
    
    # Homepage URL patterns
    HOMEPAGE_URLS = {
        'http://adressa.no',
        'http://adressa.no/',
        'https://www.adressa.no/',
        'https://www.adressa.no',
    }
    
    def __init__(self, config: DatasetConfig, chunk_size: int = 10000):
        """Initialize the Adressa converter.
        
        Args:
            config: Dataset configuration
            chunk_size: Number of lines to process per chunk (for memory efficiency)
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        
        # Internal state for tracking articles and sessions
        self._articles: Dict = {}
        self._user_sessions: Dict = {}
        self._user_stats: Dict = {}
    
    @staticmethod
    def extract_category_from_url(url: str) -> str:
        """Extract category from Adressa URL.
        
        Args:
            url: Article URL
            
        Returns:
            Category string (empty for homepage)
        """
        if not url or url in AdressaConverter.HOMEPAGE_URLS:
            return ''
        
        match = re.search(r'https?://[^/]+/([^/]+)', url)
        return match.group(1) if match else ''
    
    def _get_jsonl_files(self) -> list:
        """Get list of JSONL files in the input directory."""
        input_path = Path(self.config.input_path)
        
        if input_path.is_file():
            return [input_path]
        
        return sorted(glob.glob(str(input_path / '*.jsonl')))
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data.
        
        Vectorized implementation for better performance.
        
        Args:
            chunk: DataFrame chunk from JSONL
            
        Returns:
            Processed impressions DataFrame
        """
        # Convert timestamp to datetime
        chunk['datetime'] = pd.to_datetime(chunk['time'], unit='s')
        
        # Sort by user and time
        chunk = chunk.sort_values(['userId', 'time']).reset_index(drop=True)
        
        # Detect homepage views (vectorized)
        chunk['is_homepage'] = chunk['url'].isin(self.HOMEPAGE_URLS)
        
        # Extract category from URL (vectorized using str operations)
        chunk['category_str'] = chunk['url'].str.extract(
            r'https?://[^/]+/([^/]+)', expand=False
        ).fillna('')
        # Clear category for homepage URLs
        chunk.loc[chunk['is_homepage'], 'category_str'] = ''
        
        # Vectorized session assignment
        # Compute time difference within each user
        chunk['time_diff'] = chunk.groupby('userId')['time'].diff()
        
        # Check for sessionStart flag
        session_start_flag = chunk.get('sessionStart', pd.Series(False, index=chunk.index))
        if isinstance(session_start_flag, pd.Series):
            session_start_flag = session_start_flag.fillna(False)
        else:
            session_start_flag = pd.Series(False, index=chunk.index)
        
        # New session when: first row for user (NaN diff), timeout exceeded, or sessionStart flag
        new_session_mask = (
            chunk['time_diff'].isna() |  # First row for this user
            (chunk['time_diff'] > self.config.session_timeout_seconds) |  # Timeout
            session_start_flag  # Explicit session start
        )
        
        # Generate session IDs: cumulative sum of new_session creates groups
        # Then combine with userId to make unique session IDs
        chunk['session_group'] = new_session_mask.groupby(chunk['userId']).cumsum()
        chunk['session_id'] = chunk['userId'].astype(str) + '_' + chunk['session_group'].astype(str)
        
        # Update user sessions state for continuity across chunks
        for user_id in chunk['userId'].unique():
            user_data = chunk[chunk['userId'] == user_id]
            last_time = user_data['time'].iloc[-1]
            last_session = user_data['session_id'].iloc[-1]
            self._user_sessions[user_id] = (last_time, last_session)
        
        # Process article information (vectorized aggregation)
        non_homepage = chunk[~chunk['is_homepage']].copy()
        
        # Fill missing article IDs
        if 'id' in non_homepage.columns:
            non_homepage['article_id_clean'] = non_homepage['id'].fillna('empty').replace('', 'empty')
        else:
            non_homepage['article_id_clean'] = 'empty'
        
        # Aggregate article stats
        if len(non_homepage) > 0:
            # Get first occurrence info for new articles
            article_first = non_homepage.groupby('article_id_clean').first()
            
            # Get aggregated stats
            active_time_col = 'activeTime' if 'activeTime' in non_homepage.columns else None
            
            article_stats = non_homepage.groupby('article_id_clean').agg(
                views=('article_id_clean', 'count'),
                total_reading_time=(active_time_col, 'sum') if active_time_col else ('article_id_clean', 'count'),
            ).reset_index()
            
            # Merge and update articles dict
            for _, row in article_stats.iterrows():
                article_id = row['article_id_clean']
                if article_id not in self._articles:
                    first_row = article_first.loc[article_id]
                    self._articles[article_id] = {
                        'article_id': article_id,
                        'url': first_row.get('url', ''),
                        'time_published': first_row.get('publishtime'),
                        'category_str': first_row.get('category_str', ''),
                        'title': first_row.get('title', ''),
                        'sentiment_score': 0.5,
                        'views': int(row['views']),
                        'total_reading_time': float(row['total_reading_time']) if active_time_col else 0,
                    }
                else:
                    self._articles[article_id]['views'] += int(row['views'])
                    if active_time_col:
                        self._articles[article_id]['total_reading_time'] += float(row['total_reading_time'])
        
        # Create impressions DataFrame (vectorized)
        article_id_col = chunk['id'].fillna('empty') if 'id' in chunk.columns else pd.Series('empty', index=chunk.index)
        article_ids = np.where(chunk['is_homepage'], 'homepage', article_id_col)
        
        active_time = chunk['activeTime'].fillna(0) if 'activeTime' in chunk.columns else pd.Series(0, index=chunk.index)
        
        impressions = pd.DataFrame({
            'session_id': chunk['session_id'],
            'impression_id': chunk['eventId'].astype(str),
            'article_id': article_ids,
            'user_id': chunk['userId'],
            'impression_time': chunk['time'] * 1000,  # Convert to milliseconds
            'read_time': active_time,
        })
        
        # Track user subscription status (vectorized)
        chunk['is_subscriber'] = chunk['url'].str.contains('/pluss/', na=False)
        subscriber_users = chunk[chunk['is_subscriber']]['userId'].unique()
        
        for user_id in chunk['userId'].unique():
            if user_id not in self._user_stats:
                self._user_stats[user_id] = {'is_subscriber': user_id in subscriber_users}
            elif user_id in subscriber_users:
                self._user_stats[user_id]['is_subscriber'] = True
        
        return impressions
    
    def convert_articles(self) -> pd.DataFrame:
        """Convert Adressa articles to standard format.
        
        Note: Articles are extracted during impression processing,
        so this should be called after convert_impressions().
        
        Returns:
            Articles DataFrame
        """
        if not self._articles:
            self.logger.warning("No articles found. Run convert_impressions() first.")
            return pd.DataFrame()
        
        articles_df = pd.DataFrame(list(self._articles.values()))
        
        # Convert publish time to datetime
        if 'time_published' in articles_df.columns:
            articles_df['time_published'] = pd.to_datetime(
                articles_df['time_published'], errors='coerce'
            )
        
        # Ensure required columns exist
        if 'sentiment_score' not in articles_df.columns:
            articles_df['sentiment_score'] = 0.5
        
        return articles_df
    
    def convert_impressions(self) -> pd.DataFrame:
        """Convert Adressa impressions to standard format.
        
        Returns:
            Impressions DataFrame
        """
        jsonl_files = self._get_jsonl_files()
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.config.input_path}")
        
        self.logger.info(f"Processing {len(jsonl_files)} JSONL files...")
        
        all_impressions = []
        
        for file_idx, file_path in enumerate(jsonl_files, 1):
            self.logger.info(f"Processing file {file_idx}/{len(jsonl_files)}: {Path(file_path).name}")
            
            # Count total lines for progress tracking
            with open(file_path, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
            
            for chunk_idx, chunk in enumerate(
                pd.read_json(file_path, lines=True, chunksize=self.chunk_size), 1
            ):
                impressions = self._process_chunk(chunk)
                all_impressions.append(impressions)
                
                if chunk_idx % 10 == 0 or chunk_idx == total_chunks:
                    progress = (chunk_idx / total_chunks) * 100
                    self.logger.info(f"  Chunk {chunk_idx}/{total_chunks} ({progress:.1f}%)")
                
                # Memory cleanup
                del chunk
                gc.collect()
        
        # Combine all impressions
        self.logger.info("Combining all impressions...")
        impressions_df = pd.concat(all_impressions, ignore_index=True)
        
        # Add is_subscriber column
        impressions_df['is_subscriber'] = impressions_df['user_id'].map(
            lambda x: self._user_stats.get(x, {}).get('is_subscriber', False)
        )
        
        # Replace 'empty' article_id with None for filtering
        impressions_df.loc[impressions_df['article_id'] == 'empty', 'article_id'] = None
        
        return impressions_df
    
    def convert_all(
        self,
        output_dir: Optional[str] = None,
        validate: bool = True,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert all Adressa data.
        
        Overrides base method to ensure impressions are processed first
        (which populates the articles dictionary).
        """
        output_dir = output_dir or self.config.output_path
        
        self.logger.info(f"Starting Adressa conversion")
        
        # Reset state
        self._articles = {}
        self._user_sessions = {}
        self._user_stats = {}
        
        # Convert impressions first (this populates articles)
        self.logger.info("Converting impressions...")
        impressions_df = self.convert_impressions()
        impressions_df = self.apply_defaults(impressions_df, "impressions")
        
        if validate:
            result = self.validate_impressions(impressions_df)
            self.logger.info(f"Impressions validation: {result}")
        
        # Now convert articles
        self.logger.info("Converting articles...")
        articles_df = self.convert_articles()
        articles_df = self.apply_defaults(articles_df, "articles")
        
        if validate:
            result = self.validate_articles(articles_df)
            self.logger.info(f"Articles validation: {result}")
        
        # Save if requested
        if save and output_dir:
            from ..utils.io import ensure_dir
            ensure_dir(output_dir)
            
            articles_path = Path(output_dir) / "articles.parquet"
            impressions_path = Path(output_dir) / "behaviors.parquet"
            
            self.save(articles_df, str(articles_path))
            self.save(impressions_df, str(impressions_path))
            
            self.logger.info(f"Saved articles to {articles_path}")
            self.logger.info(f"Saved impressions to {impressions_path}")
            
            # Save samples
            articles_df.head(2000).to_csv(
                Path(output_dir) / "articles_sample.csv", index=False
            )
            impressions_df.head(2000).to_csv(
                Path(output_dir) / "behaviors_sample.csv", index=False
            )
        
        self._log_summary(articles_df, impressions_df)
        
        return articles_df, impressions_df
