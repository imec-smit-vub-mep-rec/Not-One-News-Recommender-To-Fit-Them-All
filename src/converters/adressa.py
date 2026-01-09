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
        
        Args:
            chunk: DataFrame chunk from JSONL
            
        Returns:
            Processed impressions DataFrame
        """
        # Convert timestamp to datetime
        chunk['datetime'] = pd.to_datetime(chunk['time'], unit='s')
        
        # Sort by user and time
        chunk = chunk.sort_values(['userId', 'time'])
        
        # Detect homepage views
        chunk['is_homepage'] = chunk['url'].isin(self.HOMEPAGE_URLS)
        
        # Extract category from URL
        chunk['category_str'] = chunk['url'].apply(self.extract_category_from_url)
        
        # Process sessions for each user
        chunk['session_id'] = None
        
        for user_id, user_chunk in chunk.groupby('userId'):
            last_time, current_session = self._user_sessions.get(
                user_id, (None, None)
            )
            
            for idx, row in user_chunk.iterrows():
                # Check if new session needed
                if (row.get('sessionStart', False) or
                    last_time is None or
                    (row['time'] - last_time) > self.config.session_timeout_seconds):
                    current_session = str(uuid.uuid4())
                
                chunk.at[idx, 'session_id'] = current_session
                last_time = row['time']
            
            self._user_sessions[user_id] = (last_time, current_session)
        
        # Process article information
        non_homepage = chunk[~chunk['is_homepage']]
        
        for _, row in non_homepage.iterrows():
            article_id = row.get('id', '')
            if not article_id or pd.isna(article_id):
                article_id = "empty"
            
            if article_id not in self._articles:
                self._articles[article_id] = {
                    'article_id': article_id,
                    'url': row.get('url', ''),
                    'time_published': row.get('publishtime'),
                    'category_str': row['category_str'],
                    'title': row.get('title', ''),
                    'sentiment_score': 0.5,
                    'views': 1,
                    'total_reading_time': row.get('activeTime', 0) or 0,
                }
            else:
                self._articles[article_id]['views'] += 1
                self._articles[article_id]['total_reading_time'] += row.get('activeTime', 0) or 0
        
        # Create impressions DataFrame
        impressions = pd.DataFrame({
            'session_id': chunk['session_id'],
            'impression_id': chunk['eventId'].astype(str),
            'article_id': chunk.apply(
                lambda x: 'homepage' if x['is_homepage'] else x.get('id', 'empty'),
                axis=1
            ),
            'user_id': chunk['userId'],
            'impression_time': chunk['time'] * 1000,  # Convert to milliseconds
            'read_time': chunk.get('activeTime', 0).fillna(0),
        })
        
        # Track user subscription status (if URL contains /pluss/)
        for _, row in chunk.iterrows():
            user_id = row['userId']
            is_subscriber = '/pluss/' in str(row.get('url', ''))
            
            if user_id not in self._user_stats:
                self._user_stats[user_id] = {'is_subscriber': is_subscriber}
            elif is_subscriber:
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
