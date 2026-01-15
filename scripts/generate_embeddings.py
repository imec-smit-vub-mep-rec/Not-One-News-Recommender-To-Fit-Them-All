#!/usr/bin/env python3
"""
Generate article embeddings for content-based recommendations.

This script pre-calculates sentence transformer embeddings for all articles
in a dataset. The embeddings are saved to a parquet file that can be reused
across multiple pipeline runs.

The content format is "{category}: {title}" matching the legacy behavior.

Usage:
    python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large
    python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large --model intfloat/multilingual-e5-base
    python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large --batch-size 64
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, save_dataframe, load_dataframe

logger = get_logger("generate_embeddings")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate article embeddings for content-based recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing articles.parquet",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: {input-dir}/title_category_embeddings.parquet)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Sentence transformer model name (default: intfloat/multilingual-e5-large)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32, use 128 for GPU)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def find_articles_file(input_dir: str) -> Optional[Path]:
    """Find the articles file in the input directory.
    
    Searches for articles.parquet in the input directory and common subdirectories.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        Path to articles file or None if not found
    """
    input_path = Path(input_dir)
    
    # Try direct path
    if (input_path / "articles.parquet").exists():
        return input_path / "articles.parquet"
    
    # Try train subdirectory (common in EB-NeRD)
    if (input_path / "train" / "articles.parquet").exists():
        return input_path / "train" / "articles.parquet"
    
    # Try validation subdirectory
    if (input_path / "validation" / "articles.parquet").exists():
        return input_path / "validation" / "articles.parquet"
    
    return None


def extract_articles_from_adressa(input_dir: str):
    """Extract articles from Adressa JSONL files on the fly.
    
    Args:
        input_dir: Input directory containing JSONL files
        
    Returns:
        DataFrame with articles
    """
    from src.converters.adressa import AdressaConverter
    from src.config.settings import DatasetConfig
    import glob
    
    # Check if there are JSONL files
    jsonl_files = glob.glob(str(Path(input_dir) / "*.jsonl"))
    if not jsonl_files:
        return None
        
    logger.info(f"Found {len(jsonl_files)} JSONL files. Extracting articles using AdressaConverter...")
    
    # Create temp config
    config = DatasetConfig(
        name="adressa",
        input_path=input_dir,
        format="jsonl"
    )
    
    # Run conversion
    converter = AdressaConverter(config)
    
    # Adressa converter builds article map during impression processing
    logger.info("Scanning impressions to build article catalog (this may take a while)...")
    converter.convert_impressions() 
    
    # Get articles
    articles_df = converter.convert_articles()
    
    if len(articles_df) == 0:
        logger.warning("No articles extracted from Adressa data")
        return None
        
    logger.info(f"Extracted {len(articles_df)} articles from raw data")
    return articles_df


def create_content_strings(articles_df, category_col: str = 'category_str', title_col: str = 'title'):
    """Create content strings in the format '{category}: {title}'.
    
    This matches the legacy format used for content-based recommendations.
    
    Args:
        articles_df: Articles DataFrame
        category_col: Name of category column
        title_col: Name of title column
        
    Returns:
        Series of content strings
    """
    import pandas as pd
    
    # Check for category column aliases
    if category_col not in articles_df.columns:
        if 'category' in articles_df.columns:
            logger.info(f"Column '{category_col}' not found, using 'category' instead")
            category_col = 'category'
        else:
            logger.warning(f"Column '{category_col}' not found, using empty string")
            articles_df = articles_df.copy()
            articles_df[category_col] = ''
    
    if title_col not in articles_df.columns:
        raise ValueError(f"Column '{title_col}' not found in articles DataFrame")
    
    # Fill NA values and ensure string type
    categories = articles_df[category_col].fillna('').astype(str)
    titles = articles_df[title_col].fillna('').astype(str)
    
    # Create content strings: "query: {category}: {title}"
    # The "query: " prefix is important for e5 models and matches legacy Adressa format
    content = "query: " + categories + ": " + titles
    
    return content


def generate_embeddings(
    articles_df,
    model_name: str = "intfloat/multilingual-e5-large",
    batch_size: int = 32,
    show_progress: bool = True,
):
    """Generate embeddings for all articles.
    
    Args:
        articles_df: Articles DataFrame with article_id, category_str, and title
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
        
    Returns:
        DataFrame with article_id and embedding columns
    """
    import pandas as pd
    import numpy as np
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding generation. "
            "Install with: pip install sentence-transformers"
        )
    
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Check if GPU is available
    device = model.device
    logger.info(f"Using device: {device}")
    
    # Adjust batch size for GPU
    if device.type == 'cuda' and batch_size < 64:
        batch_size = 128
        logger.info(f"Increased batch size to {batch_size} for GPU")
    
    # Create content strings
    logger.info("Creating content strings...")
    content_strings = create_content_strings(articles_df)
    
    # Filter out empty content
    valid_mask = content_strings.str.len() > len("query: : ")  # More than just the template
    valid_articles = articles_df[valid_mask].copy()
    valid_content = content_strings[valid_mask].tolist()
    
    logger.info(f"Encoding {len(valid_content)} articles (skipped {(~valid_mask).sum()} with empty content)")
    
    # Generate embeddings
    embeddings = model.encode(
        valid_content,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Create result DataFrame
    result = pd.DataFrame({
        'article_id': valid_articles['article_id'].values,
        'embedding': list(embeddings),
    })
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("=" * 60)
    logger.info("Article Embedding Generation")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Find articles file
    articles_path = find_articles_file(args.input_dir)
    articles_df = None
    
    if articles_path is not None:
        logger.info(f"Found articles file: {articles_path}")
        
        # Load articles
        logger.info("Loading articles...")
        articles_df = load_dataframe(str(articles_path))
        logger.info(f"Loaded {len(articles_df)} articles")
        
    else:
        logger.info(f"Could not find articles.parquet in {args.input_dir}")
        logger.info("Checking for raw Adressa data...")
        
        # Try to extract from Adressa raw files
        articles_df = extract_articles_from_adressa(args.input_dir)
        
        if articles_df is None:
            logger.error(f"Could not find articles.parquet or valid Adressa JSONL files in {args.input_dir}")
            logger.error("Looked in: root, train/, validation/ subdirectories")
            sys.exit(1)
            
        # Optional: Save extracted articles to avoid re-extraction
        try:
            output_articles_path = Path(args.input_dir) / "articles.parquet"
            if not output_articles_path.exists():
                logger.info(f"Saving extracted articles to {output_articles_path} for future use...")
                save_dataframe(articles_df, str(output_articles_path))
        except Exception as e:
            logger.warning(f"Could not save articles.parquet: {e}")
    
    # Check required columns
    if 'article_id' not in articles_df.columns:
        logger.error("Missing required column: article_id")
        sys.exit(1)
    
    if 'title' not in articles_df.columns:
        logger.error("Missing required column: title")
        sys.exit(1)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings_df = generate_embeddings(
        articles_df,
        model_name=args.model,
        batch_size=args.batch_size,
        show_progress=True,
    )
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path(args.input_dir) / "title_category_embeddings.parquet"
    
    # Save embeddings
    logger.info(f"Saving embeddings to {output_path}")
    save_dataframe(embeddings_df, str(output_path))
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Embedding Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Articles processed: {len(embeddings_df)}")
    logger.info(f"Embedding dimension: {len(embeddings_df['embedding'].iloc[0])}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
