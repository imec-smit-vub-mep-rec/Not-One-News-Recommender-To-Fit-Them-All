#!/usr/bin/env python3
"""
Data conversion script.

Converts raw dataset files to the standard RICON format.

Usage:
    python run_conversion.py --dataset adressa --input-dir /path/to/data --output-dir /path/to/output
    python run_conversion.py --dataset ebnerd --input-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, save_dataframe, ensure_dir
from src.converters import AdressaConverter, EBNeRDConverter, GenericConverter


logger = get_logger("conversion")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert dataset to standard RICON format",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adressa", "ebnerd", "generic"],
        help="Dataset type",
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with raw data files",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./converted",
        help="Output directory for converted files",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output file format",
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of rows to sample (for testing)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info(f"Converting {args.dataset} dataset")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Select converter
    if args.dataset == "adressa":
        converter = AdressaConverter(input_dir=args.input_dir)
    elif args.dataset == "ebnerd":
        converter = EBNeRDConverter(input_dir=args.input_dir)
    else:
        logger.error("Generic converter requires column mapping. Use run_full_pipeline.py with a config file.")
        sys.exit(1)
    
    # Convert articles
    logger.info("Converting articles...")
    articles_df = converter.convert_articles()
    
    if args.sample and len(articles_df) > args.sample:
        articles_df = articles_df.sample(n=args.sample, random_state=42)
    
    articles_path = Path(args.output_dir) / f"articles.{args.format}"
    save_dataframe(articles_df, str(articles_path), format=args.format)
    logger.info(f"Saved {len(articles_df)} articles to {articles_path}")
    
    # Convert impressions
    logger.info("Converting impressions...")
    impressions_df = converter.convert_impressions()
    
    if args.sample and len(impressions_df) > args.sample:
        impressions_df = impressions_df.sample(n=args.sample, random_state=42)
    
    impressions_path = Path(args.output_dir) / f"impressions.{args.format}"
    save_dataframe(impressions_df, str(impressions_path), format=args.format)
    logger.info(f"Saved {len(impressions_df)} impressions to {impressions_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Conversion Summary")
    logger.info("=" * 60)
    logger.info(f"Articles: {len(articles_df)} rows")
    logger.info(f"  Columns: {list(articles_df.columns)}")
    logger.info(f"Impressions: {len(impressions_df)} rows")
    logger.info(f"  Columns: {list(impressions_df.columns)}")
    logger.info(f"  Unique users: {impressions_df['user_id'].nunique()}")
    logger.info(f"  Unique articles: {impressions_df['article_id'].nunique()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
