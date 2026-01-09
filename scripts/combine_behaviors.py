#!/usr/bin/env python3
"""
Combine behaviors files from multiple EBNeRD splits.

Combines behaviors.parquet files from subfolders (e.g., train, validation, test)
into a single behaviors.parquet file in the parent folder.

Usage:
    python combine_behaviors.py --input-dir /path/to/ebnerd --splits train validation
    python combine_behaviors.py --input-dir /path/to/ebnerd --splits train validation test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, ensure_dir


logger = get_logger("combine_behaviors")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine behaviors.parquet files from multiple EBNeRD splits",
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Base input directory containing split folders (e.g., train/, validation/)",
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation"],
        help="List of split folder names to combine (default: train validation)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: <input-dir>/behaviors.parquet)",
    )
    
    parser.add_argument(
        "--keep-source-column",
        action="store_true",
        help="Keep _source_split column in output to track origin of each row",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def combine_behaviors(
    input_dir: str,
    splits: list,
    output_file: str = None,
    keep_source_column: bool = False,
) -> pd.DataFrame:
    """Combine behaviors.parquet files from multiple splits.
    
    Args:
        input_dir: Base input directory containing split folders
        splits: List of split folder names (e.g., ['train', 'validation'])
        output_file: Output file path (default: <input_dir>/behaviors.parquet)
        keep_source_column: Whether to keep the _source_split column
        
    Returns:
        Combined DataFrame
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    behaviors_dfs = []
    total_input_users = set()
    
    logger.info("=" * 60)
    logger.info("Loading behaviors from splits")
    logger.info("=" * 60)
    
    for split in splits:
        split_dir = input_path / split
        behaviors_file = split_dir / "behaviors.parquet"
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        if not behaviors_file.exists():
            logger.warning(f"behaviors.parquet not found in {split_dir}")
            continue
        
        logger.info(f"\nLoading: {behaviors_file}")
        df = pd.read_parquet(behaviors_file)
        
        # Get user column (try common names)
        user_col = None
        for col in ['user_id', 'userId', 'user']:
            if col in df.columns:
                user_col = col
                break
        
        # Log statistics for this split
        n_rows = len(df)
        n_users = df[user_col].nunique() if user_col else "N/A"
        
        logger.info(f"  Rows: {n_rows:,}")
        logger.info(f"  Unique users: {n_users:,}" if user_col else "  Unique users: N/A (no user column found)")
        
        if user_col:
            total_input_users.update(df[user_col].unique())
        
        # Add source column for tracking
        df['_source_split'] = split
        behaviors_dfs.append(df)
    
    if not behaviors_dfs:
        raise FileNotFoundError(
            f"No behaviors.parquet files found in splits: {splits}"
        )
    
    # Combine all behaviors
    logger.info("\n" + "=" * 60)
    logger.info("Combining behaviors")
    logger.info("=" * 60)
    
    combined_df = pd.concat(behaviors_dfs, ignore_index=True)
    
    # Get user column for combined df
    user_col = None
    for col in ['user_id', 'userId', 'user']:
        if col in combined_df.columns:
            user_col = col
            break
    
    # Calculate output statistics
    n_output_rows = len(combined_df)
    n_output_users = combined_df[user_col].nunique() if user_col else "N/A"
    
    # Remove source column if not needed
    if not keep_source_column:
        combined_df = combined_df.drop(columns=['_source_split'])
    
    # Determine output path
    if output_file is None:
        output_file = input_path / "behaviors.parquet"
    else:
        output_file = Path(output_file)
    
    # Ensure output directory exists
    ensure_dir(output_file.parent)
    
    # Save combined file
    combined_df.to_parquet(output_file)
    
    # Log final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Input splits: {splits}")
    logger.info(f"Total input rows: {sum(len(df) for df in behaviors_dfs):,}")
    logger.info(f"Total unique users across inputs: {len(total_input_users):,}" if total_input_users else "Total unique users across inputs: N/A")
    logger.info(f"")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Output rows: {n_output_rows:,}")
    logger.info(f"Output unique users: {n_output_users:,}" if user_col else "Output unique users: N/A")
    
    # Check for user overlap
    if len(total_input_users) > 0 and user_col:
        overlap_info = n_output_users == len(total_input_users)
        if overlap_info:
            logger.info(f"User overlap: None (all users unique across splits)")
        else:
            overlapping = len(total_input_users) - n_output_users
            logger.info(f"User overlap: {overlapping:,} users appear in multiple splits")
    
    logger.info("=" * 60)
    
    return combined_df


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info(f"Combining EBNeRD behaviors from: {args.input_dir}")
    logger.info(f"Splits to combine: {args.splits}")
    
    try:
        combine_behaviors(
            input_dir=args.input_dir,
            splits=args.splits,
            output_file=args.output_file,
            keep_source_column=args.keep_source_column,
        )
        logger.info("\nBehaviors combined successfully!")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to combine behaviors: {e}")
        raise


if __name__ == "__main__":
    main()
