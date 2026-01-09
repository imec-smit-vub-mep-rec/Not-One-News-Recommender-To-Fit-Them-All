"""
File I/O utilities for the RICON analysis pipeline.

Provides unified interfaces for loading and saving data in various formats.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataframe(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Load a DataFrame from a file.
    
    Args:
        path: Path to the file
        format: File format ('csv', 'parquet', 'json', 'jsonl'). 
                Auto-detected from extension if None.
        **kwargs: Additional arguments passed to the pandas reader
        
    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lower().lstrip('.')
    
    if format == 'csv':
        return pd.read_csv(path, **kwargs)
    elif format == 'parquet':
        # Use pyarrow engine for better performance
        return pd.read_parquet(path, engine='pyarrow', **kwargs)
    elif format == 'json':
        return pd.read_json(path, **kwargs)
    elif format == 'jsonl':
        return pd.read_json(path, lines=True, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: Optional[str] = None,
    compression: Optional[str] = None,
    **kwargs
) -> Path:
    """Save a DataFrame to a file.
    
    Args:
        df: DataFrame to save
        path: Path to save to
        format: File format ('csv', 'parquet', 'json'). 
                Auto-detected from extension if None.
        compression: Compression method (e.g., 'gzip', 'snappy').
                     Default for parquet is 'snappy' (faster than gzip).
        **kwargs: Additional arguments passed to the pandas writer
        
    Returns:
        Path to the saved file
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    if format is None:
        format = path.suffix.lower().lstrip('.')
    
    if format == 'csv':
        df.to_csv(path, index=False, **kwargs)
    elif format == 'parquet':
        # Use snappy compression (faster) and pyarrow engine
        df.to_parquet(
            path,
            index=False,
            compression=compression or 'snappy',
            engine='pyarrow',
            **kwargs
        )
    elif format == 'json':
        df.to_json(path, orient='records', indent=2, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Parsed JSON as a dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> Path:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save to
        indent: Indentation level
        
    Returns:
        Path to the saved file
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    return path


def iter_jsonl(path: Union[str, Path], chunk_size: int = 10000):
    """Iterate over a JSONL file in chunks.
    
    Args:
        path: Path to the JSONL file
        chunk_size: Number of lines per chunk
        
    Yields:
        DataFrames containing chunks of the file
    """
    for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
        yield chunk


def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(path)
    
    if not path.exists():
        return {"exists": False}
    
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.absolute()),
        "name": path.name,
        "extension": path.suffix,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": stat.st_mtime,
    }


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> list:
    """List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))
