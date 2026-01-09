"""
RICON Analysis Pipeline

A modular framework for user clustering and recommendation evaluation.

Modules:
    config: Configuration management and schema definitions
    utils: Utility functions for I/O, logging, and session management
    converters: Dataset converters for different data formats
    preprocessing: Data validation, cleaning, and transformation
    clustering: User feature engineering and clustering
    evaluation: Recommendation algorithm evaluation
"""

from .config import (
    DatasetConfig,
    ClusteringConfig,
    EvaluationConfig,
    PipelineConfig,
    load_config,
    save_config,
    PRESET_CONFIGS,
)
from .utils import (
    Session,
    setup_logging,
    get_logger,
    load_dataframe,
    save_dataframe,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "DatasetConfig",
    "ClusteringConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "load_config",
    "save_config",
    "PRESET_CONFIGS",
    # Utils
    "Session",
    "setup_logging",
    "get_logger",
    "load_dataframe",
    "save_dataframe",
    # Version
    "__version__",
]
