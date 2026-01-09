"""Evaluation module for recommendation algorithms."""

from .recpack_pipeline import (
    RecPackPipeline,
    run_evaluation,
    run_cluster_evaluation,
    create_interaction_matrix,
)
from .analysis import (
    ResultsAnalyzer,
    compute_significance,
    aggregate_results,
    compare_clusters,
)
from .algorithms.content_based import SentenceTransformerContentBased

__all__ = [
    # Pipeline
    "RecPackPipeline",
    "run_evaluation",
    "run_cluster_evaluation",
    "create_interaction_matrix",
    # Analysis
    "ResultsAnalyzer",
    "compute_significance",
    "aggregate_results",
    "compare_clusters",
    # Algorithms
    "SentenceTransformerContentBased",
]
