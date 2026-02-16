"""
Centralized configuration management for the RICON analysis pipeline.

This module provides dataclasses for configuring all aspects of the pipeline,
from data conversion to clustering to evaluation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset.
    
    Attributes:
        name: Dataset identifier (e.g., 'adressa', 'ebnerd')
        input_path: Path to raw dataset files
        output_path: Path for processed output files
        format: Input format ('jsonl', 'parquet', 'csv')
        column_mapping: Map from source columns to standard columns
        session_timeout_seconds: Seconds of inactivity before new session (default 30 min)
        min_impressions_per_user: Minimum impressions to include user
        min_users_per_item: Minimum users to include item
    """
    name: str
    input_path: str
    output_path: str = ""
    format: str = "parquet"
    column_mapping: Dict[str, str] = field(default_factory=dict)
    session_timeout_seconds: int = 1800  # 30 minutes
    min_impressions_per_user: int = 5
    min_users_per_item: int = 1
    # Optional ingestion controls for large partitioned datasets (e.g. AD on S3).
    event_types: Optional[List[str]] = None
    start_time_min: Optional[str] = None
    start_time_max: Optional[str] = None
    
    def __post_init__(self):
        if not self.output_path:
            self.output_path = str(Path(self.input_path).parent / "processed")


@dataclass
class ClusteringConfig:
    """Configuration for user clustering.
    
    Attributes:
        features: List of features to use for clustering
        n_clusters: Number of clusters (None = auto-detect via elbow method)
        max_clusters: Maximum clusters to test in elbow method
        random_state: Random seed for reproducibility
        scaler: Scaler type ('standard', 'minmax', 'robust')
        imputation_strategy: Strategy for missing values ('mean', 'median', 'zero')
        min_impressions_per_user: Minimum impressions required per user
        k_selection_method: Method for selecting K ('elbow', 'silhouette', 'manual')
        legacy_features: If True, use the legacy feature set (disables per-category
                         proportions and time-of-day features, enables session behavior
                         features like avg_category_switches and avg_session_duration)
    """
    features: List[str] = field(default_factory=lambda: [
        'count_sessions',
        'count_total_impressions',
        'count_total_homepage_impressions',
        'count_total_article_impressions',
        'count_total_unique_categories',
        'count_total_unique_articles',
        'avg_reading_time',
        'proportion_article_time',
        'avg_session_length',
        'avg_categories_per_session',
        'avg_category_switches',
        'avg_session_duration',
    ])
    n_clusters: Optional[int] = None  # None = auto-detect
    max_clusters: int = 10
    random_state: int = 42
    scaler: str = "standard"
    imputation_strategy: str = "mean"
    min_impressions_per_user: int = 5
    k_selection_method: str = "elbow"
    legacy_features: bool = False


@dataclass 
class AlgorithmConfig:
    """Configuration for a single recommendation algorithm.
    
    Attributes:
        name: Algorithm name (must match RecPack registry)
        enabled: Whether to include this algorithm
        params: Fixed parameters for the algorithm
        grid: Grid search parameters for hyperparameter tuning
    """
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    grid: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for RecPack evaluation.
    
    Attributes:
        algorithms: List of algorithm configurations
        scenarios: List of scenario names to run
        metrics: List of metric names
        k_values: List of K values for metrics like NDCG@K
        optimization_metric: Metric used for hyperparameter optimization
        optimization_k: K value for optimization metric
        content_model: Sentence transformer model for content-based
        content_embedding_dim: Embedding dimension (None = auto-detect)
        content_n_trees: Number of trees for Annoy index
        content_num_neighbors: Number of neighbors to retrieve
    """
    algorithms: List[AlgorithmConfig] = field(default_factory=lambda: [
        AlgorithmConfig(name='Popularity'),
        AlgorithmConfig(
            name='ItemKNN',
            grid={
                'K': [50, 100, 200],
                'normalize_sim': [True, False],
                'normalize_X': [True, False]
            }
        ),
        AlgorithmConfig(
            name='EASE',
            grid={'l2': [1, 10, 100, 1000]}
        ),
        AlgorithmConfig(
            name='MultVAE',
            enabled=False,
            params={
                'batch_size': 500,
                'max_epochs': 200,
                'learning_rate': 0.0001,
                'dim_bottleneck_layer': 200,
                'dim_hidden_layer': 600,
                'max_beta': 0.2,
                'anneal_steps': 200000,
                'dropout': 0.5,
            }
        ),
        AlgorithmConfig(
            name='SentenceTransformerContentBased',
            params={
                'metric': 'angular',
                'n_trees': 20,
                'num_neighbors': 100,
                'verbose': False,
            }
        ),
    ])
    scenarios: List[str] = field(default_factory=lambda: [
        'WeakGeneralization',
        'Timed',
        'LastItemPrediction',
    ])
    metrics: List[str] = field(default_factory=lambda: ['NDCGK'])
    k_values: List[int] = field(default_factory=lambda: [10, 20, 50])
    optimization_metric: str = 'NDCGK'
    optimization_k: int = 100
    
    # Content-based algorithm settings
    content_mode: str = "legacy"
    content_model: str = 'intfloat/multilingual-e5-large'
    content_embedding_dim: Optional[int] = 1024
    content_n_trees: int = 20
    content_num_neighbors: int = 100
    
    # Timed scenario settings
    validation_quantile: float = 0.71
    test_quantile: float = 0.86
    
    # WeakGeneralization settings
    frac_data_in: float = 0.8
    
    # LastItemPrediction settings
    n_most_recent_in: int = 30


@dataclass
class SessionConfig:
    """Configuration for a single analysis session/run.
    
    Attributes:
        base_output_dir: Base directory for all runs
        run_id: Unique identifier for this run (auto-generated if None)
        dataset_name: Name of the dataset being processed
        timestamp: When the session started
    """
    base_output_dir: str = "runs"
    run_id: Optional[str] = None
    dataset_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{self.dataset_name}_{self.timestamp}" if self.dataset_name else self.timestamp
    
    @property
    def session_dir(self) -> Path:
        """Get the full path to this session's output directory."""
        return Path(self.base_output_dir) / self.run_id
    
    @property
    def logs_dir(self) -> Path:
        """Get the path to the logs directory."""
        return self.session_dir / "logs"
    
    @property
    def data_dir(self) -> Path:
        """Get the path to the processed data directory."""
        return self.session_dir / "data"
    
    @property
    def clusters_dir(self) -> Path:
        """Get the path to the clustering results directory."""
        return self.session_dir / "clusters"
    
    @property
    def evaluation_dir(self) -> Path:
        """Get the path to the evaluation results directory."""
        return self.session_dir / "evaluation"
    
    @property
    def figures_dir(self) -> Path:
        """Get the path to the figures/plots directory."""
        return self.session_dir / "figures"


@dataclass
class PipelineConfig:
    """Master configuration combining all pipeline settings.
    
    Attributes:
        dataset: Dataset configuration
        clustering: Clustering configuration
        evaluation: Evaluation configuration
        session: Session/run configuration
        verbose: Whether to print detailed progress
        save_intermediate: Whether to save intermediate results
        overwrite: Whether to overwrite existing results
    """
    dataset: DatasetConfig
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    verbose: bool = True
    save_intermediate: bool = True
    overwrite: bool = False
    
    def __post_init__(self):
        # Update session with dataset name
        if self.dataset and not self.session.dataset_name:
            self.session.dataset_name = self.dataset.name
            self.session.run_id = f"{self.dataset.name}_{self.session.timestamp}"


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        PipelineConfig instance
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct nested dataclasses
    dataset = DatasetConfig(**data.get('dataset', {}))
    clustering = ClusteringConfig(**data.get('clustering', {}))
    
    # Handle algorithms list in evaluation
    eval_data = data.get('evaluation', {})
    if 'algorithms' in eval_data:
        eval_data['algorithms'] = [
            AlgorithmConfig(**algo) for algo in eval_data['algorithms']
        ]
    evaluation = EvaluationConfig(**eval_data)
    
    session = SessionConfig(**data.get('session', {}))
    
    return PipelineConfig(
        dataset=dataset,
        clustering=clustering,
        evaluation=evaluation,
        session=session,
        verbose=data.get('verbose', True),
        save_intermediate=data.get('save_intermediate', True),
        overwrite=data.get('overwrite', False),
    )


def save_config(config: PipelineConfig, config_path: str) -> None:
    """Save pipeline configuration to a JSON file.
    
    Args:
        config: PipelineConfig instance to save
        config_path: Path to save the JSON file
    """
    # Convert to dict, handling nested dataclasses
    data = asdict(config)
    
    # Ensure parent directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# Preset configurations for common datasets
PRESET_CONFIGS = {
    'ad': DatasetConfig(
        name='ad',
        input_path='',
        format='spark_csv',
        column_mapping={
            'ARTICLE_IDENTIFIER': 'article_id',
            'MAPPED_USER_IDENTIFIER': 'user_id',
            'START_TIME': 'impression_time',
            'IMPRESSION_ID': 'impression_id',
            'TIME_ON_PAGE': 'read_time',
            'SESSION_ID': 'session_id',
            'IS_LOGGED_IN': 'is_subscriber',
            'main_section': 'category_str',
            'title': 'title',
        },
        event_types=['home_page_view', 'article_page_view'],
    ),
    'adressa': DatasetConfig(
        name='adressa',
        input_path='',
        format='jsonl',
        column_mapping={
            'userId': 'user_id',
            'id': 'article_id',
            'time': 'impression_time',
            'eventId': 'impression_id',
            'activeTime': 'read_time',
            'publishtime': 'time_published',
            'title': 'title',
            'category1': 'category_str',
        },
        session_timeout_seconds=1800,
    ),
    'ebnerd': DatasetConfig(
        name='ebnerd',
        input_path='',
        format='parquet',
        column_mapping={
            'article_id': 'article_id',
            'user_id': 'user_id',
            'impression_time': 'impression_time',
            'impression_id': 'impression_id',
            'read_time': 'read_time',
            'published_time': 'time_published',
            'title': 'title',
            'category_str': 'category_str',
            'sentiment_score': 'sentiment_score',
            'session_id': 'session_id',
            'is_subscriber': 'is_subscriber',
            # Note: is_sso_user is kept as-is, not mapped to is_subscriber to avoid duplicates
        },
    ),
    # hln and vk use the same S3 Spark CSV structure as ad (article_metadata.csv + impressions/)
    'hln': DatasetConfig(
        name='hln',
        input_path='',
        format='spark_csv',
        column_mapping={
            'ARTICLE_IDENTIFIER': 'article_id',
            'MAPPED_USER_IDENTIFIER': 'user_id',
            'START_TIME': 'impression_time',
            'IMPRESSION_ID': 'impression_id',
            'TIME_ON_PAGE': 'read_time',
            'SESSION_ID': 'session_id',
            'IS_LOGGED_IN': 'is_subscriber',
            'main_section': 'category_str',
            'title': 'title',
        },
        event_types=['home_page_view', 'article_page_view'],
    ),
    'vk': DatasetConfig(
        name='vk',
        input_path='',
        format='spark_csv',
        column_mapping={
            'ARTICLE_IDENTIFIER': 'article_id',
            'MAPPED_USER_IDENTIFIER': 'user_id',
            'START_TIME': 'impression_time',
            'IMPRESSION_ID': 'impression_id',
            'TIME_ON_PAGE': 'read_time',
            'SESSION_ID': 'session_id',
            'IS_LOGGED_IN': 'is_subscriber',
            'main_section': 'category_str',
            'title': 'title',
        },
        event_types=['home_page_view', 'article_page_view'],
    ),
}


def get_preset_config(dataset_name: str, input_path: str) -> DatasetConfig:
    """Get a preset configuration for a known dataset.
    
    Args:
        dataset_name: Name of the dataset ('adressa', 'ebnerd')
        input_path: Path to the raw dataset files
        
    Returns:
        DatasetConfig with preset values
    """
    if dataset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[dataset_name]
    config.input_path = input_path
    return config
