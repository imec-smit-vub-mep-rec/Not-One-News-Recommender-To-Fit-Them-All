"""
RecPack evaluation pipeline.

Provides a unified interface for running recommendation algorithm evaluations.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import issparse, csr_matrix
import gc

from ..utils.logging import get_logger, log_memory
from ..utils.io import load_dataframe, save_dataframe, ensure_dir
from .topic_diversity import (
    _normalize_categories,
    compute_topic_diversity,
    build_topic_report,
)

logger = get_logger("evaluation.recpack_pipeline")


# RecPack imports with error handling
try:
    from recpack.preprocessing.preprocessors import DataFramePreprocessor
    from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
    from recpack.scenarios import LastItemPrediction
    from recpack.algorithms import Popularity, ItemKNN, EASE, MultVAE
    from recpack.matrix import InteractionMatrix
    HAS_RECPACK = True
except ImportError:
    HAS_RECPACK = False
    logger.warning("RecPack not available. Install with: pip install recpack")


def check_recpack_available():
    """Check if RecPack is available."""
    if not HAS_RECPACK:
        raise ImportError(
            "RecPack is not installed. Install with: pip install recpack"
        )


def _to_csr_matrix(data: Any) -> Optional[csr_matrix]:
    """Convert InteractionMatrix / sparse / dense inputs to CSR matrix."""
    if data is None:
        return None

    values = data.values if hasattr(data, "values") else data
    if values is None:
        return None

    if issparse(values):
        return values.tocsr()

    return csr_matrix(np.asarray(values))


def _normalize_validation_tuple(candidate: Any) -> Optional[Tuple[csr_matrix, csr_matrix]]:
    """Normalize validation tuple to CSR matrices."""
    if not isinstance(candidate, tuple) or len(candidate) != 2:
        return None

    val_in = _to_csr_matrix(candidate[0])
    val_out = _to_csr_matrix(candidate[1])
    if val_in is None or val_out is None or val_out.nnz == 0:
        return None
    return val_in, val_out


def _build_validation_from_train(train_data: Any) -> Optional[Tuple[csr_matrix, csr_matrix]]:
    """Build a fallback validation split by holding out one item per eligible user."""
    train_csr = _to_csr_matrix(train_data)
    if train_csr is None:
        return None

    n_users, n_items = train_csr.shape
    in_rows: List[int] = []
    in_cols: List[int] = []
    in_vals: List[float] = []
    out_rows: List[int] = []
    out_cols: List[int] = []
    out_vals: List[float] = []

    for user_idx in range(n_users):
        start = train_csr.indptr[user_idx]
        end = train_csr.indptr[user_idx + 1]
        if end - start < 2:
            continue

        user_items = train_csr.indices[start:end]
        user_vals = train_csr.data[start:end]

        holdout_item = int(user_items[-1])
        holdout_val = float(user_vals[-1])
        out_rows.append(user_idx)
        out_cols.append(holdout_item)
        out_vals.append(holdout_val)

        hist_items = user_items[:-1]
        hist_vals = user_vals[:-1]
        if hist_items.size > 0:
            in_rows.extend([user_idx] * int(hist_items.size))
            in_cols.extend(hist_items.tolist())
            in_vals.extend(hist_vals.astype(np.float64).tolist())

    if not out_rows:
        return None

    val_in = csr_matrix((in_vals, (in_rows, in_cols)), shape=(n_users, n_items), dtype=np.float64)
    val_out = csr_matrix((out_vals, (out_rows, out_cols)), shape=(n_users, n_items), dtype=np.float64)
    if val_out.nnz == 0:
        return None
    return val_in, val_out


def _extract_validation_data(scenario: Any, train_data: Any) -> Optional[Tuple[csr_matrix, csr_matrix]]:
    """Extract validation data from scenario with fallback construction."""
    for attr_name in ("validation_data", "validation_set"):
        if hasattr(scenario, attr_name):
            normalized = _normalize_validation_tuple(getattr(scenario, attr_name))
            if normalized is not None:
                return normalized

    if hasattr(scenario, "validation_in") and hasattr(scenario, "validation_out"):
        normalized = _normalize_validation_tuple(
            (getattr(scenario, "validation_in"), getattr(scenario, "validation_out"))
        )
        if normalized is not None:
            return normalized

    logger.warning("Validation data not exposed by scenario; falling back to local holdout split")
    return _build_validation_from_train(train_data)


def _resolve_algorithm_params(
    algo_name: str,
    algorithm_params: Dict[str, Dict[str, Any]],
    k_values: List[int],
    seed: int,
) -> Dict[str, Any]:
    """Resolve algorithm params with MultVAE-safe defaults."""
    params = dict(algorithm_params.get(algo_name, {}))
    if algo_name == "MultVAE":
        if k_values:
            params.setdefault("predict_topK", int(max(k_values)))
        params.setdefault("stop_early", True)
        params.setdefault("max_iter_no_change", 5)
        params.setdefault("stopping_criterion", "ndcg")
        params.setdefault("seed", seed)
    elif algo_name in ('CB-ST', 'CB-ST-sklearn', 'CB-ST-annoy', 'SentenceTransformerContentBased'):
        # Map legacy n_trees -> annoy_n_trees (SentenceTransformerContentBased expects annoy_n_trees)
        if 'n_trees' in params and 'annoy_n_trees' not in params:
            params['annoy_n_trees'] = params.pop('n_trees')
    return params


def _compute_topk_item_exposure(predictions: Any, k: int) -> np.ndarray:
    """Count how often each item appears in users' Top-K recommendations."""
    if k <= 0:
        return np.array([], dtype=np.float64)

    if issparse(predictions):
        pred_csr = predictions.tocsr()
        n_items = pred_csr.shape[1]
        exposure = np.zeros(n_items, dtype=np.float64)

        for user_idx in range(pred_csr.shape[0]):
            row = pred_csr.getrow(user_idx)
            if row.nnz == 0:
                continue

            row_scores = row.data
            row_items = row.indices
            if row_scores.size > k:
                top_idx = np.argpartition(row_scores, -k)[-k:]
                top_items = row_items[top_idx]
            else:
                top_items = row_items

            exposure[top_items] += 1.0

        return exposure

    pred_arr = np.asarray(predictions)
    if pred_arr.ndim != 2:
        return np.array([], dtype=np.float64)

    n_users, n_items = pred_arr.shape
    exposure = np.zeros(n_items, dtype=np.float64)
    topk = min(k, n_items)

    if topk <= 0:
        return exposure

    for user_idx in range(n_users):
        row_scores = pred_arr[user_idx]
        if not np.any(np.isfinite(row_scores)):
            continue
        top_items = np.argpartition(row_scores, -topk)[-topk:]
        exposure[top_items] += 1.0

    return exposure


def _gini_from_exposure(exposure: np.ndarray) -> float:
    """Compute Gini coefficient from item exposure counts."""
    if exposure.size == 0:
        return 0.0

    exposure = np.clip(exposure.astype(np.float64), 0.0, None)
    total = exposure.sum()
    if total <= 0:
        return 0.0

    sorted_exposure = np.sort(exposure)
    n = sorted_exposure.size
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(index * sorted_exposure)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def _compute_diversity_at_k(predictions: Any, k_values: List[int]) -> Dict[str, float]:
    """Compute per-K diversity metrics from prediction scores."""
    diversity: Dict[str, float] = {}
    n_items = predictions.shape[1] if hasattr(predictions, "shape") else 0

    for k in k_values:
        exposure = _compute_topk_item_exposure(predictions, k)
        coverage = 0.0 if n_items == 0 else float(np.count_nonzero(exposure) / n_items)
        gini = _gini_from_exposure(exposure)
        diversity[f'CoverageK_{k}'] = coverage
        diversity[f'GiniK_{k}'] = gini

    return diversity


def _fit_and_predict_algorithm(algo_instance: Any, train_data: Any, test_in_data: Any) -> Any:
    """Fit and predict with RecPack algorithm APIs (public or protected)."""
    if hasattr(algo_instance, 'fit'):
        algo_instance.fit(train_data)
    elif hasattr(algo_instance, '_fit'):
        algo_instance._fit(train_data)
    else:
        raise AttributeError(f"{algo_instance.__class__.__name__} has no fit/_fit method")

    if hasattr(algo_instance, 'predict'):
        return algo_instance.predict(test_in_data)
    if hasattr(algo_instance, '_predict'):
        return algo_instance._predict(test_in_data)

    raise AttributeError(f"{algo_instance.__class__.__name__} has no predict/_predict method")


def create_interaction_matrix(
    interactions_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'article_id',
    time_col: str = 'impression_time',
    min_items_per_user: int = 5,
    min_users_per_item: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """Create RecPack InteractionMatrix from DataFrame.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        user_col: Name of user ID column
        item_col: Name of item ID column
        time_col: Name of timestamp column
        min_items_per_user: Minimum items per user filter
        min_users_per_item: Minimum users per item filter
        
    Returns:
        Tuple of (InteractionMatrix, preprocessing_info)
    """
    check_recpack_available()
    
    logger.info(f"Creating interaction matrix from {len(interactions_df)} interactions...")
    
    # Set up preprocessor
    preprocessor = DataFramePreprocessor(
        item_ix=item_col,
        user_ix=user_col,
        timestamp_ix=time_col,
    )
    
    # Add filters
    preprocessor.add_filter(MinItemsPerUser(min_items_per_user, item_col, user_col))
    preprocessor.add_filter(MinUsersPerItem(min_users_per_item, item_col, user_col))
    
    # Process data
    interaction_matrix = preprocessor.process(interactions_df)
    
    # Build item mapping as dict: original_id -> internal_id
    item_mapping_df = preprocessor.item_id_mapping
    item_mapping = dict(zip(
        item_mapping_df[item_col].astype(str),
        item_mapping_df['iid']
    ))
    
    # Build user mapping as dict
    user_mapping_df = preprocessor.user_id_mapping  
    user_mapping = dict(zip(
        user_mapping_df[user_col].astype(str),
        user_mapping_df['uid']
    ))
    
    # Get preprocessing info
    info = {
        'n_users': interaction_matrix.num_active_users,
        'n_items': interaction_matrix.num_active_items,
        'n_interactions': interaction_matrix.values.nnz,  # Use nnz for sparse matrix
        'density': interaction_matrix.density,
        'user_mapping': user_mapping,
        'item_mapping': item_mapping,
    }
    
    logger.info(f"Created matrix: {info['n_users']} users, {info['n_items']} items, "
                f"{info['n_interactions']} interactions (density: {info['density']:.4f})")
    
    return interaction_matrix, info

def get_available_algorithms(include_content_based: bool = True) -> Dict[str, Any]:
    """Get dictionary of available recommendation algorithms.
    
    Args:
        include_content_based: Whether to include content-based algorithm
        
    Returns:
        Dictionary mapping algorithm names to classes
    """
    check_recpack_available()
    
    algorithms = {
        'Popularity': Popularity,
        'ItemKNN': ItemKNN,
        'EASE': EASE,
        'MultVAE': MultVAE,
    }
    
    if include_content_based:
        try:
            from .algorithms.content_based import SentenceTransformerContentBased
            algorithms['CB-ST'] = SentenceTransformerContentBased
            algorithms['CB-ST-sklearn'] = SentenceTransformerContentBased
            algorithms['CB-ST-annoy'] = SentenceTransformerContentBased
            algorithms['SentenceTransformerContentBased'] = SentenceTransformerContentBased  # Alias
        except ImportError:
            logger.warning("Content-based algorithm not available")
    
    return algorithms


def run_evaluation(
    interaction_matrix: Any,
    algorithms: Optional[List[str]] = None,
    algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None,
    k_values: List[int] = [10, 20, 50],
    validation_split: float = 0.1,  # Deprecated: not used with LastItemPrediction
    test_split: float = 0.1,  # Deprecated: not used with LastItemPrediction
    seed: int = 42,
    content_df: Optional[pd.DataFrame] = None,
    item_mapping: Optional[Dict] = None,
    embeddings_df: Optional[pd.DataFrame] = None,
    embedding_column: str = 'embedding',
    batch_size: int = 10000,
    articles_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run evaluation for multiple algorithms using LastItemPrediction scenario.
    
    Uses LastItemPrediction scenario (matching legacy behavior) where each user's
    last interaction is held out for testing and earlier interactions are used
    for training.
    
    Args:
        interaction_matrix: RecPack InteractionMatrix
        algorithms: List of algorithm names (None = all available)
        k_values: List of k values for metrics
        algorithm_params: Per-algorithm constructor kwargs
        validation_split: DEPRECATED - not used with LastItemPrediction
        test_split: DEPRECATED - not used with LastItemPrediction
        seed: Random seed
        content_df: DataFrame with article content for CB algorithms
        item_mapping: Item ID mapping for CB algorithms
        embeddings_df: Optional DataFrame with pre-calculated embeddings.
                       If provided, CB-ST will use these instead of encoding content.
        embedding_column: Column name in embeddings_df containing the embedding vectors.
        batch_size: Batch size for prediction to avoid OOM errors.
        articles_df: Optional articles DataFrame with article_id and categories for topic-level metrics.
        
    Returns:
        Tuple of (results DataFrame, topic_reports list).
        topic_reports: List of dicts with keys algorithm, k, report_df (DataFrame with topic popularity).
    """
    check_recpack_available()
    algorithm_params = algorithm_params or {}
    
    available = get_available_algorithms(
        include_content_based=(content_df is not None or embeddings_df is not None)
    )
    
    if algorithms is None:
        algorithms = list(available.keys())
    
    logger.info(f"Running evaluation for algorithms: {algorithms}")
    logger.info(f"Metrics will be computed for k = {k_values}")
    
    # Use LastItemPrediction scenario (matches legacy behavior)
    # This predicts the last item each user interacted with, using all earlier items for training
    logger.info("Using LastItemPrediction scenario (legacy behavior)")
    
    try:
        scenario = LastItemPrediction(
            validation=True,
            seed=seed,
        )
        scenario.split(interaction_matrix)
    except (ZeroDivisionError, ValueError) as e:
        logger.warning(f"Failed to create scenario split (likely insufficient data): {e}")
        logger.warning("Returning empty results for this subset")
        return pd.DataFrame(), []
    
    # Get train and test data from scenario.
    train_matrix = scenario.full_training_data
    test_in, test_out = scenario.test_data

    train_data = train_matrix.values
    test_in_data = test_in.values
    test_out_data = test_out.values
    validation_data = _extract_validation_data(scenario, train_data)

    if test_out_data is None or test_out_data.nnz == 0:
        logger.warning("No test data available for evaluation")
        return pd.DataFrame(), []

    n_users = test_in_data.shape[0]
    n_items = test_in_data.shape[1]
    logger.info(f"Test data shape: {n_users} users x {n_items} items")
    logger.info(f"Using batch size: {batch_size} for prediction")

    # Instantiate algorithms.
    algo_instances: Dict[str, Any] = {}
    for algo_name in algorithms:
        if algo_name not in available:
            logger.warning(f"Algorithm {algo_name} not available, skipping")
            continue
        
        algo_class = available[algo_name]
        params = _resolve_algorithm_params(algo_name, algorithm_params, k_values, seed)
        
        if algo_name in ('CB-ST', 'CB-ST-sklearn', 'CB-ST-annoy', 'SentenceTransformerContentBased'):
            # Content-based algorithm needs content/item mapping and optional embeddings.
            if (content_df is not None or embeddings_df is not None) and item_mapping is not None:
                if algo_name == 'CB-ST-sklearn':
                    display_name = 'CB-ST-sklearn'
                    params.setdefault('backend', 'sklearn')
                elif algo_name == 'CB-ST-annoy':
                    display_name = 'CB-ST-annoy'
                    params.setdefault('backend', 'annoy')
                else:
                    display_name = 'CB-ST'
                    params.setdefault('backend', 'annoy')  # Default to annoy (faster)
                
                algo_instances[display_name] = algo_class(
                    content=content_df if content_df is not None else {},
                    item_mapping=item_mapping,
                    embeddings=embeddings_df,
                    embedding_column=embedding_column,
                    **params,
                )
                if embeddings_df is not None:
                    logger.info(
                        f"Created {display_name} with {params.get('backend')} backend "
                        "(using pre-calculated embeddings)"
                    )
                else:
                    logger.info(f"Created {display_name} with {params.get('backend')} backend")
            else:
                logger.warning("Content-based algorithm requires content_df and item_mapping")
        else:
            algo_instances[algo_name] = algo_class(**params)

    if not algo_instances:
        return pd.DataFrame(), []

    # Build topic mapping for topic-level diversity metrics
    internal_id_to_categories: Dict[int, List[str]] = {}
    all_topics: Set[str] = set()
    if articles_df is not None and item_mapping is not None:
        internal_id_to_categories, all_topics = _normalize_categories(articles_df, item_mapping)
        if all_topics:
            logger.info(
                f"Topic diversity: {len(all_topics)} topics, "
                f"{len(internal_id_to_categories)} items with categories"
            )

    # Evaluate all algorithms with batching
    from recpack.metrics import NDCGK, RecallK, PrecisionK, CoverageK
    rows: List[Dict[str, Any]] = []
    topic_reports: List[Dict[str, Any]] = []
    
    for algo_name, algo_instance in algo_instances.items():
        try:
            logger.info(f"Running {algo_name} evaluation...")
            
            # 1. Fit algorithm (once on full training data)
            requires_validation = (
                algo_name == "MultVAE" or algo_instance.__class__.__name__ == "MultVAE"
            )
            if hasattr(algo_instance, 'fit'):
                if requires_validation:
                    if validation_data is None:
                        logger.warning(
                            "Skipping %s: validation_data could not be prepared for fit()",
                            algo_name,
                        )
                        continue
                    try:
                        algo_instance.fit(train_data, validation_data=validation_data)
                    except TypeError:
                        # Support versions expecting positional validation tuple.
                        algo_instance.fit(train_data, validation_data)
                else:
                    algo_instance.fit(train_data)
            elif hasattr(algo_instance, '_fit'):
                algo_instance._fit(train_data)
            else:
                raise AttributeError(f"{algo_instance.__class__.__name__} has no fit/_fit method")
            
            # 2. Batched prediction and metric calculation
            # Initialize accumulators
            metric_sums = {}
            for k in k_values:
                metric_sums[f'NDCGK_{k}'] = 0.0
                metric_sums[f'RecallK_{k}'] = 0.0
                metric_sums[f'PrecisionK_{k}'] = 0.0
                metric_sums[f'CoverageK_{k}'] = 0.0 # Will be recomputed globally
                # Track exposure for diversity metrics
                metric_sums[f'exposure_{k}'] = np.zeros(n_items, dtype=np.float64)
            
            total_valid_users = 0
            
            # Iterate through batches
            for start_idx in range(0, n_users, batch_size):
                end_idx = min(start_idx + batch_size, n_users)
                current_batch_size = end_idx - start_idx
                
                # Slice input and output
                batch_in = test_in_data[start_idx:end_idx]
                batch_out = test_out_data[start_idx:end_idx]
                
                # Skip batch if no active users in test_out (though LastItemPrediction usually ensures this)
                batch_valid_users = batch_out.getnnz(axis=1).nonzero()[0].size
                
                if batch_valid_users == 0:
                    continue
                
                # Predict for batch
                if hasattr(algo_instance, 'predict'):
                    batch_pred = algo_instance.predict(batch_in)
                elif hasattr(algo_instance, '_predict'):
                    batch_pred = algo_instance._predict(batch_in)
                else:
                    raise AttributeError(f"{algo_instance.__class__.__name__} has no predict/_predict method")
                
                # Calculate metrics for batch
                for k in k_values:
                    # NDCG
                    ndcg = NDCGK(k)
                    ndcg.calculate(batch_out, batch_pred)
                    metric_sums[f'NDCGK_{k}'] += ndcg.value * batch_valid_users
                    
                    # Recall
                    recall = RecallK(k)
                    recall.calculate(batch_out, batch_pred)
                    metric_sums[f'RecallK_{k}'] += recall.value * batch_valid_users
                    
                    # Precision
                    precision = PrecisionK(k)
                    precision.calculate(batch_out, batch_pred)
                    metric_sums[f'PrecisionK_{k}'] += precision.value * batch_valid_users
                    
                    # Exposure (for Diversity)
                    batch_exposure = _compute_topk_item_exposure(batch_pred, k)
                    metric_sums[f'exposure_{k}'] += batch_exposure
                
                total_valid_users += batch_valid_users
                
                # Free memory
                del batch_pred
                gc.collect()
            
            # 3. Finalize results
            row: Dict[str, Any] = {'algorithm': algo_name}
            
            if total_valid_users > 0:
                for k in k_values:
                    # Average metrics
                    row[f'NDCGK_{k}'] = metric_sums[f'NDCGK_{k}'] / total_valid_users
                    row[f'RecallK_{k}'] = metric_sums[f'RecallK_{k}'] / total_valid_users
                    row[f'PrecisionK_{k}'] = metric_sums[f'PrecisionK_{k}'] / total_valid_users
                    
                    # Global diversity metrics from accumulated exposure
                    exposure = metric_sums[f'exposure_{k}']
                    coverage = 0.0 if n_items == 0 else float(np.count_nonzero(exposure) / n_items)
                    gini = _gini_from_exposure(exposure)
                    
                    row[f'CoverageK_{k}'] = coverage
                    row[f'GiniK_{k}'] = gini

                    # Topic-level diversity metrics
                    if internal_id_to_categories and all_topics:
                        topic_coverage, topic_gini, topic_exposure = compute_topic_diversity(
                            exposure, internal_id_to_categories, all_topics, k
                        )
                        row[f'CoverageK_topics_{k}'] = topic_coverage
                        row[f'GiniK_topics_{k}'] = topic_gini
                        report_df = build_topic_report(
                            topic_exposure, all_topics, algorithm=algo_name, k=k
                        )
                        topic_reports.append({"algorithm": algo_name, "k": k, "report_df": report_df})
                    else:
                        row[f'CoverageK_topics_{k}'] = 0.0
                        row[f'GiniK_topics_{k}'] = 0.0
            else:
                logger.warning(f"{algo_name}: No valid test users found.")
                for k in k_values:
                    row[f'NDCGK_{k}'] = 0.0
                    row[f'RecallK_{k}'] = 0.0
                    row[f'PrecisionK_{k}'] = 0.0
                    row[f'CoverageK_{k}'] = 0.0
                    row[f'GiniK_{k}'] = 0.0
                    row[f'CoverageK_topics_{k}'] = 0.0
                    row[f'GiniK_topics_{k}'] = 0.0

            rows.append(row)
            log_memory(f"{algo_name} done")
            logger.info(f"{algo_name} evaluation complete")
            
        except Exception as e:
            logger.warning(f"Failed to run {algo_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    return (pd.DataFrame(rows) if rows else pd.DataFrame()), topic_reports


class RecPackPipeline:
    """Class-based interface for RecPack evaluation.
    
    Provides a complete pipeline for loading data, preprocessing,
    and evaluating recommendation algorithms.
    """
    
    def __init__(
        self,
        k_values: List[int] = [10, 20, 50],
        min_items_per_user: int = 5,
        min_users_per_item: int = 1,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        embeddings_df: Optional[pd.DataFrame] = None,
        embedding_column: str = 'embedding',
        articles_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            k_values: List of k values for metrics
            min_items_per_user: Minimum items per user filter
            min_users_per_item: Minimum users per item filter
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            seed: Random seed
            embeddings_df: Optional DataFrame with pre-calculated embeddings
            embedding_column: Column name in embeddings_df containing embeddings
            articles_df: Optional articles DataFrame for topic-level diversity metrics
        """
        check_recpack_available()
        
        self.k_values = k_values
        self.min_items_per_user = min_items_per_user
        self.min_users_per_item = min_users_per_item
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed
        self.embeddings_df = embeddings_df
        self.embedding_column = embedding_column
        self.articles_df = articles_df
        
        self.interaction_matrix: Optional[Any] = None
        self.preprocessing_info: Optional[Dict] = None
        self.content_df: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.topic_reports: List[Dict[str, Any]] = []
    
    def load_data(
        self,
        interactions_path: str,
        content_path: Optional[str] = None,
        user_col: str = 'user_id',
        item_col: str = 'article_id',
        time_col: str = 'impression_time',
    ) -> 'RecPackPipeline':
        """Load interaction and content data.
        
        Args:
            interactions_path: Path to interactions file
            content_path: Optional path to content file
            user_col: Name of user ID column
            item_col: Name of item ID column
            time_col: Name of timestamp column
            
        Returns:
            Self for chaining
        """
        # Load interactions
        interactions_df = load_dataframe(interactions_path)
        logger.info(f"Loaded {len(interactions_df)} interactions")
        
        # Create interaction matrix
        self.interaction_matrix, self.preprocessing_info = create_interaction_matrix(
            interactions_df,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
            min_items_per_user=self.min_items_per_user,
            min_users_per_item=self.min_users_per_item,
        )
        
        # Load content if provided
        if content_path:
            self.content_df = load_dataframe(content_path)
            logger.info(f"Loaded content for {len(self.content_df)} items")
        
        return self

    def load_data_from_dataframes(
        self,
        interactions_df: pd.DataFrame,
        content_df: Optional[pd.DataFrame] = None,
        articles_df: Optional[pd.DataFrame] = None,
        user_col: str = 'user_id',
        item_col: str = 'article_id',
        time_col: str = 'impression_time',
    ) -> 'RecPackPipeline':
        """Load interaction and optional content/articles data directly from DataFrames."""
        logger.info(f"Loaded {len(interactions_df)} interactions (in-memory)")

        self.interaction_matrix, self.preprocessing_info = create_interaction_matrix(
            interactions_df,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
            min_items_per_user=self.min_items_per_user,
            min_users_per_item=self.min_users_per_item,
        )

        self.content_df = content_df
        if self.content_df is not None:
            logger.info(f"Loaded content for {len(self.content_df)} items (in-memory)")

        self.articles_df = articles_df if articles_df is not None else self.articles_df
        if self.articles_df is not None:
            logger.info(f"Loaded articles for topic diversity: {len(self.articles_df)} articles")

        return self
    
    def run(
        self,
        algorithms: Optional[List[str]] = None,
        algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Run evaluation.
        
        Args:
            algorithms: List of algorithm names (None = all)
            algorithm_params: Per-algorithm constructor kwargs
            
        Returns:
            DataFrame with results
        """
        if self.interaction_matrix is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        item_mapping = self.preprocessing_info.get('item_mapping')
        
        self.results, self.topic_reports = run_evaluation(
            self.interaction_matrix,
            algorithms=algorithms,
            algorithm_params=algorithm_params,
            k_values=self.k_values,
            validation_split=self.validation_split,
            test_split=self.test_split,
            seed=self.seed,
            content_df=self.content_df,
            item_mapping=item_mapping,
            embeddings_df=self.embeddings_df,
            embedding_column=self.embedding_column,
            articles_df=self.articles_df,
        )
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to file.
        
        Args:
            output_path: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Run evaluation first.")
        
        save_dataframe(self.results, output_path)
        logger.info(f"Saved results to {output_path}")
    
    def get_results(self) -> Optional[pd.DataFrame]:
        """Get evaluation results.
        
        Returns:
            Results DataFrame or None
        """
        return self.results
    
    def get_preprocessing_info(self) -> Optional[Dict]:
        """Get preprocessing information.
        
        Returns:
            Preprocessing info dict or None
        """
        return self.preprocessing_info


def _evaluate_single_cluster(
    cluster_id: int,
    cluster_interactions: pd.DataFrame,
    content_df: Optional[pd.DataFrame],
    algorithms: Optional[List[str]],
    algorithm_params: Optional[Dict[str, Dict[str, Any]]],
    k_values: List[int],
    min_items_per_user: int,
    output_dir: Optional[str],
    embeddings_df: Optional[pd.DataFrame] = None,
    embedding_column: str = 'embedding',
    articles_df: Optional[pd.DataFrame] = None,
) -> Tuple[int, Optional[pd.DataFrame]]:
    """Evaluate a single cluster. Helper for parallel execution.
    
    Args:
        cluster_id: Cluster identifier
        cluster_interactions: Interactions for this cluster
        content_df: Optional content DataFrame
        algorithms: List of algorithm names
        algorithm_params: Per-algorithm constructor kwargs
        k_values: List of k values for metrics
        min_items_per_user: Minimum items per user for RecPack filter
        output_dir: Optional directory to save results
        embeddings_df: Optional DataFrame with pre-calculated embeddings
        embedding_column: Column name containing embeddings
        articles_df: Optional articles DataFrame for topic-level diversity metrics
        
    Returns:
        Tuple of (cluster_id, results DataFrame or None if skipped)
    """
    logger.info(f"Evaluating cluster {cluster_id} ({len(cluster_interactions)} interactions)...")
    log_memory(f"cluster {cluster_id} start")
    
    # Skip if too few interactions
    if len(cluster_interactions) < 100:
        logger.warning(f"Cluster {cluster_id} has too few interactions, skipping")
        log_memory(f"cluster {cluster_id} skipped")
        return cluster_id, None
    
    # Resume logic: Check if results already exist
    if output_dir:
        output_path = Path(output_dir) / f'cluster_{cluster_id}_results.csv'
        if output_path.exists():
            logger.info(f"Skipping cluster {cluster_id} - results already exist at {output_path}")
            try:
                # Assuming results are saved as CSV
                results = pd.read_csv(output_path)
                return cluster_id, results
            except Exception as e:
                logger.warning(f"Failed to load existing results for cluster {cluster_id}, re-running: {e}")

    # Run pipeline (min_items_per_user filtering happens here)
    pipeline = RecPackPipeline(
        k_values=k_values,
        min_items_per_user=min_items_per_user,
        embeddings_df=embeddings_df,
        embedding_column=embedding_column,
        articles_df=articles_df,
    )
    
    try:
        pipeline.load_data_from_dataframes(
            interactions_df=cluster_interactions,
            content_df=content_df,
            articles_df=articles_df,
        )
        
        cluster_results = pipeline.run(
            algorithms=algorithms,
            algorithm_params=algorithm_params,
        )
        
        # Save if output dir provided
        if output_dir:
            ensure_dir(output_dir)
            output_path = Path(output_dir) / f'cluster_{cluster_id}_results.csv'
            pipeline.save_results(str(output_path))

            # Save topic report if available
            if pipeline.topic_reports:
                report_dfs = []
                for tr in pipeline.topic_reports:
                    df = tr["report_df"].copy()
                    df["cluster_id"] = cluster_id
                    report_dfs.append(df)
                topic_report_path = Path(output_dir) / f'topic_report_cluster_{cluster_id}.csv'
                combined = pd.concat(report_dfs, ignore_index=True)
                save_dataframe(combined, str(topic_report_path), format="csv")
                logger.info(f"Saved topic report to {topic_report_path}")

        log_memory(f"cluster {cluster_id} end")
        return cluster_id, cluster_results
    
    except Exception as e:
        logger.error(f"Error evaluating cluster {cluster_id}: {e}")
        log_memory(f"cluster {cluster_id} error")
        return cluster_id, None


def run_cluster_evaluation(
    interactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    content_df: Optional[pd.DataFrame] = None,
    articles_df: Optional[pd.DataFrame] = None,
    algorithms: Optional[List[str]] = None,
    algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None,
    k_values: List[int] = [10, 20, 50],
    min_items_per_user: int = 5,
    output_dir: Optional[str] = None,
    n_jobs: int = 1,
    embeddings_df: Optional[pd.DataFrame] = None,
    embedding_column: str = 'embedding',
) -> Dict[int, pd.DataFrame]:
    """Run evaluation for each user cluster.
    
    NOTE: User filtering (min_items_per_user) happens HERE during RecPack
    preprocessing, NOT during data cleaning. This ensures clustering happens
    on ALL users, while evaluation filters to users with enough interactions.
    
    Args:
        interactions_df: Full interactions DataFrame (all users, including those
                         who will be filtered out by min_items_per_user)
        users_df: DataFrame with user_id and cluster_id columns (from clustering,
                  which includes ALL users)
        content_df: Optional content DataFrame
        articles_df: Optional articles DataFrame for topic-level diversity metrics
        algorithms: List of algorithm names
        algorithm_params: Per-algorithm constructor kwargs
        k_values: List of k values for metrics
        min_items_per_user: Minimum items per user for RecPack filter (default: 5).
                            Users with fewer interactions are excluded from evaluation
                            but were still included in clustering.
        output_dir: Optional directory to save results
        n_jobs: Number of parallel jobs (1 = sequential, -1 = all cores).
                Note: Use n_jobs=1 if running on GPU to avoid memory issues.
        embeddings_df: Optional DataFrame with pre-calculated embeddings.
                       If provided, CB-ST algorithm will use these instead of encoding.
                       This significantly speeds up evaluation on large datasets.
        embedding_column: Column name in embeddings_df containing the embedding vectors.
                         Default: 'google-bert/bert-base-multilingual-cased'
        
    Returns:
        Dictionary mapping cluster_id to results DataFrame
    """
    check_recpack_available()
    
    cluster_ids = sorted(users_df['cluster_id'].unique())
    
    logger.info(f"Running evaluation for {len(cluster_ids)} clusters (n_jobs={n_jobs})...")
    logger.info(f"NOTE: Users with < {min_items_per_user} interactions will be filtered by RecPack")
    if embeddings_df is not None:
        logger.info(f"Using pre-calculated embeddings from column '{embedding_column}'")
    
    # Prepare cluster data
    cluster_data = {}
    users_df_local = users_df.copy()
    users_df_local['user_id'] = users_df_local['user_id'].astype(str)
    interactions_df_local = interactions_df.copy()
    interactions_df_local['user_id'] = interactions_df_local['user_id'].astype(str)
    for cluster_id in cluster_ids:
        cluster_users = users_df_local[users_df_local['cluster_id'] == cluster_id]['user_id']
        cluster_interactions = interactions_df_local[
            interactions_df_local['user_id'].isin(cluster_users)
        ].copy()
        cluster_data[cluster_id] = cluster_interactions
        logger.info(f"Cluster {cluster_id}: {len(cluster_users)} users, {len(cluster_interactions)} interactions")
    
    # Run evaluations
    if n_jobs == 1:
        # Sequential execution
        results_list = []
        for cluster_id, cluster_interactions in cluster_data.items():
            results_list.append(_evaluate_single_cluster(
                cluster_id, cluster_interactions, content_df,
                algorithms, algorithm_params, k_values, min_items_per_user, output_dir,
                embeddings_df, embedding_column, articles_df
            ))
            gc.collect()
            log_memory("between clusters")
    else:
        # Parallel execution
        from joblib import Parallel, delayed
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_single_cluster)(
                cluster_id, cluster_interactions, content_df,
                algorithms, algorithm_params, k_values, min_items_per_user, output_dir,
                embeddings_df, embedding_column, articles_df
            )
            for cluster_id, cluster_interactions in cluster_data.items()
        )
    
    # Convert to dict, filtering out None results
    results = {cid: res for cid, res in results_list if res is not None}
    
    logger.info(f"Completed evaluation for {len(results)}/{len(cluster_ids)} clusters")
    
    return results
