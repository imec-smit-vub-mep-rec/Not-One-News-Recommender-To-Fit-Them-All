"""
RecPack evaluation pipeline.

Provides a unified interface for running recommendation algorithm evaluations.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.io import load_dataframe, save_dataframe, ensure_dir


logger = get_logger("evaluation.recpack_pipeline")


# RecPack imports with error handling
try:
    from recpack.preprocessing.preprocessors import DataFramePreprocessor
    from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
    from recpack.scenarios import LastItemPrediction
    from recpack.pipelines import PipelineBuilder
    from recpack.algorithms import Popularity, ItemKNN, EASE
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
    k_values: List[int] = [10, 20, 50],
    validation_split: float = 0.1,  # Deprecated: not used with LastItemPrediction
    test_split: float = 0.1,  # Deprecated: not used with LastItemPrediction
    seed: int = 42,
    content_df: Optional[pd.DataFrame] = None,
    item_mapping: Optional[Dict] = None,
    embeddings_df: Optional[pd.DataFrame] = None,
    embedding_column: str = 'embedding',
) -> pd.DataFrame:
    """Run evaluation for multiple algorithms using LastItemPrediction scenario.
    
    Uses LastItemPrediction scenario (matching legacy behavior) where each user's
    last interaction is held out for testing and earlier interactions are used
    for training.
    
    Args:
        interaction_matrix: RecPack InteractionMatrix
        algorithms: List of algorithm names (None = all available)
        k_values: List of k values for metrics
        validation_split: DEPRECATED - not used with LastItemPrediction
        test_split: DEPRECATED - not used with LastItemPrediction
        seed: Random seed
        content_df: DataFrame with article content for CB algorithms
        item_mapping: Item ID mapping for CB algorithms
        embeddings_df: Optional DataFrame with pre-calculated embeddings.
                       If provided, CB-ST will use these instead of encoding content.
        embedding_column: Column name in embeddings_df containing the embedding vectors.
        
    Returns:
        DataFrame with evaluation results
    """
    check_recpack_available()
    
    available = get_available_algorithms(include_content_based=content_df is not None)
    
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
        return pd.DataFrame()
    
    # Build and run pipeline for standard RecPack algorithms
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)
    
    # Track which algorithms need special handling
    cb_algo_instances = {}  # Store multiple CB instances with different backends
    recpack_algorithms = []
    
    # Add algorithms
    for algo_name in algorithms:
        if algo_name not in available:
            logger.warning(f"Algorithm {algo_name} not available, skipping")
            continue
        
        algo_class = available[algo_name]
        
        if algo_name in ('CB-ST', 'CB-ST-sklearn', 'CB-ST-annoy', 'SentenceTransformerContentBased'):
            # Content-based algorithm needs special initialization - handle separately
            if (content_df is not None or embeddings_df is not None) and item_mapping is not None:
                # Determine backend
                if algo_name == 'CB-ST-sklearn':
                    backend = 'sklearn'
                    display_name = 'CB-ST-sklearn'
                elif algo_name == 'CB-ST-annoy':
                    backend = 'annoy'
                    display_name = 'CB-ST-annoy'
                else:
                    backend = 'annoy'  # Default to annoy (faster)
                    display_name = 'CB-ST'
                
                # Use pre-calculated embeddings if available
                cb_algo_instances[display_name] = algo_class(
                    content=content_df if content_df is not None else {},
                    item_mapping=item_mapping,
                    backend=backend,
                    embeddings=embeddings_df,
                    embedding_column=embedding_column,
                )
                if embeddings_df is not None:
                    logger.info(f"Created {display_name} with {backend} backend (using pre-calculated embeddings)")
                else:
                    logger.info(f"Created {display_name} with {backend} backend")
            else:
                logger.warning("Content-based algorithm requires content_df and item_mapping")
        else:
            # Standard RecPack algorithms
            builder.add_algorithm(algo_class)
            recpack_algorithms.append(algo_name)
    
    # Add metrics
    from recpack.metrics import NDCGK, RecallK, PrecisionK
    
    for k in k_values:
        builder.add_metric(NDCGK, K=k)
        builder.add_metric(RecallK, K=k)
        builder.add_metric(PrecisionK, K=k)
    
    # Run pipeline for standard algorithms
    results_list = []
    
    if recpack_algorithms:
        pipeline = builder.build()
        pipeline.run()
        
        # Get results and format properly
        results = pipeline.get_metrics(short=True)
        results = results.reset_index()
        results = results.rename(columns={'index': 'algorithm'})
        results_list.append(results)
    
    # Run CB-ST instances separately if configured
    if cb_algo_instances:
        # Get train and test data from scenario
        # full_training_data is the data for training (before test time)
        train_matrix = scenario.full_training_data
        
        # test_data is a tuple of (in_data, out_data)
        # in_data is the known interactions for users in test set
        # out_data is the ground truth (interactions to predict)
        test_in, test_out = scenario.test_data
        
        # Convert to sparse matrices for our algorithm
        train_data = train_matrix.values
        test_in_data = test_in.values
        test_out_data = test_out.values
        
        if test_out_data is None or test_out_data.nnz == 0:
            logger.warning("No test data available for CB-ST evaluation")
        else:
            for cb_name, cb_algo_instance in cb_algo_instances.items():
                try:
                    logger.info(f"Running {cb_name} evaluation...")
                    
                    # Fit the model on training data
                    cb_algo_instance._fit(train_data)
                    
                    # Predict for test users using their known interactions
                    predictions = cb_algo_instance._predict(test_in_data)
                    
                    # Calculate metrics manually
                    from recpack.metrics import NDCGK, RecallK, PrecisionK
                    
                    cb_results = {'algorithm': cb_name}
                    for k in k_values:
                        ndcg = NDCGK(k)
                        recall = RecallK(k)
                        precision = PrecisionK(k)
                        
                        ndcg.calculate(test_out_data, predictions)
                        recall.calculate(test_out_data, predictions)
                        precision.calculate(test_out_data, predictions)
                        
                        cb_results[f'NDCGK_{k}'] = ndcg.value
                        cb_results[f'RecallK_{k}'] = recall.value
                        cb_results[f'PrecisionK_{k}'] = precision.value
                    
                    results_list.append(pd.DataFrame([cb_results]))
                    logger.info(f"{cb_name} evaluation complete")
                    
                except Exception as e:
                    logger.warning(f"Failed to run {cb_name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
    
    # Combine results
    if results_list:
        return pd.concat(results_list, ignore_index=True)
    else:
        return pd.DataFrame()


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
        
        self.interaction_matrix: Optional[Any] = None
        self.preprocessing_info: Optional[Dict] = None
        self.content_df: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
    
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
    
    def run(
        self,
        algorithms: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run evaluation.
        
        Args:
            algorithms: List of algorithm names (None = all)
            
        Returns:
            DataFrame with results
        """
        if self.interaction_matrix is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        item_mapping = self.preprocessing_info.get('item_mapping')
        
        self.results = run_evaluation(
            self.interaction_matrix,
            algorithms=algorithms,
            k_values=self.k_values,
            validation_split=self.validation_split,
            test_split=self.test_split,
            seed=self.seed,
            content_df=self.content_df,
            item_mapping=item_mapping,
            embeddings_df=self.embeddings_df,
            embedding_column=self.embedding_column,
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
    k_values: List[int],
    min_items_per_user: int,
    output_dir: Optional[str],
    embeddings_df: Optional[pd.DataFrame] = None,
    embedding_column: str = 'embedding',
) -> Tuple[int, Optional[pd.DataFrame]]:
    """Evaluate a single cluster. Helper for parallel execution.
    
    Args:
        cluster_id: Cluster identifier
        cluster_interactions: Interactions for this cluster
        content_df: Optional content DataFrame
        algorithms: List of algorithm names
        k_values: List of k values for metrics
        min_items_per_user: Minimum items per user for RecPack filter
        output_dir: Optional directory to save results
        embeddings_df: Optional DataFrame with pre-calculated embeddings
        embedding_column: Column name containing embeddings
        
    Returns:
        Tuple of (cluster_id, results DataFrame or None if skipped)
    """
    import tempfile
    import os
    
    logger.info(f"Evaluating cluster {cluster_id} ({len(cluster_interactions)} interactions)...")
    
    # Skip if too few interactions
    if len(cluster_interactions) < 100:
        logger.warning(f"Cluster {cluster_id} has too few interactions, skipping")
        return cluster_id, None
    
    # Run pipeline (min_items_per_user filtering happens here)
    pipeline = RecPackPipeline(
        k_values=k_values,
        min_items_per_user=min_items_per_user,
        embeddings_df=embeddings_df,
        embedding_column=embedding_column,
    )
    
    # Create temporary files for the cluster data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        cluster_interactions.to_csv(f.name, index=False)
        interactions_path = f.name
    
    try:
        content_path = None
        if content_df is not None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                content_df.to_csv(f.name, index=False)
                content_path = f.name
        
        pipeline.load_data(
            interactions_path=interactions_path,
            content_path=content_path,
        )
        
        cluster_results = pipeline.run(algorithms=algorithms)
        
        # Save if output dir provided
        if output_dir:
            ensure_dir(output_dir)
            output_path = Path(output_dir) / f'cluster_{cluster_id}_results.csv'
            pipeline.save_results(str(output_path))
        
        return cluster_id, cluster_results
    
    except Exception as e:
        logger.error(f"Error evaluating cluster {cluster_id}: {e}")
        return cluster_id, None
    
    finally:
        # Clean up temp files
        os.unlink(interactions_path)
        if content_path:
            os.unlink(content_path)


def run_cluster_evaluation(
    interactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    content_df: Optional[pd.DataFrame] = None,
    algorithms: Optional[List[str]] = None,
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
        algorithms: List of algorithm names
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
    for cluster_id in cluster_ids:
        cluster_users = users_df[users_df['cluster_id'] == cluster_id]['user_id'].astype(str)
        cluster_interactions = interactions_df[
            interactions_df['user_id'].astype(str).isin(cluster_users)
        ].copy()
        cluster_data[cluster_id] = cluster_interactions
        logger.info(f"Cluster {cluster_id}: {len(cluster_users)} users, {len(cluster_interactions)} interactions")
    
    # Run evaluations
    if n_jobs == 1:
        # Sequential execution
        results_list = [
            _evaluate_single_cluster(
                cluster_id, cluster_interactions, content_df,
                algorithms, k_values, min_items_per_user, output_dir,
                embeddings_df, embedding_column
            )
            for cluster_id, cluster_interactions in cluster_data.items()
        ]
    else:
        # Parallel execution
        from joblib import Parallel, delayed
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_single_cluster)(
                cluster_id, cluster_interactions, content_df,
                algorithms, k_values, min_items_per_user, output_dir,
                embeddings_df, embedding_column
            )
            for cluster_id, cluster_interactions in cluster_data.items()
        )
    
    # Convert to dict, filtering out None results
    results = {cid: res for cid, res in results_list if res is not None}
    
    logger.info(f"Completed evaluation for {len(results)}/{len(cluster_ids)} clusters")
    
    return results
