# Refactoring Plan: Streamlining the RICON Analysis Pipeline

## Overview

This document outlines a comprehensive plan to refactor the codebase into a modular, reusable, and scalable structure that can efficiently process different news recommendation datasets.

---

## Current Issues Identified

### 1. **Hardcoded Paths and Dataset Names**
- Dataset paths are hardcoded throughout scripts (e.g., `result_folder = 'ekstra-small'`)
- No centralized configuration management
- Input/output paths scattered across files

### 2. **Code Duplication**
- Similar data loading logic in multiple files
- Duplicate user/article statistics calculation in clustering and conversion scripts
- Category extraction logic repeated

### 3. **No Clear Data Contract**
- `general_data_format.md` exists but scripts don't strictly adhere to it
- Missing validation for input data formats
- Inconsistent column naming across datasets

### 4. **Poor Separation of Concerns**
- Data conversion, cleaning, and analysis mixed in single scripts
- Helper functions scattered in `helpers/` without clear organization
- No clear module boundaries

### 5. **Missing Error Handling & Logging**
- Limited error handling for missing files or invalid data
- Inconsistent logging across scripts
- No progress tracking for long-running operations

### 6. **No Session/Run Management**
- Results scattered across multiple folders
- No timestamp or run ID for reproducibility
- Difficult to track which parameters produced which results

---

## Target Architecture

```
ricon-analysis/
├── config/
│   ├── __init__.py
│   ├── settings.py              # Centralized configuration
│   └── schema.py                # Data schema definitions
├── converters/
│   ├── __init__.py
│   ├── base.py                  # Abstract base converter
│   ├── adressa.py               # Adressa-specific converter
│   ├── ebnerd.py                # EB-NeRD-specific converter
│   └── generic.py               # Generic CSV/Parquet converter
├── preprocessing/
│   ├── __init__.py
│   ├── cleaners.py              # Data cleaning functions
│   ├── validators.py            # Schema validation
│   └── transformers.py          # Data transformations
├── clustering/
│   ├── __init__.py
│   ├── feature_engineering.py   # User feature calculations
│   ├── clustering.py            # K-means clustering logic
│   └── visualization.py         # Clustering visualizations
├── evaluation/
│   ├── __init__.py
│   ├── recpack_pipeline.py      # RecPack evaluation pipeline
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── sentence_transformer.py
│   └── analysis.py              # Results analysis
├── utils/
│   ├── __init__.py
│   ├── io.py                    # File I/O utilities
│   ├── logging.py               # Logging configuration
│   └── session.py               # Session/run management
├── tests/
│   ├── __init__.py
│   ├── test_converters.py
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   └── test_evaluation.py
├── scripts/
│   ├── run_full_pipeline.py     # End-to-end pipeline
│   ├── run_conversion.py        # Data conversion only
│   ├── run_clustering.py        # Clustering only
│   └── run_evaluation.py        # Evaluation only
├── requirements.txt
├── setup.py
└── README.md
```

---

## Detailed TODO List

### Phase 1: Foundation & Configuration ✅

#### 1.1 Create Configuration Module
- [ ] Create `config/settings.py` with:
  - `DatasetConfig` dataclass (paths, column mappings, dataset-specific settings)
  - `ClusteringConfig` dataclass (features, n_clusters, scaler settings)
  - `EvaluationConfig` dataclass (algorithms, metrics, K values)
  - `SessionConfig` dataclass (output paths, timestamps, run IDs)
- [ ] Create `config/schema.py` with:
  - Pydantic/dataclass models for Articles, Impressions, Users schemas
  - Validation functions for data conformity
  - Column name constants matching `general_data_format.md`

#### 1.2 Create Session Management
- [ ] Create `utils/session.py`:
  - `Session` class that creates timestamped output folders
  - Format: `runs/{dataset_name}_{YYYYMMDD_HHMMSS}/`
  - Auto-saves configuration to `config.json` in session folder
  - Logging output to session folder

#### 1.3 Update General Data Format
- [ ] Review and finalize `general_data_format.md`:
  - Add `title` field to ARTICLES (required for content-based)
  - Ensure `sentiment_score` is optional with default 0.5
  - Document required vs optional fields clearly
  - Add data type specifications (int64, float32, etc.)

---

### Phase 2: Data Converters ✅

#### 2.1 Create Base Converter
- [ ] Create `converters/base.py`:
  ```python
  class BaseConverter(ABC):
      @abstractmethod
      def convert_articles(self, input_path: str) -> pd.DataFrame: ...
      @abstractmethod
      def convert_impressions(self, input_path: str) -> pd.DataFrame: ...
      def validate_output(self, df: pd.DataFrame, schema: str) -> bool: ...
      def save(self, df: pd.DataFrame, output_path: str, format: str): ...
  ```

#### 2.2 Refactor Adressa Converter
- [ ] Extract from `1.adressa_to_ekstra_format.py` into `converters/adressa.py`:
  - `AdressaConverter(BaseConverter)`
  - Move `extract_category_from_url()` to converter class
  - Add session ID detection logic as configurable timeout parameter
  - Support chunked processing with progress tracking
  - Add memory optimization (already has `gc.collect()`)

#### 2.3 Create EB-NeRD Converter
- [ ] Create `converters/ebnerd.py`:
  - `EBNeRDConverter(BaseConverter)`
  - Handle parquet input format
  - Map EB-NeRD columns to general format:
    - `article_id` → `article_id`
    - `category_str` → `category_str`
    - `title` → `title`
    - `published_time` → `time_published`
    - `sentiment_score` → `sentiment_score`

#### 2.4 Create Generic Converter
- [ ] Create `converters/generic.py`:
  - `GenericConverter(BaseConverter)`
  - Accept column mapping as config
  - Auto-detect file format (CSV, Parquet, JSON)

---

### Phase 3: Preprocessing Module ✅

#### 3.1 Create Validators
- [ ] Create `preprocessing/validators.py`:
  - `validate_articles(df)` - check required columns, data types
  - `validate_impressions(df)` - check required columns, data types
  - `validate_interactions(df)` - check RecPack-ready format
  - Return detailed validation report with warnings/errors

#### 3.2 Create Cleaners
- [ ] Create `preprocessing/cleaners.py`:
  - `remove_empty_articles(df)` - extract from `_adressa_remove_empty_articles.py`
  - `remove_invalid_sessions(df)` - sessions with < N impressions
  - `remove_outlier_users(df, min_impressions, max_impressions)`
  - `clean_categories(df)` - normalize category strings
  - `handle_missing_values(df, strategy)` - imputation strategies

#### 3.3 Create Transformers
- [ ] Create `preprocessing/transformers.py`:
  - `behaviors_to_interactions(df)` - extract from `3.behaviors_to_interactions.py`
  - `articles_to_content(df, template)` - extract from `4.articles_to_content.py`
  - `add_time_of_day_features(df)` - morning/afternoon/evening/night
  - `normalize_timestamps(df)` - ensure consistent timestamp format

---

### Phase 4: Clustering Module ✅

#### 4.1 Feature Engineering
- [ ] Create `clustering/feature_engineering.py`:
  - Extract user feature calculations from `2.user_clustering.py`:
    - `calculate_session_features(df)` - session count, duration, length
    - `calculate_reading_features(df)` - reading time, article vs homepage
    - `calculate_category_features(df)` - unique categories, switches
    - `calculate_time_of_day_features(df)` - proportions by time period
  - Create `UserFeatureEngineer` class with configurable feature list
  - Support incremental feature calculation for large datasets

#### 4.2 Clustering Logic
- [ ] Create `clustering/clustering.py`:
  - Extract from `2.user_clustering.py`:
    - `find_optimal_clusters(X, max_k)` - elbow method with kneed
    - `perform_kmeans(X, n_clusters)` - K-means clustering
  - Add additional clustering algorithms (optional):
    - `perform_hierarchical(X, n_clusters)`
    - `perform_dbscan(X, eps, min_samples)`
  - Add `ClusteringPipeline` class that orchestrates:
    1. Feature scaling (StandardScaler)
    2. Optimal K selection
    3. Clustering
    4. Cluster assignment

#### 4.3 Visualization
- [ ] Create `clustering/visualization.py`:
  - Extract from `2.user_clustering.py`:
    - `plot_elbow_curve(distortions, optimal_k)`
    - `plot_cluster_sizes(cluster_sizes, subscription_status)`
    - `plot_cluster_metrics(summary_df)`
    - `plot_radar_chart(cluster_means, features)`
    - `plot_category_heatmap(category_popularity)`
  - All plots save to session folder

---

### Phase 5: Evaluation Module ✅

#### 5.1 Refactor RecPack Pipeline
- [ ] Create `evaluation/recpack_pipeline.py`:
  - Extract from `pipeline.py`:
    - `prepare_recpack_data(interactions_df, articles_content_df)`
    - `run_scenario(scenario, interaction_matrix, content_dict, algorithms)`
    - `process_results(metrics, all_metrics, user_mapping)`
  - Create `RecPackPipeline` class with configurable:
    - Algorithms (Popularity, ItemKNN, EASE, Content-Based)
    - Scenarios (WeakGeneralization, Timed, LastItemPrediction)
    - Metrics (NDCGK with configurable K values)
  - Support for adding custom algorithms via registry

#### 5.2 Move SentenceTransformer Algorithm
- [ ] Move `SentenceTransformerContentBased.py` to `evaluation/algorithms/sentence_transformer.py`
- [ ] Ensure it's properly integrated with RecPack's algorithm registry
- [ ] Add configuration for:
  - Model name (default: 'intfloat/multilingual-e5-large')
  - Embedding dimension
  - Number of trees
  - Number of neighbors

#### 5.3 Results Analysis
- [ ] Create `evaluation/analysis.py`:
  - Extract from `pipeline.py`:
    - `load_cluster_data(cluster_folder)`
    - `analyze_scenario_results(results_folder, user_cluster_map)`
    - `generate_performance_report(all_performance)`
  - Add statistical significance testing from `tests/significance/index.py`:
    - `run_anova(algorithm_scores)`
    - `run_tukey_hsd(algorithm_scores)`

---

### Phase 6: Unified Pipeline Scripts ✅

#### 6.1 Create Main Pipeline Script
- [ ] Create `scripts/run_full_pipeline.py`:
  ```python
  def main(dataset_name: str, config_path: str = None):
      # 1. Load config or use defaults
      # 2. Initialize session
      # 3. Convert raw data to general format
      # 4. Validate and clean data
      # 5. Run user clustering
      # 6. Export cluster data
      # 7. Run RecPack evaluation
      # 8. Generate reports
      # 9. Save session summary
  ```

#### 6.2 Create Individual Pipeline Scripts
- [ ] Create `scripts/run_conversion.py`:
  - CLI: `python run_conversion.py --dataset adressa --input ./raw --output ./processed`
  - Support for specifying converter type
- [ ] Create `scripts/run_clustering.py`:
  - CLI: `python run_clustering.py --input ./processed --n_clusters auto`
  - Option to specify features, clustering method
- [ ] Create `scripts/run_evaluation.py`:
  - CLI: `python run_evaluation.py --input ./clusters --algorithms pop,knn,ease,cb`
  - Option to specify scenarios, metrics, K values

---

### Phase 7: Testing ✅

#### 7.1 Unit Tests
- [ ] Create `tests/test_converters.py`:
  - Test Adressa conversion with sample data
  - Test EB-NeRD conversion with sample data
  - Test column mapping validation
- [ ] Create `tests/test_preprocessing.py`:
  - Test validators with valid/invalid data
  - Test cleaners preserve data integrity
  - Test transformers produce expected output
- [ ] Create `tests/test_clustering.py`:
  - Test feature engineering calculations
  - Test clustering produces valid assignments
  - Test elbow method finds reasonable K
- [ ] Create `tests/test_evaluation.py`:
  - Keep and expand existing `test_sentence_transformer_content_based.py`
  - Test RecPack pipeline integration

#### 7.2 Integration Tests
- [ ] Create sample test datasets (small subsets of real data)
- [ ] Create end-to-end test for full pipeline
- [ ] Add CI/CD configuration (GitHub Actions)

---

### Phase 8: Documentation & Cleanup ✅

#### 8.1 Documentation
- [ ] Update main `README.md` with:
  - Installation instructions
  - Quick start guide
  - Full pipeline usage
  - Individual script usage
  - Configuration options
- [ ] Add docstrings to all public functions and classes
- [ ] Create `docs/` folder with:
  - `data_format.md` (updated from `general_data_format.md`)
  - `configuration.md`
  - `extending.md` (adding new datasets/algorithms)

#### 8.2 Cleanup
- [ ] Remove old numbered scripts after migration
- [ ] Remove duplicate/unused helper files
- [ ] Consolidate requirements.txt files
- [ ] Add `setup.py` or `pyproject.toml` for package installation

---

## Implementation Priority

### High Priority (Do First)
1. **Phase 1.1-1.2**: Configuration and session management - foundation for everything
2. **Phase 2.1-2.2**: Base converter and Adressa refactoring - preserves existing functionality
3. **Phase 4.1-4.2**: Feature engineering and clustering - core analysis logic
4. **Phase 5.1**: RecPack pipeline refactoring - enables evaluation

### Medium Priority (Do Second)
5. **Phase 3**: Preprocessing module - improves data quality
6. **Phase 2.3-2.4**: Additional converters - extensibility
7. **Phase 6**: Pipeline scripts - usability

### Lower Priority (Do Last)
8. **Phase 7**: Testing - ensures reliability
9. **Phase 8**: Documentation - maintainability
10. **Phase 5.3**: Statistical analysis - research completeness

---

## Sagemaker Optimization Notes

For running on Sagemaker with large datasets:

1. **Memory Management**:
   - Use chunked processing for conversions (already implemented in Adressa)
   - Use `gc.collect()` after processing large chunks
   - Consider Dask for very large datasets

2. **Data Storage**:
   - Use Parquet format with compression (`gzip` or `snappy`)
   - Store intermediate results to prevent re-computation

3. **Parallelization**:
   - Feature engineering can be parallelized per user
   - RecPack scenarios can run in parallel
   - Use `joblib` for parallel processing

4. **Progress Tracking**:
   - Use `tqdm` for progress bars
   - Log progress to CloudWatch
   - Implement checkpointing for long runs

---

## Migration Strategy

1. **Create new structure alongside old code**
   - Don't break existing scripts during development
   - Import and wrap existing functions initially

2. **Test with small dataset first**
   - Use `ekstra-small` or create test subset
   - Verify identical results before proceeding

3. **Migrate incrementally**
   - One module at a time
   - Run both old and new code, compare results

4. **Deprecate old scripts last**
   - Add deprecation warnings to old scripts
   - Remove only after full verification

---

## Success Criteria

The refactoring is complete when:

- [ ] Running `python scripts/run_full_pipeline.py --dataset adressa` produces identical results to current workflow
- [ ] Adding a new dataset requires only creating a new converter class
- [ ] All results are stored in timestamped session folders
- [ ] Tests pass with >80% code coverage
- [ ] README provides clear usage instructions
- [ ] Code runs successfully on Sagemaker with large datasets
