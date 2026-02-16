# RICON Analysis Pipeline

A modular framework for user clustering and recommendation evaluation in news recommendation systems.

## Overview

This project provides tools for:
1. **Data Conversion** - Converting various dataset formats (Adressa, EB-NeRD) to a standard format
2. **User Clustering** - Clustering users based on behavioral features (categories, time patterns, activity)
3. **Recommendation Evaluation** - Evaluating recommendation algorithms (Popularity, ItemKNN, EASE, MultVAE, Content-based) using RecPack

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd not-one-recommender-to-fit-them-all

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

- **RecPack**: Required for evaluation (`pip install recpack`)
- **Sentence Transformers**: Required for content-based recommendations (`pip install sentence-transformers`)
- **Annoy**: Required for approximate nearest neighbors (`pip install annoy`)

## Quick Start

### Full Pipeline

**Step 1:** (Optional) Combine EB-NeRD train + validation behaviors:
```bash
python scripts/combine_behaviors.py --input-dir data/ebnerd/ebnerd_large
```

**Step 2:** Generate embeddings for CB-ST (required, run once per dataset):
```bash
python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large
```

**Step 3:** Run the complete pipeline:
```bash
# Using a preset dataset configuration
python scripts/run_full_pipeline.py --dataset adressa --input-dir /path/to/adressa/data

# Using a custom configuration file
python scripts/run_full_pipeline.py --config my_config.json

# Legacy features with 4 clusters
python scripts/run_full_pipeline.py --dataset ebnerd --input-dir ./data/ebnerd/ebnerd_large --legacy-features --n-clusters 4

# With specific options
python scripts/run_full_pipeline.py --dataset ebnerd --input-dir /data/ebnerd/ebnerd_small --n-clusters 5
```

### Step-by-Step

#### 1. Data Conversion

Convert raw datasets to standard format:

```bash
python scripts/run_conversion.py --dataset adressa --input-dir /path/to/raw --output-dir ./converted
```

#### 2. User Clustering

Cluster users based on behavior:

```bash
python scripts/run_clustering.py \
    --impressions ./converted/impressions.parquet \
    --articles ./converted/articles.parquet \
    --output-dir ./clustering_results
```

#### 3. Evaluation

Evaluate recommendation algorithms:

```bash
python scripts/run_evaluation.py \
    --interactions ./converted/interactions.csv \
    --clusters ./clustering_results/user_clusters.csv \
    --content ./converted/articles_content.csv
```

## Project Structure

```
not-one-recommender-to-fit-them-all/
├── src/                          # Main source code
│   ├── config/                   # Configuration management
│   │   ├── settings.py           # Dataclass configs
│   │   └── schema.py             # Data schema definitions
│   ├── utils/                    # Utility functions
│   │   ├── io.py                 # File I/O helpers
│   │   ├── logging.py            # Logging setup
│   │   └── session.py            # Session management
│   ├── converters/               # Dataset converters
│   │   ├── base.py               # Base converter class
│   │   ├── adressa.py            # Adressa JSONL converter
│   │   ├── ebnerd.py             # EB-NeRD Parquet converter
│   │   └── generic.py            # Generic configurable converter
│   ├── preprocessing/            # Data preprocessing
│   │   ├── validators.py         # Data validation
│   │   ├── cleaners.py           # Data cleaning
│   │   └── transformers.py       # Data transformation
│   ├── clustering/               # User clustering
│   │   ├── feature_engineering.py  # Feature extraction
│   │   ├── clustering.py         # K-Means clustering
│   │   └── visualization.py      # Cluster visualization
│   └── evaluation/               # RecPack evaluation
│       ├── recpack_pipeline.py   # Evaluation pipeline
│       ├── analysis.py           # Results analysis
│       └── algorithms/           # Custom algorithms
│           └── content_based.py  # Content-based recommender
├── scripts/                      # CLI scripts
│   ├── run_full_pipeline.py      # Complete pipeline
│   ├── run_conversion.py         # Data conversion only
│   ├── run_clustering.py         # Clustering only
│   └── run_evaluation.py         # Evaluation only
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data Format

### Input Data Requirements

The converters expect specific file structures for each dataset type:

#### Adressa Dataset (`data/adressa/`)

```
data/adressa/
├── 20170101.jsonl       # Daily interaction files (JSONL format)
├── 20170102.jsonl       # One file per day
├── 20170103.jsonl
├── ...
└── 20170107.jsonl
```

**JSONL file format** (one JSON object per line):
```json
{"userId": "abc123", "time": 1483228800, "url": "https://www.adressa.no/nyheter/article123.html", "title": "Article Title", "id": "article123"}
```

Expected fields in each JSONL line:
| Field | Type | Description |
|-------|------|-------------|
| `userId` | string | Anonymous user identifier |
| `time` | int | Unix timestamp (seconds) |
| `url` | string | Full article URL |
| `title` | string | Article title (optional) |
| `id` | string | Article identifier (optional, extracted from URL if missing) |

**Notes:**
- Homepage views (`https://www.adressa.no/`) are filtered out
- Category is extracted from URL path (e.g., `/nyheter/` → `nyheter`)
- Sessions are detected using 30-minute inactivity threshold

#### EB-NeRD Dataset (`data/ebnerd/`)

```
data/ebnerd/
├── articles.parquet     # Article metadata
├── behaviors.parquet    # User behaviors/impressions
└── (optional subdirectories)
    ├── train/
    │   ├── articles.parquet
    │   └── behaviors.parquet
    └── validation/
        ├── articles.parquet
        └── behaviors.parquet
```

**articles.parquet** columns:
| Column | Type | Description |
|--------|------|-------------|
| `article_id` | int/string | Unique article identifier |
| `title` | string | Article title |
| `category` or `category_str` | string | Article category |
| `published_time` | datetime | Publication timestamp (optional) |
| `subtitle` | string | Article subtitle (optional) |
| `body` | string | Article body text (optional) |

**behaviors.parquet** columns:
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int/string | User identifier |
| `article_id` or `article_id_fixed` | int/string | Viewed article |
| `impression_time` | datetime/int | Interaction timestamp |
| `read_time` | float | Time spent reading (optional) |
| `scroll_percentage` | float | Scroll depth (optional) |
| `impression_id` | string | Unique impression identifier (optional) |

**Notes:**
- Parquet or CSV formats are supported
- The converter searches subdirectories if files aren't in root
- Column names are mapped automatically to standard format

### Standard Schema

The pipeline uses a standardized data format (see [general_data_format.md](general_data_format.md)):

**Articles** (`articles.parquet`):
- `article_id`: Unique article identifier
- `title`: Article title
- `category_str`: Article category

**Impressions** (`impressions.parquet`):
- `user_id`: Unique user identifier
- `article_id`: Article that was viewed
- `impression_time`: Unix timestamp (milliseconds)
- `session_id`: Session identifier

**Interactions** (`interactions.csv`):
- `user_id`: User identifier
- `article_id`: Article identifier
- `impression_time`: Unix timestamp (seconds)

## Configuration

Create a JSON configuration file:

```json
{
  "dataset": {
    "name": "my_dataset",
    "type": "generic",
    "input_dir": "/path/to/data",
    "column_mapping": {
      "user_id": "userId",
      "article_id": "itemId",
      "impression_time": "timestamp"
    }
  },
  "clustering": {
    "n_clusters": null,
    "k_selection_method": "elbow",
    "min_impressions_per_user": 5
  },
  "evaluation": {
    "algorithms": [
      {"name": "Popularity", "enabled": true},
      {"name": "ItemKNN", "enabled": true},
      {"name": "EASE", "enabled": true},
      {
        "name": "MultVAE",
        "enabled": false,
        "params": {
          "batch_size": 500,
          "max_epochs": 200,
          "learning_rate": 0.0001,
          "dim_bottleneck_layer": 200,
          "dim_hidden_layer": 600,
          "max_beta": 0.2,
          "anneal_steps": 200000,
          "dropout": 0.5,
          "validation_sample_size": 20000
        }
      },
      {"name": "CB-ST", "enabled": true}
    ],
    "k_values": [10, 20, 50]
  }
}
```

### Preset Configurations

- `ad`: Large AD dataset exported by Spark (S3 partitioned CSV)
- `hln`: HLN dataset, same S3 Spark CSV format as ad
- `vk`: VK dataset, same S3 Spark CSV format as ad
- `adressa`: Norwegian news dataset (Adressa)
- `ebnerd`: Danish news dataset (EB-NeRD/Ekstra Bladet)

## Running Large AD Data on SageMaker (S3)

This section covers the high-memory SageMaker workflow for the Spark-written AD dataset.

### Expected S3 Layout

```
s3://<bucket>/<prefix>/ad/
├── article_metadata.csv
└── impressions/
    ├── event_type=home_page_view/
    │   └── part-*.csv
    └── event_type=article_page_view/
        └── part-*.csv
```

The converter reads the partition root (`impressions/`) with `awswrangler` and automatically includes the `event_type` partition column.

### SageMaker Prerequisites

#### 1) IAM permissions

Your SageMaker execution role needs:
- `s3:ListBucket` on the dataset bucket/prefix
- `s3:GetObject` on dataset objects
- Optional: `s3:PutObject` if you sync results back to S3

#### 2) Instance sizing

For very large CSV exports, start with high-memory instances:
- Recommended start: `ml.r5.8xlarge` (256 GiB RAM)
- If out-of-memory: `ml.r5.16xlarge` (512 GiB RAM)
- EBS volume: 200-500 GB (depending on run artifacts)

#### 3) Install dependencies

```bash
pip install -r requirements.txt
```

`awswrangler` is required for S3 partitioned reads.

### Quick Run (Preset Mode)

```bash
python scripts/run_full_pipeline.py \
  --dataset ad \
  --input-dir s3://<bucket>/<prefix>/ad \
  --legacy-features \
  --n-clusters 4
```

### Reproducible Run (Config File Mode)

Create `config_ad_s3.json`:

```json
{
  "dataset": {
    "name": "ad",
    "input_path": "s3://<bucket>/<prefix>/ad",
    "format": "spark_csv",
    "event_types": ["home_page_view", "article_page_view"],
    "start_time_min": null,
    "start_time_max": null
  },
  "clustering": {
    "n_clusters": 4,
    "k_selection_method": "elbow",
    "legacy_features": true
  },
  "evaluation": {
    "k_values": [10, 20, 50]
  }
}
```

Run:

```bash
python scripts/run_full_pipeline.py --dataset ad --config config_ad_s3.json
```

```bash
# HLN dataset (same S3 layout as ad)
python scripts/run_full_pipeline.py --dataset hln --config config_hln.json --skip-evaluation

# VK dataset (same S3 layout as ad)
python scripts/run_full_pipeline.py --dataset vk --config config_vk.json
```

### Smoke Test Before Full Run (Recommended)

Start with a small run first:
1. Restrict to one partition via `event_types` (for example only `article_page_view`)
2. Restrict time range with `start_time_min` and `start_time_max`
3. Use fewer clusters
4. Skip evaluation initially:

```bash
python scripts/run_full_pipeline.py \
  --dataset ad \
  --input-dir s3://<bucket>/<prefix>/ad \
  --n-clusters 3 \
  --skip-evaluation
```
or skip clustering
```bash
python scripts/run_full_pipeline.py --config runs/ad_20260212_160426/config.json --skip-conversion --skip-clustering --content-mode embeddings --verbose
```

```bash
python scripts/run_full_pipeline.py --config runs/hln_20260216_110723/config.json --skip-conversion --skip-clustering --content-mode embeddings --verbose
```

Then remove limits for the full run.

### Embeddings for CB-ST on AD

AD includes `bert_embedding` in `article_metadata.csv`.
During conversion, the pipeline parses these embeddings and writes:

`runs/<run_id>/data/title_category_embeddings.parquet`

Evaluation uses this session-local embeddings file first, so no separate `generate_embeddings.py` step is required for AD.

### Outputs and Optional S3 Sync

By default outputs are written locally under:

`runs/<dataset>_<timestamp>/`

Optional sync back to S3:

```bash
aws s3 sync runs/ s3://<bucket>/<results-prefix>/runs/
```

## Algorithms

### Collaborative Filtering
- **Popularity**: Recommends most popular items
- **ItemKNN**: Item-based k-nearest neighbors
- **EASE**: Embarrassingly Shallow Autoencoders
- **MultVAE**: Variational autoencoder for collaborative filtering (RecPack implementation)

#### MultVAE notes

- `MultVAE` uses RecPack's `fit(X, validation_data=(validation_in, validation_out))` API.
- In this pipeline, sensible defaults are applied if not provided in config:
  - `predict_topK = max(k_values)`
  - `stop_early = true`
  - `max_iter_no_change = 5`
  - `stopping_criterion = "ndcg"`
  - `seed = evaluation seed`
- Override any of these in `evaluation.algorithms[].params`.

### Content-Based
- **CB-ST**: Sentence Transformer embeddings with Annoy approximate nearest neighbors

#### Pre-calculated Embeddings (REQUIRED for CB-ST)

CB-ST **requires** pre-calculated embeddings. Generate them before running the pipeline:

```bash
# Generate embeddings (run once per dataset)
python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large

# This creates: ./data/ebnerd/ebnerd_large/title_category_embeddings.parquet
```

The embedding format is `{category}: {title}` matching the legacy content-based approach.

**Generated file structure:**
```
data/ebnerd/ebnerd_large/
├── articles.parquet
├── behaviors.parquet
└── title_category_embeddings.parquet  # Generated embeddings (REQUIRED)
```

**Schema for `title_category_embeddings.parquet`:**
| Column | Type | Description |
|--------|------|-------------|
| `article_id` | Int32 | Article identifier |
| `embedding` | List[Float] | Embedding vector (1024-dim for e5-large) |

If the embeddings file is missing and CB-ST is enabled, the pipeline will throw an error with instructions.

#### Apple Silicon Warning ⚠️

CB-ST uses the **Annoy** backend by default for fast approximate nearest neighbor search. However, **Annoy has a known bug on Apple Silicon (M1/M2/M3/M4 Macs)** where it may only return 1 neighbor regardless of how many are requested.

If you experience poor CB-ST results on Apple Silicon, you can switch to the sklearn backend by modifying the algorithm configuration or using `CB-ST-sklearn` instead of `CB-ST`.

## Evaluation Methodology

### Scenario: LastItemPrediction

The pipeline uses RecPack's **LastItemPrediction** scenario (matching legacy behavior):
- For each user, the **last interaction** is held out for testing
- All **earlier interactions** are used for training
- This evaluates how well algorithms predict what a user will read next based on their history

### Metrics (Per-K)

For each configured value in `evaluation.k_values`, the pipeline reports:

- `NDCGK_<k>`, `RecallK_<k>`, `PrecisionK_<k>`: ranking quality at cutoff `k`
- `CoverageK_<k>`: catalog coverage at `k`, computed as unique recommended items divided by total available items
- `GiniK_<k>`: inequality of item exposure at `k`, computed from recommendation frequency across items
- `CoverageK_topics_<k>`, `GiniK_topics_<k>`: topic-level diversity (when `articles_cleaned.parquet` with `categories` is available)
  - Uses the `categories` column (array of strings); falls back to `category_str` if missing
  - Multi-topic items split exposure across all their categories
  - Unknown items (no valid categories) are excluded from topic metrics

Interpretation:
- Higher `CoverageK_<k>` means recommendations are spread over more of the catalog
- Lower `GiniK_<k>` means item exposure is more evenly distributed (less concentration on a few items)
- `CoverageK_topics_<k>`: fraction of topics (categories) that receive at least one recommendation
- `GiniK_topics_<k>`: inequality of topic exposure (lower = more even spread across topics)

### Data Filtering (Legacy Parity)

To match the legacy pipeline, the following filters are applied:

1. **Session Bot Filter**: Sessions with >50 interactions are removed (likely bots)
2. **Minimum User Activity**: Users with <5 article interactions are filtered for RecPack evaluation
3. **Empty Article Removal**: Impressions without valid article_id are removed for evaluation (but kept for clustering to capture homepage behavior)

The session filter can be disabled by passing `max_impressions_per_session=None` to the `DataCleaner`.

### Legacy vs Current Evaluation Design

The pipeline keeps legacy-compatible inputs and reporting, but the RecPack evaluation design is intentionally modernized in a few places:

- **Per-cluster training (current)**: models are trained and evaluated separately inside each cluster.
- **Post-hoc cluster slicing (legacy)**: a single global model is trained, then results are analyzed by cluster.
- **Hyperparameter search**: legacy used RecPack grid search for `ItemKNN` and `EASE`; current pipeline uses configured defaults.
- **Scenario coverage**: current pipeline focuses on `LastItemPrediction`; legacy scripts also contained `WeakGeneralization` and `Timed` experiments.
- **History cap**: legacy `LastItemPrediction` used `n_most_recent_in=30`; current pipeline uses all available user history unless you add a custom cap.

These differences mainly affect comparability of absolute metric values with old experiments. Cluster profiles and feature semantics remain aligned with the legacy clustering behavior.

## Output

Results are saved to timestamped session directories:

```
runs/
└── dataset_20241215_120000/
    ├── config.json           # Configuration used
    ├── articles.parquet      # Converted articles
    ├── impressions.parquet   # Converted impressions
    ├── interactions.csv      # RecPack interactions
    ├── articles_content.csv  # Content for CB algorithm
    ├── user_features.parquet # Extracted features
    ├── user_clusters.csv     # Cluster assignments
    ├── visualizations/       # Cluster plots
    │   ├── elbow_curve.png
    │   ├── cluster_distribution.png
    │   └── cluster_profiles.png
    ├── evaluation_results/   # Per-cluster results
    │   ├── cluster_0_results.csv
    │   ├── topic_report_cluster_0.csv  # Topic popularity (when articles with categories available)
    │   └── ...
    └── evaluation_report.txt # Summary report
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Adding a New Converter

1. Create a new file in `src/converters/`
2. Inherit from `BaseConverter`
3. Implement `convert_articles()` and `convert_impressions()`
4. Register in `src/converters/__init__.py`

```python
from .base import BaseConverter

class MyDatasetConverter(BaseConverter):
    def convert_articles(self):
        # Your conversion logic
        pass
    
    def convert_impressions(self):
        # Your conversion logic
        pass
```

## License

[Specify license]

## Citation

If you use this code, please cite:

```bibtex
@misc{ricon-analysis,
  title={RICON Analysis Pipeline},
  year={2024},
  url={repository-url}
}
```


## Command dump
```bash
# Step 0: Combine behaviors if needed (EB-NeRD only)
python scripts/combine_behaviors.py --input-dir data/ebnerd/ebnerd_large

# Step 1: Generate embeddings (REQUIRED for CB-ST) - run once per dataset
python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large

# Step 2: Run full pipeline
python scripts/run_full_pipeline.py --dataset ebnerd --input-dir ./data/ebnerd/ebnerd_large --legacy-features --n-clusters 4

# Small dataset test
python scripts/run_full_pipeline.py --dataset ebnerd --input-dir ./data/ebnerd/ebnerd_small --legacy-features

# Generate embeddings with different model
python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large --model intfloat/multilingual-e5-base

# Generate embeddings with GPU (larger batch size)
python scripts/generate_embeddings.py --input-dir ./data/ebnerd/ebnerd_large --batch-size 128
```

```bash
python scripts/generate_embeddings.py --input-dir ./data/adressa/one_week
python scripts/run_full_pipeline.py --dataset adressa --input-dir ./data/adressa/one_week --legacy-features --n-clusters 3
```