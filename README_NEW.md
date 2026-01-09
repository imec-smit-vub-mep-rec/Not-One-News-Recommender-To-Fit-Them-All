# RICON Analysis Pipeline

A modular framework for user clustering and recommendation evaluation in news recommendation systems.

## Overview

This project provides tools for:
1. **Data Conversion** - Converting various dataset formats (Adressa, EB-NeRD) to a standard format
2. **User Clustering** - Clustering users based on behavioral features (categories, time patterns, activity)
3. **Recommendation Evaluation** - Evaluating recommendation algorithms (Popularity, ItemKNN, EASE, Content-based) using RecPack

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

Run the complete pipeline with a single command:

```bash
# Using a preset dataset configuration
python scripts/run_full_pipeline.py --dataset adressa --input-dir /path/to/adressa/data

# Using a custom configuration file
python scripts/run_full_pipeline.py --config my_config.json

# With specific options
python scripts/run_full_pipeline.py --dataset ebnerd --input-dir /path/to/data --n-clusters 5
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
    "algorithms": ["Popularity", "ItemKNN", "EASE", "CB-ST"],
    "k_values": [10, 20, 50]
  }
}
```

### Preset Configurations

- `adressa`: Norwegian news dataset (Adressa)
- `ebnerd`: Danish news dataset (EB-NeRD/Ekstra Bladet)

## Algorithms

### Collaborative Filtering
- **Popularity**: Recommends most popular items
- **ItemKNN**: Item-based k-nearest neighbors
- **EASE**: Embarrassingly Shallow Autoencoders

### Content-Based
- **CB-ST**: Sentence Transformer embeddings with Annoy index

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
