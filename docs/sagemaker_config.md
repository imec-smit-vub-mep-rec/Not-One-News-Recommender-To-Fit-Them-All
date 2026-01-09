# SageMaker Configuration Guide

Optimal configurations for running the RICON pipeline efficiently on AWS SageMaker.

## Recommended Instance Types

### For Development/Small Datasets (<100k users)
```
Instance: ml.m5.xlarge
- vCPUs: 4
- Memory: 16 GB
- Cost: ~$0.23/hr
```

### For Medium Datasets (100k-1M users)
```
Instance: ml.m5.4xlarge
- vCPUs: 16
- Memory: 64 GB
- Cost: ~$0.92/hr
```

### For Large Datasets (>1M users) or Content-Based with GPU
```
Instance: ml.g4dn.xlarge (GPU)
- vCPUs: 4
- Memory: 16 GB
- GPU: 1x NVIDIA T4 (16 GB)
- Cost: ~$0.74/hr
- Best for: Sentence Transformer embeddings
```

### For Very Large Datasets with GPU
```
Instance: ml.g5.2xlarge (GPU)
- vCPUs: 8
- Memory: 32 GB
- GPU: 1x NVIDIA A10G (24 GB)
- Cost: ~$1.52/hr
- Best for: Large embedding models + clustering
```

## Environment Variables

Set these for optimal performance:

```bash
# Use all available CPUs for sklearn parallel operations
export SKLEARN_N_JOBS=-1

# Optimize BLAS/LAPACK for linear algebra
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Reduce memory fragmentation
export MALLOC_TRIM_THRESHOLD_=100000

# For GPU instances - optimize CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Pipeline Configuration for SageMaker

Create a SageMaker-optimized config file (`config_sagemaker.json`):

```json
{
  "dataset": {
    "name": "ebnerd_large",
    "type": "ebnerd",
    "input_dir": "/opt/ml/input/data/training"
  },
  "clustering": {
    "n_clusters": null,
    "k_selection_method": "elbow",
    "k_range": [2, 3, 4, 5, 6, 7, 8],
    "min_impressions_per_user": 5,
    "use_minibatch": true,
    "batch_size": 4096
  },
  "evaluation": {
    "algorithms": ["Popularity", "ItemKNN", "EASE"],
    "k_values": [10, 20],
    "min_items_per_user": 5
  },
  "output_dir": "/opt/ml/output"
}
```

### For GPU Instances (Content-Based)

```json
{
  "dataset": {
    "name": "ebnerd_large",
    "type": "ebnerd",
    "input_dir": "/opt/ml/input/data/training"
  },
  "clustering": {
    "n_clusters": 5,
    "use_minibatch": true,
    "batch_size": 8192
  },
  "evaluation": {
    "algorithms": ["Popularity", "ItemKNN", "CB-ST"],
    "k_values": [10, 20, 50],
    "content_based": {
      "model_name": "intfloat/multilingual-e5-large",
      "backend": "sklearn",
      "batch_size": 128
    }
  },
  "output_dir": "/opt/ml/output"
}
```

## SageMaker Processing Job Example

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# Create processor
processor = ScriptProcessor(
    image_uri='YOUR_ECR_IMAGE',  # Or use sagemaker sklearn container
    role='YOUR_SAGEMAKER_ROLE',
    instance_type='ml.m5.4xlarge',
    instance_count=1,
    command=['python3'],
    env={
        'SKLEARN_N_JOBS': '-1',
        'OMP_NUM_THREADS': '16',
    }
)

# Run processing job
processor.run(
    code='scripts/run_full_pipeline.py',
    inputs=[
        ProcessingInput(
            source='s3://your-bucket/ebnerd/',
            destination='/opt/ml/input/data/training'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/output',
            destination='s3://your-bucket/results/'
        )
    ],
    arguments=[
        '--config', '/opt/ml/input/data/training/config_sagemaker.json'
    ]
)
```

## Performance Tips

### 1. Data Loading
- Use **Parquet** format (already default) - faster than CSV
- Pre-combine behaviors files before uploading to S3:
  ```bash
  python scripts/combine_behaviors.py --input-dir data/ebnerd --splits train validation
  ```

### 2. Clustering
- **Auto-enabled**: MiniBatchKMeans activates automatically for >50k users
- Reduce `k_range` if you have a good estimate of clusters
- Set explicit `n_clusters` to skip elbow search entirely

### 3. Content-Based Recommendations
- Use GPU instances for Sentence Transformer embeddings
- Smaller models are faster: `sentence-transformers/all-MiniLM-L6-v2`
- Use `backend='sklearn'` (more reliable than annoy)

### 4. Memory Optimization
- Filter users with very few interactions early
- Use `--sample` flag for testing before full runs
- For very large datasets, consider chunked processing

### 5. Skip Expensive Operations
```bash
# Skip content-based if not needed (much faster)
python scripts/run_full_pipeline.py \
    --dataset ebnerd \
    --input-dir /path/to/data \
    --algorithms Popularity ItemKNN EASE \
    --n-clusters 5  # Skip elbow search
```

## Dockerfile for SageMaker

```dockerfile
FROM python:3.10-slim

WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV SKLEARN_N_JOBS=-1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/run_full_pipeline.py"]
```

## Cost Estimates

| Dataset Size | Instance | Runtime (est.) | Cost (est.) |
|-------------|----------|----------------|-------------|
| 10k users | ml.m5.large | 5-10 min | $0.02 |
| 100k users | ml.m5.xlarge | 20-40 min | $0.15 |
| 500k users | ml.m5.4xlarge | 1-2 hr | $1.50 |
| 1M+ users | ml.g4dn.xlarge | 2-4 hr | $3.00 |
| 1M+ users + CB | ml.g5.2xlarge | 3-5 hr | $7.00 |

*Estimates vary based on data characteristics and algorithm selection.*
