# Create venv

`python -m venv venv`

# Activate venv

`source venv/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# Workflow

## 1. Add the dataset to the datasets folder

Required files:

- Interaction matrix: interactions.csv
- Articles content: articles_content.csv
- User cluster: 1.input.clusters/cluster\_<number>\_merged.csv

Zipped example datasets with results can be found in the datasets folder.

## 2.pipeline.py

```bash
python pipeline.py datasets/ekstra/large
```

For each dataset (Ekstra and adressa), we have defined n user clusters.
We use the interaction matrix to train and run the predictions.
We use this combined dataset to train 4 simple recommendation algorithms: Popularity, ItemKNN, EASE and Content-based with sentence transformers.
After this, we evaluate the predictions using NDCG@K with k=10,20,50 and relate the performance and zero-prediction ratio to the user clusters.

We then predict the rating for each (user,item) pair in the combined dataset using the 3 algorithms.
We evaluate the predictions using NDCG@K with k=10,20,50.

For each combination of dataset, algorithm, and k, we write the results to a csv file.
The results are saved in the results folder within the corresponding dataset folder.

## Example folder structure after running the workflow

```
datasets/
├── ekstra-large/
│ ├── interactions.csv
│ ├── articles_content.csv
│ └── 1.input.clusters/
│ │ └── cluster_1_merged.csv
│ │ └── cluster_2_merged.csv
│ │ └── ...
│ └── results/
│ │ └── popularity/
│ │ └── itemknn/
│ │ └── ease/
│ │ └── content-based/
│ │ └── ...
├── adressa/
│ ├── interactions.csv
│ ├── articles_content.csv
│ └── 1.input.clusters/
│ │ └── cluster_1_merged.csv
│ │ └── cluster_2_merged.csv
│ │ └── ...
│ └── results/
│ │ └── popularity/
│ │ └── itemknn/
│ │ └── ease/
│ │ └── content-based/
│ │ └── ...