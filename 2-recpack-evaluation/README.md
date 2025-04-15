# Create venv

python -m venv venv

# Activate venv

source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Workflow

## 1.pipeline.py

```bash
python 1.pipeline.py datasets/ekstra/large
```

For each dataset (Ekstra and Addressa), we have defined n user clusters.
Now we combine these clusters to create the original dataset with (user,item) pairs.
We use this combined dataset to train 3 common news recommendation algorithms: Popularity, ItemKNN, and EASE.

We then predict the rating for each (user,item) pair in the combined dataset using the 3 algorithms.
We evaluate the predictions using NDCG@K with k=10,20,50.

For each combination of dataset, algorithm, and k, we write the results to a csv file.

## 2.analyze_results.py

```bash
python 2.analyze_results.py datasets/ekstra/large
```

We load the results from the csv files and analyze them.
This creates a general overview file:

- cluster_performance.csv

## 3.cluster_results.py

```bash
python 3.cluster_results.py datasets/ekstra/large
```

We match the results to the original user clusters and write them to a csv file.
