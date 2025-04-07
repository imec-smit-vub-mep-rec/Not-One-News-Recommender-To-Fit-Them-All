# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Idea
1. Identify 6 user clusters in Ekstra dataset
2. Per cluster, create a new dataset with all the users in the cluster
3. Run a pipeline with 3 recommendation algorithms to compare on each of the 6 datasets
4. Compare the results: do the clusters have different preferences?