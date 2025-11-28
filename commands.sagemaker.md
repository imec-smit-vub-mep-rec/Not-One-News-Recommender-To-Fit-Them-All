# Create dataset directory
mkdir -p datasets/ekstra-large

# Download EB-NeRD dataset files (replace URLs with actual download links)
wget -O datasets/ekstra-large/behaviors.parquet "https://recsys.eb.dk/download/behaviors.parquet"
wget -O datasets/ekstra-large/articles.parquet "https://recsys.eb.dk/download/articles.parquet"

# Or using curl
curl -L -o datasets/ekstra-large/behaviors.parquet "https://recsys.eb.dk/download/behaviors.parquet"
curl -L -o datasets/ekstra-large/articles.parquet "https://recsys.eb.dk/download/articles.parquet"

python run_analysis_pipeline.py --input-dir datasets/ekstra-small --output-dir output_experiment_1


# Datasets
* EB-NERD: https://docs.google.com/forms/d/e/1FAIpQLSdo6YZ1mVewLmqhsqqOjXTKsSp3OmCMHbMjEpsW0t_j-Hjtbg/formResponse