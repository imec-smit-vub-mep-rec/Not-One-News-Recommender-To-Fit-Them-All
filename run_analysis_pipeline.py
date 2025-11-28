import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories(output_dir):
    """Creates necessary subdirectories for the pipeline."""
    dirs = [
        output_dir,
        os.path.join(output_dir, "1.input.clusters"),
        os.path.join(output_dir, "results"),
        os.path.join(output_dir, "clustering_artifacts") # For elbow plots, etc.
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def generate_interactions(behaviors_path, output_path):
    """Step 1: Convert behaviors.parquet to interactions.csv"""
    logger.info("STEP 1: Generating interactions.csv from behaviors...")
    
    try:
        df = pd.read_parquet(behaviors_path)
        initial_rows = len(df)
        
        # Filtering logic matches 3.behaviors_to_interactions.py
        df = df.dropna(subset=['user_id', 'article_id', 'impression_time'])
        df = df[~df['article_id'].isin(['homepage', ''])]
        
        # Select and clean columns
        df = df[['user_id', 'article_id', 'impression_time']]
        
        # Ensure timestamp format
        if not pd.api.types.is_integer_dtype(df['impression_time']):
             # specific check for datetime objects vs strings if needed
             pass 

        out_file = os.path.join(output_path, 'interactions.csv')
        df.to_csv(out_file, index=False)
        
        logger.info(f"✓ Generated interactions.csv ({len(df)} rows, kept {len(df)/initial_rows:.1%})")
        return df # Return for clustering usage if needed
    except Exception as e:
        logger.error(f"Failed to generate interactions: {e}")
        raise

def generate_content(articles_path, output_path):
    """Step 2: Convert articles.parquet to articles_content.csv"""
    logger.info("STEP 2: Generating articles_content.csv from articles...")
    
    try:
        articles = pd.read_parquet(articles_path)
        
        # Concatenate category and title for content-based methods
        # Matches logic in 4.articles_to_content.py
        texts = [
            f"query: {row.get('category_str', '')}: {row.get('title', '')}"
            for _, row in articles.iterrows()
        ]
        
        content_df = pd.DataFrame({
            'article_id': articles['article_id'],
            'content': texts
        })
        
        out_file = os.path.join(output_path, 'articles_content.csv')
        content_df.to_csv(out_file, index=False)
        logger.info(f"✓ Generated articles_content.csv ({len(content_df)} articles)")
        return articles # Return for clustering
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        raise

def run_clustering_logic(behaviors_path, articles_path, output_dir):
    """Step 3: Run User Clustering (Adapted from 2.user_clustering.py)"""
    logger.info("STEP 3: Running User Clustering...")
    
    # Note: We import logic or rewrite it here to avoid hardcoded paths in original script
    # For robustness, I am embedding the core logic here to ensure it uses dynamic paths
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    
    # 1. Load Data
    articles_df = pd.read_parquet(articles_path)
    behaviors_df = pd.read_parquet(behaviors_path)
    
    # 2. Merge Data
    behaviors_with_article = behaviors_df[behaviors_df['article_id'].notna()]
    behaviors_without_article = behaviors_df[behaviors_df['article_id'].isna()]
    
    merged_with_articles = behaviors_with_article.merge(
        articles_df[['article_id', 'category_str']], 
        on='article_id', 
        how='left'
    )
    merged_df = pd.concat([merged_with_articles, behaviors_without_article], ignore_index=True)
    
    # 3. Create User Features (Simplified aggregation for reliability)
    # Convert impression_time to datetime if needed
    if pd.api.types.is_numeric_dtype(merged_df['impression_time']):
         merged_df['datetime'] = pd.to_datetime(merged_df['impression_time'], unit='s') # Assuming s or ms
    
    user_table = merged_df.groupby('user_id').agg(
        count_sessions=('session_id', 'nunique'),
        count_total_impressions=('impression_id', 'count'),
        count_total_article_impressions=('article_id', lambda x: x.notna().sum()),
        count_total_unique_categories=('category_str', 'nunique'),
        avg_reading_time=('read_time', 'mean')
    ).reset_index()
    
    # 4. Prepare for Clustering
    features = ['count_sessions', 'count_total_impressions', 'count_total_unique_categories', 'avg_reading_time']
    X = user_table[features].copy()
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 5. Perform Clustering (Fixed K=5 for stability, or implement elbow)
    k = 5
    logger.info(f"Clustering users into {k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    user_table['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 6. Export Clusters
    cluster_output_dir = os.path.join(output_dir, "1.input.clusters")
    
    for cluster_id in sorted(user_table['cluster'].unique()):
        # Get users in this cluster
        cluster_users = user_table[user_table['cluster'] == cluster_id]['user_id']
        
        # Get all behaviors for users in this cluster (required format for RecPack pipeline)
        # The pipeline expects 'merged' csvs, but usually just needs interactions.
        # Based on pipeline.py: "df_original = pd.read_csv(cluster_file)"
        # It expects the interactions with user_id and article_id.
        
        cluster_data = merged_df[merged_df['user_id'].isin(cluster_users)]
        
        filename = f"cluster_{cluster_id}_merged.csv"
        filepath = os.path.join(cluster_output_dir, filename)
        cluster_data.to_csv(filepath, index=False)
        logger.info(f"  Exported Cluster {cluster_id}: {len(cluster_users)} users -> {filepath}")

    logger.info("✓ User Clustering complete.")

def run_recpack_pipeline(output_dir):
    """Step 4: Run the existing RecPack pipeline"""
    logger.info("STEP 4: Running RecPack Evaluation...")
    
    pipeline_script = os.path.join("2-recpack-evaluation", "pipeline.py")
    
    # The pipeline.py expects the folder path as an argument
    cmd = [sys.executable, pipeline_script, output_dir]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end='') # Stream output to console
            
    if p.returncode != 0:
        logger.error("RecPack pipeline failed.")
        sys.exit(1)
    
    logger.info("✓ RecPack Evaluation complete.")

def main():
    parser = argparse.ArgumentParser(description="Run complete RICON analysis pipeline.")
    parser.add_argument("--input-dir", help="Directory containing behaviors.parquet and articles.parquet (Optional if --s3-folder is provided)")
    parser.add_argument("--s3-folder", help="S3 folder URI (e.g., s3://my-bucket/data) containing inputs. If set, inputs are read from here and results are uploaded back.")
    parser.add_argument("--output-dir", required=True, help="Local directory to store temporary results and final outputs")
    
    args = parser.parse_args()
    
    if not args.input_dir and not args.s3_folder:
        parser.error("Either --input-dir or --s3-folder must be provided.")

    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time}")
    
    # Define Input Paths
    if args.s3_folder:
        # Ensure no trailing slash for consistency
        s3_base = args.s3_folder.rstrip('/')
        behaviors_file = f"{s3_base}/behaviors.parquet"
        articles_file = f"{s3_base}/articles.parquet"
        logger.info(f"Input (S3): {s3_base}")
    else:
        behaviors_file = os.path.join(args.input_dir, "behaviors.parquet")
        articles_file = os.path.join(args.input_dir, "articles.parquet")
        logger.info(f"Input (Local): {args.input_dir}")

    logger.info(f"Output (Local): {args.output_dir}")
    
    # 0. Checks
    def check_exists(path):
        if path.startswith("s3://"):
            try:
                import s3fs
                fs = s3fs.S3FileSystem()
                return fs.exists(path)
            except ImportError:
                logger.error("s3fs not installed. Run 'pip install s3fs'")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error checking S3 path {path}: {e}")
                return False
        return os.path.exists(path)

    if not check_exists(behaviors_file) or not check_exists(articles_file):
        logger.error(f"Missing input files. Ensure 'behaviors.parquet' and 'articles.parquet' exist at {behaviors_file} and {articles_file}")
        sys.exit(1)

    # 1. Setup
    setup_directories(args.output_dir)
    
    # 2. Preprocessing
    # Reads from S3/Local -> Writes to Local output_dir
    generate_interactions(behaviors_file, args.output_dir)
    generate_content(articles_file, args.output_dir)
    
    # 3. Clustering
    # Reads from S3/Local -> Writes to Local output_dir
    run_clustering_logic(behaviors_file, articles_file, args.output_dir)
    
    # 4. Evaluation
    # Runs on Local output_dir
    run_recpack_pipeline(args.output_dir)
    
    # 5. Upload Results if S3
    if args.s3_folder:
        logger.info("STEP 5: Uploading results to S3...")
        try:
            import s3fs
            fs = s3fs.S3FileSystem()
            # Define remote output path
            s3_output_path = f"{s3_base}/pipeline_output"
            logger.info(f"Uploading {args.output_dir} to {s3_output_path}...")
            fs.put(args.output_dir, s3_output_path, recursive=True)
            logger.info("✓ Upload complete.")
        except Exception as e:
            logger.error(f"Failed to upload results to S3: {e}")
    
    duration = datetime.now() - start_time
    logger.info(f"Pipeline completed successfully in {duration}.")
    if args.s3_folder:
         logger.info(f"Results available in S3: {s3_base}/pipeline_output")
    else:
         logger.info(f"Results available in: {args.output_dir}/results")

if __name__ == "__main__":
    main()
