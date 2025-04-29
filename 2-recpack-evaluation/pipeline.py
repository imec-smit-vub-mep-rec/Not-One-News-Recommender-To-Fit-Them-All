"""
Usage: python 1.pipeline.py <folder path>

Description: Run the recommendation pipeline analysis for a given cluster file.
"""

from recpack.pipelines import PipelineBuilder
from recpack.scenarios import WeakGeneralization, Timed, LastItemPrediction
from SentenceTransformerContentBased import SentenceTransformerContentBased
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
from recpack.pipelines import ALGORITHM_REGISTRY
import logging
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pathlib import Path

# Set more restrictive logging levels for all loggers
logger = logging.getLogger("recpack")
logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Disable other verbose loggers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)  # Root logger


def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist."""
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_pipeline_for_scenario(scenario, interaction_matrix, content_dict, proc):
    """Run the recommendation pipeline for a specific scenario."""
    scenario_name = scenario.__class__.__name__
    print(f"\nRunning scenario: {scenario_name}")

    scenario.split(interaction_matrix)

    # Set up pipeline
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)

    builder.add_algorithm('Popularity')
    builder.add_algorithm('SentenceTransformerContentBased', params={
        'content': content_dict,
        'language': 'intfloat/multilingual-e5-large',
        'metric': 'angular',
        'embedding_dim': 1024,
        'n_trees': 20,
        'num_neighbors': 100,
        'verbose': True,
    })
    builder.add_algorithm('ItemKNN', grid={
        'K': [50, 100, 200],
        'normalize_sim': [True, False],
        'normalize_X': [True, False]
    })
    builder.add_algorithm('EASE', grid={
        'l2': [1, 10, 100, 1000],
    })

    builder.set_optimisation_metric('NDCGK', K=100)
    builder.add_metric('NDCGK', K=[10, 20, 50])
    # builder.add_metric('CoverageK', K=[10, 20])

    # Build and run pipeline
    pipeline = builder.build()
    pipeline.run()

    # Get metrics and return results
    metrics = pipeline.get_metrics()
    return scenario_name, metrics, pipeline._metric_acc.acc


def prepare_data(cluster_file, articles_content_path):
    """Prepare data for pipeline analysis."""
    # Ensure input files exist
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Input file not found: {cluster_file}")
    if not os.path.exists(articles_content_path):
        raise FileNotFoundError(
            f"Articles content file not found: {articles_content_path}")

    # Load data
    df_original = pd.read_csv(cluster_file)

    # PREPROCESSING
    # Only keep interactions where article_id is not null
    df = df_original[df_original['article_id'].notna()]
    # Only keep interactions where user_id is not null
    df = df[df['user_id'].notna()]
    # Only keep interactions where impression_time is not null
    df = df[df['impression_time'].notna()]
    print(f"Processing {os.path.basename(cluster_file)}")
    print(
        f"Only keeping interactions where article_id is not null. Number of removed rows: {len(df_original) - len(df)}")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Number of items: {df['article_id'].nunique()}")

    proc = DataFramePreprocessor(
        item_ix='article_id', user_ix='user_id', timestamp_ix='impression_time'
    )
    proc.add_filter(MinItemsPerUser(
        5, item_ix='article_id', user_ix='user_id'))

    # Process the data
    interaction_matrix = proc.process(df)
    print("Interaction matrix shape:", interaction_matrix.shape)

    # Load articles content
    articles_content_df = pd.read_csv(articles_content_path)

    # Print the actual column names from the mappings to avoid errors
    print("Item mapping columns:", proc.item_id_mapping.columns.tolist())

    # Create content dictionary using internal IDs from RecPack
    # Map the original article IDs to the internal IDs used by RecPack
    content_dict = {}

    # Get the ID mapping - adjust column names based on actual dataframe structure
    try:
        if 'iid' in proc.item_id_mapping.columns and 'uid' in proc.item_id_mapping.columns:
            id_mapping = proc.item_id_mapping.set_index('iid')['uid'].to_dict()
        elif 'iid' in proc.item_id_mapping.columns and 'id' in proc.item_id_mapping.columns:
            id_mapping = proc.item_id_mapping.set_index('iid')['id'].to_dict()
        else:
            # Fallback: assume first column is original ID, second column is internal ID
            cols = proc.item_id_mapping.columns.tolist()
            id_mapping = proc.item_id_mapping.set_index(cols[0])[
                cols[1]].to_dict()
            print(f"Using columns {cols[0]} -> {cols[1]} for ID mapping")
    except Exception as e:
        print(f"Error creating ID mapping: {str(e)}")
        print("Mapping dataframe sample:")
        print(proc.item_id_mapping.head())
        # Fallback: direct mapping (assume original ID = internal ID)
        id_mapping = {
            i: i for i in articles_content_df['article_id'].unique() if not pd.isna(i)}
        print("Using direct 1:1 mapping as fallback")

    # Create content dictionary using internal IDs
    article_count = 0
    for _, row in articles_content_df.iterrows():
        orig_id = row['article_id']
        if orig_id in id_mapping:
            internal_id = id_mapping[orig_id]
            # Truncate content to reduce memory and prevent verbose printing
            content = row['content']
            if len(content) > 500:  # Limit content length to reduce verbosity
                content = content[:500]
            content_dict[internal_id] = content
            article_count += 1

    print(f"Loaded content for {article_count} articles")

    # Calculate timestamps for splits (needed for Timed scenario)
    t_validation = df['impression_time'].quantile(0.71)
    t_test = df['impression_time'].quantile(0.86)

    return proc, interaction_matrix, content_dict, t_validation, t_test


def process_results(proc, metrics, all_metrics, output_folder):
    """Process and save the results for a scenario."""
    # Ensure output folder exists
    ensure_dir_exists(output_folder)

    # Create subfolder for raw data if needed
    raw_folder = os.path.join(output_folder, "raw_data")
    ensure_dir_exists(raw_folder)

    # Write overall metrics to file
    metrics.to_csv(
        f'{output_folder}/overall_metrics.csv', index=False)

    # Print the results
    print("\nFinal metrics:")
    print(metrics)

    for metric in all_metrics.keys():
        print(f"\n{metric}:")
        for k in all_metrics[metric].keys():
            print(f"K={k}:")
            print(all_metrics[metric][k])

            # Limit metric name to 10 characters
            # Ensure metric is converted to string before slicing
            metric_key_str = str(metric)
            metric_name = metric_key_str[:10]

            try:
                # Determine join column for the left DataFrame (proc.user_id_mapping)
                if 'uid' in proc.user_id_mapping.columns:
                    join_col = 'uid'
                elif 'id' in proc.user_id_mapping.columns:
                    join_col = 'id'
                else:
                    # Assume second column is internal ID
                    join_col = proc.user_id_mapping.columns[1]

                results_df = all_metrics[metric][k].results

                # Determine how to merge based on results_df structure
                if 'user_id' in results_df.columns:
                    print(
                        f"Merging results_df on 'user_id' column with proc.user_id_mapping on '{join_col}' column")
                    results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_on='user_id',
                        how="left",
                        suffixes=('_ext', '_internal')
                    )
                elif results_df.index.name == 'user_id':
                    print(
                        f"Merging results_df on index with proc.user_id_mapping on '{join_col}' column")
                    results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_index=True,  # Merge on the right DataFrame's index
                        how="left",
                        suffixes=('_ext', '_internal')
                    )
                elif len(results_df.columns) > 0 and 'score' in results_df.columns:
                    # Heuristic: If 'user_id' is missing but 'score' is present, assume the first column is user ID
                    potential_user_col = results_df.columns[0]
                    print(
                        f"Warning: 'user_id' not found. Assuming first column '{potential_user_col}' is user ID for merge.")
                    results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_on=potential_user_col,
                        how="left",
                        suffixes=('_ext', '_internal')
                    )
                else:
                    print(
                        f"Error: Cannot determine user ID column or index in results_df for {metric_key_str} - {k}.")
                    print("Skipping merge for this combination.")
                    continue  # Skip this metric/k combination

                # --- Post-merge processing ---
                # Determine the actual name of the external user ID column after merge
                # Original external user ID column name
                user_id_col_ext_orig = proc.user_id_mapping.columns[0]
                user_id_col_ext_suffixed = user_id_col_ext_orig + '_ext'

                if user_id_col_ext_suffixed in results.columns:
                    user_id_col_to_use = user_id_col_ext_suffixed
                    print(
                        f"Using suffixed external user ID column: '{user_id_col_to_use}'")
                elif user_id_col_ext_orig in results.columns:
                    user_id_col_to_use = user_id_col_ext_orig
                    print(
                        f"Using original external user ID column: '{user_id_col_to_use}'")
                else:
                    print(
                        f"Error: External user ID column not found in merged results.")
                    print("Merged columns:", results.columns.tolist())
                    continue

                # Select external user ID and score
                if 'score' in results.columns:
                    results_final = results[[
                        user_id_col_to_use, 'score']].copy()
                # Fallback if 'score' is not found, assume last column is score if user ID is not the last column
                elif len(results.columns) > 1 and results.columns[-1] != user_id_col_to_use:
                    score_col = results.columns[-1]
                    print(
                        f"Warning: 'score' column not found. Assuming last column '{score_col}' is score.")
                    results_final = results[[
                        user_id_col_to_use, score_col]].copy()
                else:
                    print(
                        f"Error: Cannot determine score column in merged results for {metric_key_str} - {k}.")
                    # Create dummy score column
                    results_final = results[[user_id_col_to_use]].copy()
                    results_final['score'] = np.nan

                # Rename columns for consistency AFTER selection
                results_final.columns = ['user_id_ext', 'score']

                # Save to csv
                output_path = f'{output_folder}/{metric_name}_k{k}.csv'
                print(f"Saving results to: {output_path}")
                # Ensure parent directory exists
                ensure_dir_exists(os.path.dirname(output_path))
                results_final.to_csv(output_path, index=False)

            except Exception as e:
                print(
                    f"\n--- CRITICAL Error processing metric results for {metric_key_str} - {k} ---")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")

                # Print details only if results_df was defined
                if 'results_df' in locals():
                    print("results_df columns:", results_df.columns.tolist())
                    print("results_df index:", results_df.index)
                    print("results_df head:\n", results_df.head())
                else:
                    print("results_df was not defined before the error.")

                # Print traceback for detailed debugging
                import traceback
                traceback.print_exc()

                print("Attempting to save raw results instead...")
                try:
                    raw_output_path = f'{raw_folder}/{metric_name}_k{k}_raw.csv'
                    # Save with index to help debug raw data
                    ensure_dir_exists(os.path.dirname(raw_output_path))
                    all_metrics[metric][k].results.to_csv(
                        raw_output_path, index=True)
                    print(f"Saved raw results to: {raw_output_path}")
                except Exception as e_save:
                    print(f"Failed to save raw results: {str(e_save)}")


def load_cluster_data(folder):
    """Load all cluster files and create a mapping of user_ids to cluster numbers."""
    cluster_folder = os.path.join(folder, '1.input.clusters')
    if not os.path.exists(cluster_folder):
        print(f"Cluster data folder not found in {folder}")
        return None, None
    else:
        print(f"Cluster data folder found in {folder}")

    user_cluster_map = {}
    cluster_user_counts = {}  # Track original number of users per cluster

    # Process each cluster file
    for file in os.listdir(cluster_folder):
        if file.startswith('cluster_') and file.endswith('_merged.csv'):
            cluster_num = int(file.split('_')[1])
            df = pd.read_csv(os.path.join(
                cluster_folder, file), low_memory=False)

            # Count users in this cluster
            unique_users = df['user_id'].unique()
            cluster_user_counts[cluster_num] = len(unique_users)

            # Map each user to their cluster
            for user_id in unique_users:
                user_cluster_map[user_id] = cluster_num

    return user_cluster_map, cluster_user_counts


def analyze_scenario_results(base_folder, results_folder, scenario_name):
    """Analyze the results for a specific scenario."""
    print(f"\n=== Analyzing results for scenario: {scenario_name} ===")

    user_cluster_map, original_cluster_sizes = load_cluster_data(base_folder)

    if not user_cluster_map:
        print("No cluster data found. Please check the folder path.")
        return

    print(
        f"Found {len(user_cluster_map)} users across {len(set(user_cluster_map.values()))} clusters")

    # Print original cluster sizes
    print("\nOriginal cluster sizes (before preprocessing/splitting):")
    for cluster, count in sorted(original_cluster_sizes.items()):
        print(f"Cluster {cluster}: {count} users")

    # Create output directory for reports
    report_dir = os.path.join(results_folder, 'analysis')
    os.makedirs(report_dir, exist_ok=True)

    # Read overall metrics for reference if available
    overall_metrics_path = os.path.join(results_folder, 'overall_metrics.csv')
    if os.path.exists(overall_metrics_path):
        overall_metrics = pd.read_csv(overall_metrics_path, low_memory=False)
        print("\nOverall Metrics:")
        print(overall_metrics)

    # Get all results files
    result_files = [f for f in os.listdir(results_folder) if f.endswith(
        '.csv') and f != 'overall_metrics.csv']

    if not result_files:
        print("No result files found in the results folder.")
        return

    # Create a combined dataframe for all results
    all_data = []

    for file in result_files:
        print(f"\nProcessing file: {file}")

        # Parse filename components - improved to handle various formats
        parts = file.replace('.csv', '').split('_')

        # Try to extract algorithm, metric, and k value from filename
        try:
            # First get the algorithm and metric parts
            algorithm = parts[0]
            metric = parts[1] if len(parts) > 1 else "unknown"

            # For the k value, look for a part that starts with 'k' followed by digits
            k = None
            for part in parts[2:]:
                if part.startswith('k') and part[1:].isdigit():
                    k = int(part[1:])
                    break
                # Also try to handle format with just numbers at the end
                elif part.isdigit():
                    k = int(part)
                    break

            # If k wasn't found, look for K= in the algorithm name
            if k is None and '(K=' in algorithm:
                k_str = algorithm.split('(K=')[1].split(')')[0]
                k = int(k_str)

            # If still no k value found, use a default
            if k is None:
                print(
                    f"Warning: Couldn't extract k value from filename {file}, using default k=100")
                k = 100

            print(f"Algorithm: {algorithm}, Metric: {metric}, k: {k}")
        except Exception as e:
            print(f"Error parsing filename {file}: {e}")
            print("Skipping this file")
            continue

        # Read results
        try:
            file_path = os.path.join(results_folder, file)
            df = pd.read_csv(file_path, low_memory=False)

            # Remove all rows where score is undefined, NaN, or None
            df = df[df['score'].notna()]

            # Add metadata columns
            df['algorithm'] = algorithm
            df['metric'] = metric
            df['k'] = k

            # Check if this file has the expected structure
            if 'user_id_ext' not in df.columns or 'score' not in df.columns:
                print(
                    f"Skipping {file} - missing required columns (user_id_ext or score)")
                continue

            # Match user with cluster
            df['cluster'] = df['user_id_ext'].map(user_cluster_map)

            # Keep only rows where we could identify the cluster
            df = df.dropna(subset=['cluster'])
            df['cluster'] = df['cluster'].astype(int)

            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    # Combine all data
    if not all_data:
        print("No valid data found after processing.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Generate comprehensive reports
    performance_report = []

    # Unique combinations of algorithm, metric, k
    combinations = combined_df[['algorithm',
                                'metric', 'k']].drop_duplicates().values

    for algorithm, metric, k in combinations:
        subset = combined_df[(combined_df['algorithm'] == algorithm) &
                             (combined_df['metric'] == metric) &
                             (combined_df['k'] == k)]

        # Define functions for zero value analysis
        def count_zeros(x):
            return (x == 0).sum()

        def proportion_zeros(x):
            return (x == 0).sum() / len(x) if len(x) > 0 else 0

        # Analyze by cluster
        cluster_stats = []
        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster]

            # Calculate statistics
            avg_score = cluster_data['score'].mean()
            std_score = cluster_data['score'].std()
            zero_count = count_zeros(cluster_data['score'])
            zero_prop = proportion_zeros(cluster_data['score'])
            recpack_user_count = len(cluster_data)  # Users in predictions
            original_user_count = original_cluster_sizes.get(
                cluster, 0)  # Original users in cluster

            # Calculate coverage ratio
            coverage_ratio = recpack_user_count / \
                original_user_count if original_user_count > 0 else 0

            cluster_stats.append({
                'algorithm': algorithm,
                'metric': metric,
                'k': k,
                'cluster': cluster,
                'mean': avg_score,
                'std': std_score,
                'recpack_user_count': recpack_user_count,
                'original_user_count': original_user_count,
                'coverage_ratio': coverage_ratio,
                'zeros': zero_count,
                'zero_proportion': zero_prop
            })

        # Create cluster performance dataframe
        cluster_df = pd.DataFrame(cluster_stats)
        performance_report.append(cluster_df)

        # Print summary of this algorithm/metric/k
        print(f"\n=== {algorithm} - {metric} (k={k}) ===")
        print(cluster_df[['cluster', 'mean', 'zero_proportion',
              'coverage_ratio']].sort_values('mean', ascending=False))

    # Combine all performance data
    all_performance = pd.concat(performance_report, ignore_index=True)
    all_performance.to_csv(os.path.join(
        report_dir, 'all_cluster_performance.csv'), index=False)

    # Create summary visualizations

    # 1. General overview of average performance per algorithm per cluster (across all metrics)
    # Group by algorithm and cluster, averaging across metrics and k values
    bundled_performance = all_performance.groupby(['algorithm', 'cluster']).agg({
        'mean': 'mean',
        'zero_proportion': 'mean',
        'recpack_user_count': 'mean',
        'original_user_count': 'first'
    }).reset_index()

    # Calculate overall coverage ratio
    bundled_performance['coverage_ratio'] = bundled_performance['recpack_user_count'] / \
        bundled_performance['original_user_count']

    # Get unique clusters and algorithms for consistent ordering
    clusters = sorted(bundled_performance['cluster'].unique())
    algorithms = sorted(bundled_performance['algorithm'].unique())

    # Get average coverage ratio per cluster for title
    cluster_coverage = all_performance.groupby('cluster').agg({
        'recpack_user_count': 'mean',
        'original_user_count': 'first'
    })
    cluster_coverage['coverage_ratio'] = cluster_coverage['recpack_user_count'] / \
        cluster_coverage['original_user_count']
    avg_coverage = cluster_coverage['coverage_ratio']

    # --- OVERALL PERFORMANCE GRAPH ---
    plt.figure(figsize=(15, 10))

    # Set up the plot
    bar_width = 0.8 / len(algorithms)
    opacity = 0.8

    # Plot each algorithm's performance across clusters
    for i, algo in enumerate(algorithms):
        algo_data = bundled_performance[bundled_performance['algorithm'] == algo]
        positions = np.arange(len(clusters)) + (i * bar_width)

        # Get values for each cluster (filling with zeros for missing clusters)
        values = []
        for cluster in clusters:
            cluster_value = algo_data[algo_data['cluster'] == cluster]['mean']
            values.append(cluster_value.iloc[0] if len(
                cluster_value) > 0 else 0)

        plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

    # Add coverage information to x-tick labels
    x_tick_labels = []
    for cluster in clusters:
        coverage = avg_coverage.get(cluster, 0)
        x_tick_labels.append(f"{cluster}\n({coverage:.1%} coverage)")

    plt.xlabel('Cluster (with coverage ratio)')
    plt.ylabel('Average Score')
    plt.title(
        f'{scenario_name}: Average Performance by Algorithm Across Clusters (All Metrics)')
    plt.xticks(np.arange(len(clusters)) + bar_width *
               (len(algorithms) - 1) / 2, x_tick_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        report_dir, 'algorithm_performance_all_metrics.png'))
    plt.close()

    # --- OVERALL ZERO PROPORTION GRAPH ---
    plt.figure(figsize=(15, 10))

    # Plot each algorithm's zero proportion across clusters
    for i, algo in enumerate(algorithms):
        algo_data = bundled_performance[bundled_performance['algorithm'] == algo]
        positions = np.arange(len(clusters)) + (i * bar_width)

        # Get zero proportion values for each cluster
        values = []
        for cluster in clusters:
            cluster_value = algo_data[algo_data['cluster']
                                      == cluster]['zero_proportion']
            values.append(cluster_value.iloc[0] if len(
                cluster_value) > 0 else 0)

        plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

    plt.xlabel('Cluster (with coverage ratio)')
    plt.ylabel('Proportion of Zero Scores')
    plt.title(
        f'{scenario_name}: Zero Score Proportion by Algorithm Across Clusters (All Metrics)')
    plt.xticks(np.arange(len(clusters)) + bar_width *
               (len(algorithms) - 1) / 2, x_tick_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        report_dir, 'algorithm_zero_proportion_all_metrics.png'))
    plt.close()

    # 2. One overview per metric (bar chart)
    metrics = all_performance['metric'].unique()

    for metric in metrics:
        # Filter data for this metric
        metric_data = all_performance[all_performance['metric'] == metric]
        metric_avg = metric_data.groupby(['algorithm', 'cluster']).agg({
            'mean': 'mean',
            'zero_proportion': 'mean'
        }).reset_index()

        # --- PERFORMANCE GRAPH FOR THIS METRIC ---
        plt.figure(figsize=(15, 10))

        # Set up the plot for this metric
        bar_width = 0.8 / len(algorithms)
        opacity = 0.8

        # Plot each algorithm's performance across clusters
        for i, algo in enumerate(algorithms):
            algo_data = metric_avg[metric_avg['algorithm'] == algo]
            positions = np.arange(len(clusters)) + (i * bar_width)

            # Get values for each cluster
            values = []
            for cluster in clusters:
                cluster_value = algo_data[algo_data['cluster']
                                          == cluster]['mean']
                values.append(cluster_value.iloc[0] if len(
                    cluster_value) > 0 else 0)

            plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

        plt.xlabel('Cluster')
        plt.ylabel('Average Score')
        plt.title(
            f'{scenario_name}: Average Performance by Algorithm - {metric}')
        plt.xticks(np.arange(len(clusters)) + bar_width *
                   (len(algorithms) - 1) / 2, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            report_dir, f'algorithm_performance_{metric}.png'))
        plt.close()

        # --- ZERO PROPORTION GRAPH FOR THIS METRIC ---
        plt.figure(figsize=(15, 10))

        # Plot each algorithm's zero proportion across clusters
        for i, algo in enumerate(algorithms):
            algo_data = metric_avg[metric_avg['algorithm'] == algo]
            positions = np.arange(len(clusters)) + (i * bar_width)

            # Get zero proportion values for each cluster
            values = []
            for cluster in clusters:
                cluster_value = algo_data[algo_data['cluster']
                                          == cluster]['zero_proportion']
                values.append(cluster_value.iloc[0] if len(
                    cluster_value) > 0 else 0)

            plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

        plt.xlabel('Cluster')
        plt.ylabel('Proportion of Zero Scores')
        plt.title(
            f'{scenario_name}: Zero Score Proportion by Algorithm - {metric}')
        plt.xticks(np.arange(len(clusters)) + bar_width *
                   (len(algorithms) - 1) / 2, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            report_dir, f'algorithm_zero_proportion_{metric}.png'))
        plt.close()

    # 3. One graph for each K value with NDCGK metric
    # Filter for NDCGK metric
    ndcgk_data = all_performance[all_performance['metric'].str.contains(
        'NDCGK', case=False)]

    # Get unique K values
    k_values = sorted(ndcgk_data['k'].unique())

    for k_value in k_values:
        # Filter data for this K value
        k_data = ndcgk_data[ndcgk_data['k'] == k_value]
        k_avg = k_data.groupby(['algorithm', 'cluster']).agg({
            'mean': 'mean',
            'zero_proportion': 'mean'
        }).reset_index()

        # --- PERFORMANCE GRAPH FOR THIS K VALUE ---
        plt.figure(figsize=(15, 10))

        # Set up the plot for this K value
        bar_width = 0.8 / len(algorithms)
        opacity = 0.8

        # Plot each algorithm's performance across clusters
        for i, algo in enumerate(algorithms):
            algo_data = k_avg[k_avg['algorithm'] == algo]
            positions = np.arange(len(clusters)) + (i * bar_width)

            # Get values for each cluster
            values = []
            for cluster in clusters:
                cluster_value = algo_data[algo_data['cluster']
                                          == cluster]['mean']
                values.append(cluster_value.iloc[0] if len(
                    cluster_value) > 0 else 0)

            plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

        plt.xlabel('Cluster')
        plt.ylabel('NDCGK Score')
        plt.title(f'{scenario_name}: NDCGK@{k_value} Performance by Algorithm')
        plt.xticks(np.arange(len(clusters)) + bar_width *
                   (len(algorithms) - 1) / 2, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            report_dir, f'algorithm_performance_NDCGK_k{k_value}.png'))
        plt.close()

        # --- ZERO PROPORTION GRAPH FOR THIS K VALUE ---
        plt.figure(figsize=(15, 10))

        # Plot each algorithm's zero proportion across clusters
        for i, algo in enumerate(algorithms):
            algo_data = k_avg[k_avg['algorithm'] == algo]
            positions = np.arange(len(clusters)) + (i * bar_width)

            # Get zero proportion values for each cluster
            values = []
            for cluster in clusters:
                cluster_value = algo_data[algo_data['cluster']
                                          == cluster]['zero_proportion']
                values.append(cluster_value.iloc[0] if len(
                    cluster_value) > 0 else 0)

            plt.bar(positions, values, bar_width, alpha=opacity, label=algo)

        plt.xlabel('Cluster')
        plt.ylabel('Proportion of Zero Scores')
        plt.title(
            f'{scenario_name}: NDCGK@{k_value} Zero Score Proportion by Algorithm')
        plt.xticks(np.arange(len(clusters)) + bar_width *
                   (len(algorithms) - 1) / 2, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            report_dir, f'algorithm_zero_proportion_NDCGK_k{k_value}.png'))
        plt.close()

    # Save summary data
    bundled_performance.to_csv(os.path.join(
        report_dir, 'summary_performance.csv'), index=False)

    print(
        f"\nAnalysis for scenario {scenario_name} complete. Summary visualizations saved to {report_dir}")


def main():
    # Setup
    if len(sys.argv) < 2:
        print("Usage: python 1.pipeline.py <folder path>")
        sys.exit(1)

    folder = sys.argv[1]
    print(f"Processing {folder}")

    # Ensure the input folder exists
    if not os.path.exists(folder):
        print(f"Error: Input folder '{folder}' does not exist.")
        sys.exit(1)

    base_output_folder = f'{folder}/2.output.recpack_results'

    # Create base output directory if it doesn't exist
    ensure_dir_exists(base_output_folder)

    file_path = f'{folder}/interactions.csv'
    articles_content_path = f'{folder}/articles_content.csv'

    # Prepare data (only once)
    try:
        proc, interaction_matrix, content_dict, t_validation, t_test = prepare_data(
            file_path, articles_content_path)
        # Register the custom algorithm once
        ALGORITHM_REGISTRY.register(
            'SentenceTransformerContentBased', SentenceTransformerContentBased)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Define different scenarios to test
    SCENARIOS = [
        WeakGeneralization(
            frac_data_in=0.8,
            validation=True
        ),
        Timed(
            t=t_test,
            t_validation=t_validation,
            validation=True
        ),
        # Same data split as WeakGen?
        LastItemPrediction(
            validation=True,
            n_most_recent_in=30
        ),
    ]

    # Run each scenario separately and save results to scenario-specific subfolder
    for scenario in SCENARIOS:
        scenario_name, metrics, all_metrics = run_pipeline_for_scenario(
            scenario, interaction_matrix, content_dict, proc)

        # Create scenario-specific output folder
        scenario_output_folder = f'{base_output_folder}/{scenario_name}'
        ensure_dir_exists(scenario_output_folder)

        # Process and save results for this scenario
        process_results(proc, metrics, all_metrics, scenario_output_folder)

        # Analyze the scenario results
        analyze_scenario_results(folder, scenario_output_folder, scenario_name)


if __name__ == "__main__":
    main()
