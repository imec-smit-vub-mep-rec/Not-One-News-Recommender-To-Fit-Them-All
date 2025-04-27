"""
Usage: python 2.analyze_results.py <folder path>

Description: Analyze algorithm performance across clusters, combining statistical analysis,
zero-score proportion analysis, and cross-algorithm comparisons.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
import itertools


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


def analyze_results(base_folder):
    """Analyze the results for each algorithm and metric across clusters."""
    results_folder = os.path.join(base_folder, '2.output.recpack_results')
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
    report_dir = os.path.join(base_folder, '3.output.mapping')
    os.makedirs(report_dir, exist_ok=True)

    # Read overall metrics for reference if available
    overall_metrics_path = os.path.join(results_folder, 'overall_metrics.csv')
    if os.path.exists(overall_metrics_path):
        overall_metrics = pd.read_csv(overall_metrics_path, low_memory=False)
        print("\nOverall Metrics:")
        print(overall_metrics)

    # Get all results files
    result_files = [f for f in os.listdir(
        results_folder) if f.endswith('.csv')]

    if not result_files:
        print("No result files found in the results folder.")
        return

    # Create a combined dataframe for all results
    all_data = []

    for file in result_files:
        print(f"\nProcessing file: {file}")

        # Skip overall_metrics.csv if it somehow ended up in the results folder
        if file == 'overall_metrics.csv':
            print("Skipping overall_metrics.csv as it's not a standard results file")
            continue

        # Parse filename components - improved to handle various formats
        parts = file.replace('.csv', '').split('_')

        # Try to extract algorithm, metric, and k value from filename
        try:
            # First get the algorithm and metric parts
            algorithm = parts[0]
            metric = parts[1]

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

    # Function to perform statistical significance tests between clusters
    def test_cluster_significance(data, algorithm, metric, k, output_file=None):
        """Test for statistical significance between clusters for a given algorithm/metric/k."""
        # Filter data for the specific algorithm/metric/k
        subset = data[(data['algorithm'] == algorithm) &
                      (data['metric'] == metric) &
                      (data['k'] == k)]

        clusters = sorted(subset['cluster'].unique())

        # Create a matrix to store p-values
        p_values = pd.DataFrame(index=clusters, columns=clusters)

        # For each pair of clusters, perform t-test
        for c1, c2 in itertools.combinations(clusters, 2):
            cluster1_scores = subset[subset['cluster'] == c1]['score']
            cluster2_scores = subset[subset['cluster'] == c2]['score']

            # Perform t-test if both clusters have enough data
            if len(cluster1_scores) > 1 and len(cluster2_scores) > 1:
                t_stat, p_val = stats.ttest_ind(cluster1_scores, cluster2_scores,
                                                equal_var=False)  # Welch's t-test
                p_values.loc[c1, c2] = p_val
                p_values.loc[c2, c1] = p_val
            else:
                p_values.loc[c1, c2] = None
                p_values.loc[c2, c1] = None

        # Fill diagonal with 1.0 (same cluster)
        for c in clusters:
            p_values.loc[c, c] = 1.0

        # Save to file if specified
        if output_file:
            p_values.to_csv(output_file)

        return p_values

    # Generate comprehensive reports
    performance_report = []

    # Unique combinations of algorithm, metric, k
    combinations = combined_df[['algorithm',
                                'metric', 'k']].drop_duplicates().values

    for algorithm, metric, k in combinations:
        subset = combined_df[(combined_df['algorithm'] == algorithm) &
                             (combined_df['metric'] == metric) &
                             (combined_df['k'] == k)]

        # Calculate metrics per cluster
        # Add custom aggregations for zero values
        def count_zeros(x):
            return (x == 0).sum()

        def proportion_zeros(x):
            return (x == 0).sum() / len(x) if len(x) > 0 else 0

        # Analyze by cluster
        cluster_stats = []
        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster]

            # Calculate statistics using the defined functions
            avg_score = cluster_data['score'].mean()
            std_score = cluster_data['score'].std()
            zero_count = count_zeros(cluster_data['score'])
            zero_prop = proportion_zeros(cluster_data['score'])
            recpack_user_count = len(cluster_data)  # Users in predictions
            original_user_count = original_cluster_sizes.get(
                cluster, 0)  # Original users in cluster

            # Calculate coverage ratio (what % of original users have predictions)
            coverage_ratio = recpack_user_count / \
                original_user_count if original_user_count > 0 else 0

            cluster_stats.append({
                'algorithm': algorithm,
                'metric': metric,
                'k': k,
                'cluster': cluster,
                'mean': avg_score,
                'std': std_score,
                'recpack_user_count': recpack_user_count,  # Users in predictions
                'original_user_count': original_user_count,  # Original cluster size
                'coverage_ratio': coverage_ratio,  # Proportion of users covered
                'zeros': zero_count,
                'zero_proportion': zero_prop
            })

        # Create cluster performance dataframe
        cluster_df = pd.DataFrame(cluster_stats)
        performance_report.append(cluster_df)

        # Print detailed analysis
        print(f"\n=== {algorithm} - {metric} (k={k}) ===")
        print(cluster_df.sort_values('mean', ascending=False))

        # Create visualizations - Style from script 2
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bars = ax.bar(cluster_df['cluster'], cluster_df['mean'])
        ax.errorbar(cluster_df['cluster'], cluster_df['mean'],
                    yerr=cluster_df['std'],
                    fmt='none', color='black', capsize=5)

        # Add annotations for zero values and user counts
        for i, row in cluster_df.iterrows():
            ax.annotate(f"Original: {int(row['original_user_count'])} users\n"
                        f"In RecPack: {int(row['recpack_user_count'])} ({row['coverage_ratio']:.1%})\n"
                        f"Zeros: {int(row['zeros'])} ({row['zero_proportion']:.2%})",
                        (row['cluster'], row['mean']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8)

        plt.title(f'{algorithm} - {metric} (k={k}) Performance by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Mean Score')

        # Save plot
        plot_path = os.path.join(
            report_dir, f'plot_{algorithm}_{metric}_k{k}.png')
        plt.savefig(plot_path)
        plt.close()

        # For calculating coverage stats per algorithm/metric
        coverage_stats = cluster_df.groupby('cluster').agg({
            'original_user_count': 'first',  # Original number of users in cluster
            'recpack_user_count': 'sum',     # Number of users with predictions
        }).reset_index()

        coverage_stats['coverage_ratio'] = coverage_stats['recpack_user_count'] / \
            coverage_stats['original_user_count']

        # Output coverage statistics
        print("\nUser coverage statistics:")
        print(coverage_stats)

        # Save coverage stats
        coverage_stats.to_csv(os.path.join(
            report_dir, f'coverage_{algorithm}_{metric}_k{k}.csv'), index=False)

        # Create visualizations - Style from script 3 (dual subplot)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot 1: Average Score by Cluster
        sns.barplot(x='cluster', y='mean', data=cluster_df, ax=ax1)
        ax1.set_title(
            f'{algorithm} - {metric} (k={k}): Average Score by Cluster')
        ax1.set_ylabel('Average Score')
        ax1.set_xlabel('Cluster')

        # Add annotation to first plot
        for i, row in cluster_df.iterrows():
            ax1.annotate(f"Original: {int(row['original_user_count'])}\n"
                         f"In RecPack: {int(row['recpack_user_count'])} ({row['coverage_ratio']:.1%})",
                         (i, row['mean']),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         fontsize=8)

        # Plot 2: Zero Proportion by Cluster
        sns.barplot(x='cluster', y='zero_proportion', data=cluster_df, ax=ax2)
        ax2.set_title(
            f'{algorithm} - {metric} (k={k}): Proportion of Zero Scores by Cluster')
        ax2.set_ylabel('Proportion of Zero Scores')
        ax2.set_xlabel('Cluster')

        # Add annotation to second plot
        for i, row in cluster_df.iterrows():
            ax2.annotate(f"Zeros: {int(row['zeros'])} / {int(row['recpack_user_count'])}",
                         (i, row['zero_proportion']),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         fontsize=8)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(
            report_dir, f'cluster_performance_{algorithm}_{metric}_k{k}.png'))
        plt.close()

        # Save detailed data
        detail_file = os.path.join(
            report_dir, f'detail_{algorithm}_{metric}_k{k}.csv')
        cluster_df.to_csv(detail_file, index=False)

        # Test for statistical significance between clusters
        try:
            p_values = test_cluster_significance(
                combined_df, algorithm, metric, k,
                output_file=os.path.join(
                    report_dir, f'pvalues_{algorithm}_{metric}_k{k}.csv')
            )

            # Create a heatmap of p-values
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(p_values, dtype=bool))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            sns.heatmap(p_values, mask=mask, cmap=cmap, annot=True,
                        vmin=0, vmax=0.05, center=0.025,
                        square=True, linewidths=.5, fmt='.4f')

            plt.title(
                f'P-values for {algorithm} - {metric} (k={k})\nClusters Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(
                report_dir, f'pvalues_heatmap_{algorithm}_{metric}_k{k}.png'))
            plt.close()

            # Add summary of significant differences to the printed output
            significant_pairs = []
            for c1, c2 in itertools.combinations(sorted(p_values.index), 2):
                p_val = p_values.loc[c1, c2]
                if p_val is not None and p_val < 0.05:
                    significant_pairs.append((c1, c2, p_val))

            if significant_pairs:
                print(
                    "\nStatistically significant differences between clusters (p < 0.05):")
                for c1, c2, p_val in significant_pairs:
                    print(f"  Cluster {c1} vs Cluster {c2}: p = {p_val:.4f}")
            else:
                print(
                    "\nNo statistically significant differences between clusters found.")

        except Exception as e:
            print(f"Error calculating statistical significance: {e}")
            import traceback
            traceback.print_exc()

    # Combine all performance data
    all_performance = pd.concat(performance_report, ignore_index=True)
    all_performance.to_csv(os.path.join(
        report_dir, 'all_cluster_performance.csv'), index=False)

    # Statistical analysis from script 2
    print("\nStatistical Analysis:")
    for algorithm in all_performance['algorithm'].unique():
        algorithm_data = all_performance[all_performance['algorithm'] == algorithm]

        for metric in algorithm_data['metric'].unique():
            print(f"\n{algorithm} - {metric}:")
            metric_data = algorithm_data[algorithm_data['metric'] == metric]

            # Identify best performing clusters
            best_clusters = metric_data.loc[metric_data.groupby('k')[
                'mean'].idxmax()]
            print("\nBest performing clusters:")
            print(best_clusters[['k', 'cluster', 'mean']])

            # Calculate relative performance
            metric_data['relative_performance'] = metric_data.groupby('k')['mean'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

            # Identify consistently good/bad clusters
            cluster_consistency = metric_data.groupby(
                'cluster')['relative_performance'].agg(['mean', 'std'])
            print("\nCluster consistency (higher mean = better performance):")
            print(cluster_consistency.sort_values('mean', ascending=False))

            # Save the consistency analysis
            consistency_file = os.path.join(
                report_dir, f'consistency_{algorithm}_{metric}.csv')
            cluster_consistency.to_csv(consistency_file)

    # Generate cross-algorithm comparison from script 3
    print("\n=== Cross-Algorithm Comparison ===")
    pivot_avg = all_performance.pivot_table(
        index=['cluster', 'metric'],
        columns='algorithm',
        values='mean',
        aggfunc='mean'
    )

    pivot_zero = all_performance.pivot_table(
        index=['cluster', 'metric'],
        columns='algorithm',
        values='zero_proportion',
        aggfunc='mean'
    )

    print("\nAverage Score by Cluster and Algorithm:")
    print(pivot_avg)

    print("\nZero Score Proportion by Cluster and Algorithm:")
    print(pivot_zero)

    # Save pivot tables
    pivot_avg.to_csv(os.path.join(report_dir, 'avg_score_comparison.csv'))
    pivot_zero.to_csv(os.path.join(
        report_dir, 'zero_proportion_comparison.csv'))

    # Generate summary figures - heatmaps
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_avg, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title('Average Score by Cluster and Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'avg_score_heatmap.png'))
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_zero, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title('Zero Score Proportion by Cluster and Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'zero_proportion_heatmap.png'))
    plt.close()

    # Generate bundled cluster performance report (averaging across metrics)
    print("\n=== Bundled Cluster Performance (Average Across Metrics) ===")

    # Group by algorithm and cluster, averaging across metrics and k values
    bundled_performance = all_performance.groupby(['algorithm', 'cluster']).agg({
        'mean': 'mean',
        'zero_proportion': 'mean',
        'recpack_user_count': 'mean',  # Changed from 'sum' to 'mean' to avoid overcounting
        # Take the first value as it should be the same for all
        'original_user_count': 'first'
    }).reset_index()

    # Calculate overall coverage ratio
    bundled_performance['coverage_ratio'] = bundled_performance['recpack_user_count'] / \
        bundled_performance['original_user_count']

    # Create a pivot table with algorithms as columns and clusters as rows
    bundled_pivot = bundled_performance.pivot_table(
        index='cluster',
        columns='algorithm',
        values='mean',
        aggfunc='mean'
    )

    bundled_zero_pivot = bundled_performance.pivot_table(
        index='cluster',
        columns='algorithm',
        values='zero_proportion',
        aggfunc='mean'
    )

    print("\nBundled Average Score by Cluster and Algorithm:")
    print(bundled_pivot)

    print("\nBundled Zero Proportion by Cluster and Algorithm:")
    print(bundled_zero_pivot)

    # Save bundled reports
    bundled_performance.to_csv(os.path.join(
        report_dir, 'bundled_cluster_performance.csv'), index=False)
    bundled_pivot.to_csv(os.path.join(
        report_dir, 'bundled_avg_score_comparison.csv'))
    bundled_zero_pivot.to_csv(os.path.join(
        report_dir, 'bundled_zero_proportion_comparison.csv'))

    # Generate bundled visualizations - heatmaps
    plt.figure(figsize=(14, 10))
    sns.heatmap(bundled_pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title('Bundled Average Score by Cluster and Algorithm (Across All Metrics)')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'bundled_avg_score_heatmap.png'))
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(bundled_zero_pivot, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title(
        'Bundled Zero Score Proportion by Cluster and Algorithm (Across All Metrics)')
    plt.tight_layout()
    plt.savefig(os.path.join(
        report_dir, 'bundled_zero_proportion_heatmap.png'))
    plt.close()

    # Create a bar chart showing average performance per algorithm per cluster
    plt.figure(figsize=(15, 10))

    # Get unique clusters and algorithms for consistent ordering
    clusters = sorted(bundled_performance['cluster'].unique())
    algorithms = sorted(bundled_performance['algorithm'].unique())

    # Get average coverage ratio per cluster for title
    # We'll compute this directly from the original data to avoid aggregation issues
    cluster_coverage = all_performance.groupby('cluster').agg({
        'recpack_user_count': 'mean',  # Use mean instead of sum
        # Original count should be the same for all rows of a cluster
        'original_user_count': 'first'
    })
    cluster_coverage['coverage_ratio'] = cluster_coverage['recpack_user_count'] / \
        cluster_coverage['original_user_count']
    avg_coverage = cluster_coverage['coverage_ratio']

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
    plt.title('Average Performance by Algorithm Across Clusters (All Metrics)')
    plt.xticks(np.arange(len(clusters)) + bar_width *
               (len(algorithms) - 1) / 2, x_tick_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        report_dir, 'bundled_algorithm_performance_bar.png'))
    plt.close()

    # Generate a coverage ratio heatmap
    plt.figure(figsize=(14, 10))
    # Create a new pivot table specifically for coverage ratio to avoid aggregation issues
    coverage_pivot = bundled_performance.pivot_table(
        index='cluster',
        columns='algorithm',
        values='coverage_ratio',
        aggfunc='mean'  # Mean is appropriate here since we're already working with ratios
    )
    sns.heatmap(coverage_pivot, annot=True, cmap="Greens", fmt=".1%")
    plt.title(
        'User Coverage Ratio by Cluster and Algorithm (% of Original Users with Predictions)')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'coverage_ratio_heatmap.png'))
    plt.close()

    print(f"\nAll report files saved to {report_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        base_folder = 'datasets/adressa'  # Default folder

    analyze_results(base_folder)
