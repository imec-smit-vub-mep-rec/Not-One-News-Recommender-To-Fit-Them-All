from recpack.pipelines import PipelineBuilder
from recpack.scenarios import StrongGeneralizationTimed
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
import pandas as pd
import numpy as np
from scipy import stats
import os


def run_pipeline_analysis(cluster_file):
    """Run the recommendation pipeline analysis for a given cluster file."""
    # Load data
    df_original = pd.read_csv(cluster_file)

    # Only keep interactions where article_id is not null
    df = df_original[df_original['article_id'].notna()]
    print(f"Processing {os.path.basename(cluster_file)}")
    print(
        f"Only keeping interactions where article_id is not null. Number of removed rows: {len(df_original) - len(df)}")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Number of items: {df['article_id'].nunique()}")

    # Convert impression_time to datetime and then to Unix timestamp (seconds)
    df['impression_time'] = pd.to_datetime(df['impression_time'])
    df['impression_time'] = df['impression_time'].astype(np.int64) // 10**9

    proc = DataFramePreprocessor(
        item_ix='article_id', user_ix='user_id', timestamp_ix='impression_time')
    proc.add_filter(MinUsersPerItem(
        5, item_ix='article_id', user_ix='user_id'))
    proc.add_filter(MinItemsPerUser(
        5, item_ix='article_id', user_ix='user_id'))

    # Process the data
    interaction_matrix = proc.process(df)
    print("Interaction matrix shape:", interaction_matrix.shape)

    # Calculate timestamps for splits
    t_validation = df['impression_time'].quantile(0.8)
    t_test = df['impression_time'].quantile(0.9)

    scenario = StrongGeneralizationTimed(
        frac_users_in=0.8,
        t=t_test,
        t_validation=t_validation,
        validation=True
    )
    scenario.split(interaction_matrix)

    # Set up pipeline
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)

    builder.add_algorithm('Popularity')
    builder.add_algorithm('EASE', grid={
        'l2': [0.01, 0.1, 0.5],
        'alpha': [0.1, 0.5, 1.0],
        'density': [0.01, 0.05, 0.1]
    })
    builder.add_algorithm('ItemKNN', grid={
        'K': [50, 100, 200],
        'normalize_sim': [True]
    })

    builder.set_optimisation_metric('NDCGK', K=10)
    builder.add_metric('NDCGK', K=[10, 20, 50])
    builder.add_metric('CoverageK', K=[10, 20])

    # Build and run pipeline
    pipeline = builder.build()
    pipeline.run()

    return pipeline.get_metrics()


def perform_statistical_analysis(all_metrics):
    """Perform statistical analysis on the results."""
    # Extract metrics for each cluster and algorithm
    results = {
        'NDCGK_10': {},
        'NDCGK_20': {},
        'NDCGK_50': {},
        'CoverageK_10': {},
        'CoverageK_20': {}
    }

    # Organize data by metric and algorithm
    for cluster, metrics in all_metrics.items():
        for algorithm in metrics.index:
            for metric in results.keys():
                if algorithm not in results[metric]:
                    results[metric][algorithm] = {}
                results[metric][algorithm][cluster] = metrics.loc[algorithm, metric]

    statistical_results = []
    statistical_results.append("=== Comprehensive Statistical Analysis ===\n")

    # Analyze each metric
    for metric in results.keys():
        statistical_results.append(f"\n=== {metric} Analysis ===")

        for algorithm in results[metric].keys():
            scores = results[metric][algorithm]
            scores_list = list(scores.values())
            clusters = list(scores.keys())

            statistical_results.append(f"\nAlgorithm: {algorithm}")
            statistical_results.append(
                f"Number of clusters analyzed: {len(scores)}")

            # Calculate mean performance for each cluster
            mean_scores = {cluster: score for cluster, score in scores.items()}

            # Sort clusters by performance
            sorted_clusters = sorted(
                mean_scores.items(), key=lambda x: x[1], reverse=True)

            # Report performance ranking
            statistical_results.append("\nPerformance Ranking:")
            for cluster, score in sorted_clusters:
                statistical_results.append(f"{cluster}: {score:.4f}")

            # Calculate overall mean and std
            mean = np.mean(scores_list)
            std = np.std(scores_list)

            # Identify significantly different clusters (beyond 1 standard deviation)
            outperformers = [c for c, s in scores.items() if s > mean + std]
            underperformers = [c for c, s in scores.items() if s < mean - std]

            statistical_results.append(f"\nMean: {mean:.4f}")
            statistical_results.append(f"Standard Deviation: {std:.4f}")

            if outperformers:
                statistical_results.append(
                    f"Significantly outperforming clusters (>1 std): {', '.join(outperformers)}")
            if underperformers:
                statistical_results.append(
                    f"Significantly underperforming clusters (<1 std): {', '.join(underperformers)}")

            # Perform ANOVA if we have enough clusters
            if len(scores) >= 2:
                f_stat, p_value = stats.f_oneway(
                    *[[score] for score in scores.values()])
                statistical_results.append(f"\nOne-way ANOVA:")
                statistical_results.append(f"F-statistic: {f_stat:.4f}")
                statistical_results.append(f"p-value: {p_value:.4f}")
                if p_value < 0.05:
                    statistical_results.append(
                        "Result: Significant differences found between clusters (p < 0.05)")
                else:
                    statistical_results.append(
                        "Result: No significant differences found between clusters (p >= 0.05)")

            statistical_results.append("\n" + "="*50)

    return "\n".join(statistical_results)


def main():
    # Setup
    folder = 'datasets/ekstra/large/0407'
    print(f"Processing {folder}")

    # Process all clusters
    all_metrics = {}
    for i in range(0, 4):
        cluster_file = f'{folder}/cluster_{i}_merged.csv'
        print(f"Processing {cluster_file}")
        if os.path.exists(cluster_file):
            metrics = run_pipeline_analysis(cluster_file)
            all_metrics[f'cluster_{i}'] = metrics

    # Perform statistical analysis
    statistical_results = perform_statistical_analysis(all_metrics)

    # Write results to file
    output_file = f'{folder}/recpack_results.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("=== Detailed Results ===\n\n")
        for cluster, metrics in all_metrics.items():
            f.write(f"\n{cluster}:\n")
            f.write(str(metrics))
            f.write("\n" + "="*50 + "\n")

        f.write("\n=== Statistical Analysis ===\n\n")
        f.write(statistical_results)


if __name__ == "__main__":
    main()
