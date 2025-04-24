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


def run_pipeline_analysis(cluster_file, articles_content_path):
    """Run the recommendation pipeline analysis for a given cluster file."""
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

    # Convert impression_time to datetime and then to Unix timestamp (seconds)
    # df['impression_time'] = pd.to_datetime(df['impression_time'])
    # df['impression_time'] = df['impression_time'].astype(np.int64) // 10**9

    proc = DataFramePreprocessor(
        item_ix='article_id', user_ix='user_id', timestamp_ix='impression_time'
    )
    # proc.add_filter(MinUsersPerItem(
    #     5, item_ix='article_id', user_ix='user_id'))
    proc.add_filter(MinItemsPerUser(
        5, item_ix='article_id', user_ix='user_id'))

    # Process the data
    interaction_matrix = proc.process(df)
    print("Interaction matrix shape:", interaction_matrix.shape)

    # Calculate timestamps for splits
    t_validation = df['impression_time'].quantile(0.71)
    t_test = df['impression_time'].quantile(0.86)

    # scenario = WeakGeneralization(
    #     frac_data_in=0.8,
    #     validation=True
    # )
    # scenario = Timed(
    #     t=t_test,
    #     t_validation=t_validation,
    #     validation=True
    # )
    scenario = LastItemPrediction(
        validation=True,
        # How much of the historic events to use as input history. Defaults to the maximal integer value.
        n_most_recent_in=10
    )

    articles_content = pd.read_csv(articles_content_path)
    articles_content = articles_content.set_index('article_id')
    articles_content = articles_content['content'].to_dict()

    ALGORITHM_REGISTRY.register(
        'SentenceTransformerContentBased', SentenceTransformerContentBased)

    scenario.split(interaction_matrix)

    # Set up pipeline
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)

    builder.add_algorithm('Popularity')
    builder.add_algorithm('SentenceTransformerContentBased', params={
        'content': articles_content,
        'language': 'intfloat/multilingual-e5-large',
        'metric': 'dot',
        'embedding_dim': 1024,
        'n_trees': 10,
        'num_neighbors': 30,
        'user_offset_factor': 1
    })
    builder.add_algorithm('ItemKNN', grid={
        'K': [50, 100, 200],
        'normalize_sim': [True, False],
        'normalize_X': [True, False]
    })

    builder.set_optimisation_metric('NDCGK', K=100)
    builder.add_metric('NDCGK', K=[10, 20, 50])
    # builder.add_metric('CoverageK', K=[10, 20])

    # Build and run pipeline
    pipeline = builder.build()
    pipeline.run()

    # Details are present in pipeline
    # .acc --> details: call .results on metric objects (nested under metric name and k)
    # 1. user_ids score (internal user_ids)
    # 2. Convert back to normal uids: proc.user_id_mapping (uid is internal user_ids)
    # 3. Left join on original df to link results to original users: proc.user_id_mapping.join(ndcg20_itemknn, left_on='usd', right_on='user_id', how='left')
    # .metrics --> Aggregated results

    metrics = pipeline.get_metrics()

    return proc, metrics, pipeline._metric_acc.acc


def main():
    # Setup
    folder = sys.argv[1]
    print(f"Processing {folder}")
    output_folder = f'{folder}/2.output.recpack_results'

    file_path = f'{folder}/interactions.csv'

    articles_content_path = f'{folder}/articles_content.csv'

    # Run the pipeline analysis on the combined file
    proc, metrics, all_metrics = run_pipeline_analysis(
        file_path, articles_content_path)

    # Write overall metrics to file
    metrics.to_csv(
        f'{output_folder}/overall_metrics.csv', index=False)

    # Print the results
    print("Proc mapping:")
    print(proc.user_id_mapping)
    for metric in all_metrics.keys():
        print(f"\n{metric}:")
        for k in all_metrics[metric].keys():
            print(f"K={k}:")
            print(all_metrics[metric][k])
            # Left join on original df to link results to original users
            results = proc.user_id_mapping.merge(
                all_metrics[metric][k].results, left_on="uid", right_on="user_id", how="left", suffixes=('_ext', '_internal'))
            print(results.head())
            # userId and score columns
            results = results[['user_id_ext', 'score']]
            # Save to csv
            results.to_csv(
                f'{output_folder}/{metric}_k{k}.csv', index=False)
            # todo: map to original users in other file


if __name__ == "__main__":
    main()
