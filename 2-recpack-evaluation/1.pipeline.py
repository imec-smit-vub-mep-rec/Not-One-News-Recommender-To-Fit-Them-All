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

# Set more restrictive logging levels for all loggers
logger = logging.getLogger("recpack")
logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Disable other verbose loggers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)  # Root logger


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

    # Load articles content
    articles_content_df = pd.read_csv(articles_content_path)

    # Print the actual column names from the mappings to avoid errors
    print("Item mapping columns:", proc.item_id_mapping.columns.tolist())

    # Create content dictionary using internal IDs from RecPack
    # Map the original article IDs to the internal IDs used by RecPack
    content_dict = {}

    # Get the ID mapping - adjust column names based on actual dataframe structure
    # Typically the mapping has 'iid' for original ID and 'uid' or 'id' for internal ID
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

    # Calculate timestamps for splits
    t_validation = df['impression_time'].quantile(0.71)
    t_test = df['impression_time'].quantile(0.86)

    scenario = WeakGeneralization(
        frac_data_in=0.8,
        validation=True
    )
    # scenario = Timed(
    #     t=t_test,
    #     t_validation=t_validation,
    #     validation=True
    # )
    # scenario = LastItemPrediction(
    #     validation=True,
    #     # How much of the historic events to use as input history. Defaults to the maximal integer value.
    #     n_most_recent_in=30
    # )

    scenario.split(interaction_matrix)

    # Set up pipeline
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)

    # Register the custom algorithm
    ALGORITHM_REGISTRY.register(
        'SentenceTransformerContentBased', SentenceTransformerContentBased)

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

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    file_path = f'{folder}/interactions.csv'
    articles_content_path = f'{folder}/articles_content.csv'

    # Run the pipeline analysis on the combined file
    proc, metrics, all_metrics = run_pipeline_analysis(
        file_path, articles_content_path)

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
            # Left join on original df to link results to original users
            # Adjust column names here too based on actual mapping structure

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

                # --- Debugging Info ---
                print(f"\n--- Processing {metric_key_str} - {k} ---")
                print("proc.user_id_mapping columns:", proc.user_id_mapping.columns.tolist())
                print("proc.user_id_mapping head:\n", proc.user_id_mapping.head())
                print("results_df columns:", results_df.columns.tolist())
                print("results_df index:", results_df.index)
                print("results_df head:\n", results_df.head())
                # --- End Debugging Info ---


                # Determine how to merge based on results_df structure
                if 'user_id' in results_df.columns:
                    print(f"Merging results_df on 'user_id' column (right_on='user_id') with proc.user_id_mapping on '{join_col}' column (left_on='{join_col}')")
                    results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_on='user_id',
                        how="left",
                        suffixes=('_ext', '_internal')
                    )
                elif results_df.index.name == 'user_id':
                    print(f"Merging results_df on index (right_index=True) with proc.user_id_mapping on '{join_col}' column (left_on='{join_col}')")
                    results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_index=True, # Merge on the right DataFrame's index
                        how="left",
                        suffixes=('_ext', '_internal')
                    )
                elif len(results_df.columns) > 0 and 'score' in results_df.columns:
                     # Heuristic: If 'user_id' is missing but 'score' is present, assume the first column is user ID
                     potential_user_col = results_df.columns[0]
                     print(f"Warning: 'user_id' not found. Assuming first column '{potential_user_col}' is user ID for merge.")
                     print(f"Merging results_df on '{potential_user_col}' column (right_on='{potential_user_col}') with proc.user_id_mapping on '{join_col}' column (left_on='{join_col}')")
                     results = proc.user_id_mapping.merge(
                        results_df,
                        left_on=join_col,
                        right_on=potential_user_col,
                        how="left",
                        suffixes=('_ext', '_internal')
                     )
                else:
                     print(f"Error: Cannot determine user ID column or index in results_df for {metric_key_str} - {k}.")
                     print("Skipping merge for this combination.")
                     continue # Skip this metric/k combination


                # --- Post-merge processing ---
                # Determine the actual name of the external user ID column after merge
                user_id_col_ext_orig = proc.user_id_mapping.columns[0] # Original external user ID column name
                user_id_col_ext_suffixed = user_id_col_ext_orig + '_ext'

                if user_id_col_ext_suffixed in results.columns:
                    user_id_col_to_use = user_id_col_ext_suffixed
                    print(f"Using suffixed external user ID column: '{user_id_col_to_use}'")
                elif user_id_col_ext_orig in results.columns:
                    user_id_col_to_use = user_id_col_ext_orig
                    print(f"Using original external user ID column: '{user_id_col_to_use}'")
                else:
                    print(f"Error: External user ID column ('{user_id_col_ext_orig}' or '{user_id_col_ext_suffixed}') not found in merged results.")
                    print("Merged columns:", results.columns.tolist())
                    continue

                # Select external user ID and score
                if 'score' in results.columns:
                    results_final = results[[user_id_col_to_use, 'score']].copy()
                # Fallback if 'score' is not found, assume last column is score if user ID is not the last column
                elif len(results.columns) > 1 and results.columns[-1] != user_id_col_to_use:
                     score_col = results.columns[-1]
                     print(f"Warning: 'score' column not found. Assuming last column '{score_col}' is score.")
                     results_final = results[[user_id_col_to_use, score_col]].copy()
                else:
                     print(f"Error: Cannot determine score column in merged results for {metric_key_str} - {k}.")
                     # Create dummy score column
                     results_final = results[[user_id_col_to_use]].copy()
                     results_final['score'] = np.nan

                # Rename columns for consistency AFTER selection
                results_final.columns = ['user_id_ext', 'score']

                # Save to csv
                output_path = f'{output_folder}/{metric_name}_k{k}.csv'
                print(f"Saving results to: {output_path}")
                results_final.to_csv(output_path, index=False)

            except Exception as e:
                print(f"\n--- CRITICAL Error processing metric results for {metric_key_str} - {k} ---")
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
                    raw_output_path = f'{output_folder}/{metric_name}_k{k}_raw.csv'
                    # Save with index to help debug raw data
                    all_metrics[metric][k].results.to_csv(raw_output_path, index=True)
                    print(f"Saved raw results to: {raw_output_path}")
                except Exception as e_save:
                    print(f"Failed to save raw results: {str(e_save)}")


if __name__ == "__main__":
    main()
