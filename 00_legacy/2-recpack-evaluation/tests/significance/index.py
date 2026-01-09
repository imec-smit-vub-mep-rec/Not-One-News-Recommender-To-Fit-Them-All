"""
This script demonstrates how to perform statistical tests (ANOVA and Tukey's HSD)
to verify claims about the significance of recommendation algorithm performance.

IMPORTANT: This script uses MOCK DATA generated to match the summary statistics
in the provided table. To use this for your own research, you must replace the
mock data generation with your actual, raw, per-user experimental results.
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_values(algorithm_name, dataset, K):
    """
    Reads the NDCG scores for a given algorithm from a CSV file.
    Args:
        algorithm_name (str): The name of the algorithm (e.g., 'EASE', 'ItemKNN', 'Pop', 'CB-ST').
        dataset (str): The dataset name (default is 'adressa').
        K (int): The number of recommendations considered (default is 50).
    Returns:
        np.array: An array of NDCG scores for the specified algorithm.
    """
    import os
    import pandas as pd
    # Construct the file path
    file_path = os.path.join(
        'data', dataset, f'{algorithm_name}_{K}.csv')
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Check if the expected columns are present
    if 'user_id_ext' not in df.columns or 'score' not in df.columns:
        raise ValueError(
            f"Expected columns 'user_id_ext' and 'score' not found in {file_path}")
    # Return the score column as a numpy array
    return df['score'].to_numpy()


def get_mean(algorithm_name, dataset='adressa', K=50):
    """
    Computes the mean NDCG score for a given algorithm from a CSV file.

    Args:
        algorithm_name (str): The name of the algorithm (e.g., 'EASE', 'ItemKNN', 'Pop', 'CB-ST').
        dataset (str): The dataset name (default is 'adressa').
        K (int): The number of recommendations considered (default is 50).

    Returns:
        float: The mean NDCG score for the specified algorithm.
    """
    values = get_values(algorithm_name, dataset, K)
    mean = np.mean(values)
    print(f"Mean NDCG for {algorithm_name} on {dataset} @ {K}: {mean:.4f}")
    return mean


def run_significance_test(condition_name, performance_means, dataset='adressa', k=50):
    """
    Performs ANOVA and Tukey's HSD test for a given experimental condition.

    Args:
        condition_name (str): A descriptive name for the test (e.g., "Adressa, Focused @50").
        performance_means (dict): A dictionary mapping algorithm names to their mean scores.
        num_samples (int): Number of mock data points to generate per algorithm.
        std_dev (float): Standard deviation for mock data generation.
    """
    print("="*80)
    print(f"🔬 STATISTICAL ANALYSIS FOR: {condition_name}")
    print("="*80)

    # --- Generate mock raw data for each algorithm ---
    # In a real scenario, you would load your data here instead.
    # For example: data = pd.read_csv("my_raw_results.csv")
    raw_data = {
        algo: get_values(algo, dataset, k)
        for algo, mean in performance_means.items()
    }
    # -> Insert RecPack cluster results

    # --- Step 2: Perform One-Way ANOVA ---
    # ANOVA tells us if there is *any* significant difference among the group means.
    # H₀ (Null Hypothesis): The means of all algorithms are equal.
    # H₁ (Alternative Hypothesis): At least one mean is different.

    f_statistic, p_value_anova = f_oneway(
        raw_data['EASE'],
        raw_data['ItemKNN'],
        raw_data['Pop'],
        raw_data['CB-ST']
    )

    print("\n--- Part 1: One-Way ANOVA ---")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {p_value_anova:.4g}")

    if p_value_anova < 0.05:
        print("✅ Result: The p-value is less than 0.05. We reject the null hypothesis.")
        print("Conclusion: There is a statistically significant difference among the algorithms.\n")
    else:
        print("❌ Result: The p-value is greater than 0.05. We fail to reject the null hypothesis.")
        print("Conclusion: We cannot claim a statistically significant difference. Stopping here.\n")
        return

    # --- Step 3: Perform Tukey's HSD Post-Hoc Test ---
    # Since ANOVA was significant, we use Tukey's test to find out *which* specific pairs
    # of algorithms are significantly different from each other.

    # First, we need to structure the data into a "long format" for statsmodels.
    data_long = pd.DataFrame([
        {'score': score, 'algorithm': algo}
        for algo, scores in raw_data.items()
        for score in scores
    ])

    tukey_results = pairwise_tukeyhsd(
        endog=data_long['score'],
        groups=data_long['algorithm'],
        alpha=0.05  # Significance level
    )

    print("--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---")
    print(tukey_results)

    # --- Step 4: Interpret the Results to Prove the Claim ---
    print("\n--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---")

    # The `tukey_results.summary()` gives a table. We can parse it for our claim.
    results_df = pd.DataFrame(
        data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Filter for comparisons involving CB-ST
    cb_st_comparisons = results_df[results_df['group1'].str.contains(
        'CB-ST') | results_df['group2'].str.contains('CB-ST')]

    all_significant_and_worse = True  # Flag to track if CB-ST is worse than all others
    # Handle case where CB-ST might not be in comparisons (e.g., if only 2 algos and CB-ST is one)
    if cb_st_comparisons.empty:
        all_significant_and_worse = False

    for _, row in cb_st_comparisons.iterrows():
        is_cb_st_worse = False
        if row['reject']:  # Is the difference significant?
            # meandiff = mean(group2) - mean(group1)
            if row['group1'] == 'CB-ST' and row['meandiff'] > 0:
                # CB-ST is group1, and mean(group2) > mean(group1), so CB-ST is worse
                is_cb_st_worse = True
            elif row['group2'] == 'CB-ST' and row['meandiff'] < 0:
                # CB-ST is group2, and mean(group2) < mean(group1), so CB-ST is worse
                is_cb_st_worse = True

            if is_cb_st_worse:
                print(
                    f"✅ Comparison '{row['group1']}' vs. '{row['group2']}': CB-ST significantly WORSE (p={row['p-adj']:.4g}, meandiff={row['meandiff']:.4f})")
            else:
                # Significant difference, but CB-ST is not worse (could be better or no difference in the expected direction)
                print(
                    f"⚠️  Comparison '{row['group1']}' vs. '{row['group2']}': SIGNIFICANT, but CB-ST NOT WORSE (p={row['p-adj']:.4g}, meandiff={row['meandiff']:.4f})")
                all_significant_and_worse = False
        else:
            print(
                f"❌ Comparison '{row['group1']}' vs. '{row['group2']}': NOT significant (p={row['p-adj']:.4g})")
            # If any comparison is not significant, or CB-ST not worse, then the overall claim for this condition fails
            all_significant_and_worse = False

    if all_significant_and_worse:
        print("\n🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.")
        print("CB-ST performs significantly worse than all other compared algorithms.")
    else:
        print("\n⚠️ FINAL CONCLUSION: The claim is NOT fully substantiated for this condition.")


def run_analysis_at_k(datasets=['adressa', 'ebnerd'], k=[10, 20, 50]):
    """
    Runs the significance tests for multiple datasets and K values.

    Args:
        datasets (list): List of dataset names to analyze.
        k (list): List of K values to analyze.
    """
    for dataset in datasets:
        for k_value in k:
            print(f"\nRunning analysis for dataset: {dataset}, K: {k_value}")
            performance_means = {
                'EASE': get_mean('EASE', dataset, k_value),
                'ItemKNN': get_mean('ItemKNN', dataset, k_value),
                'Pop': get_mean('Pop', dataset, k_value),
                'CB-ST': get_mean('CB-ST', dataset, k_value)
            }
            run_significance_test(
                condition_name=f"{dataset} @ {k_value}",
                performance_means=performance_means,
                dataset=dataset,
                k=k_value
            )


if __name__ == '__main__':
    # --- PROVING CLAIM 2 & 3 ---
    # Claim 2: "Our content-based recommender significantly underperforms the other algorithms."
    # Claim 3: "Analogous to the EBNeRD results, the content-based recommender performs significantly worse."

    # To prove this, we run the test on both datasets and show the same pattern holds.
    run_analysis_at_k(datasets=['adressa', 'ebnerd'], k=[10, 20, 50])
