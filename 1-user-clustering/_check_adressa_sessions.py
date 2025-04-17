"""
File that summarizes the sessions of Adressa
1. Loads datasets/adressa-large-0416/behaviors_with_homepage.parquet
2. Groups by session_id and counts the number of impressions
3. Reports average number of impressions per session and the top 5 sessions with the most impressions
"""

import pandas as pd


def main():
    # Load the dataset
    print("Loading dataset...")
    # "datasets/ekstra-large/behaviors.parquet"
    path = 'datasets/adressa-large-0416/behaviors_with_homepage.parquet'
    df = pd.read_parquet(
        path)

    # Group by session_id and count impressions
    print("Analyzing sessions...")
    session_counts = df.groupby('session_id').size(
    ).reset_index(name='impression_count')

    # Calculate and report stats
    avg_impressions = session_counts['impression_count'].mean()
    print(
        f"\nAverage number of impressions per session: {avg_impressions:.2f}")

    # Get top 5 sessions with most impressions
    top_sessions = session_counts.nlargest(5, 'impression_count')
    print("\nTop 5 sessions with most impressions:")
    for i, (session_id, count) in enumerate(zip(top_sessions['session_id'], top_sessions['impression_count']), 1):
        print(f"{i}. Session ID: {session_id}, Impressions: {count}")

    # Count sessions with more than MAX_IMPRESSIONS impressions
    MAX_IMPRESSIONS = 20
    high_impression_sessions = session_counts[session_counts['impression_count']
                                              > MAX_IMPRESSIONS]
    print(
        f"\nNumber of sessions with more than {MAX_IMPRESSIONS} impressions: {len(high_impression_sessions)}")

    # Calculate 99.99th percentile
    percentile_9999 = session_counts['impression_count'].quantile(0.9999)
    extreme_sessions = session_counts[session_counts['impression_count']
                                      >= percentile_9999]
    print(
        f"\n99.99th percentile of impressions per session: {percentile_9999:.2f}")
    print(
        f"Number of sessions at or above 99.99th percentile: {len(extreme_sessions)}")
    print(
        f"Impression_count cutoff point: {percentile_9999:.2f}")

    # Log total number of sessions
    print(f"\nTotal number of sessions: {len(session_counts)}")

    # Summary statistics
    print("\nSession impression count statistics:")
    print(session_counts['impression_count'].describe())

    # Count number of unique sessions
    num_sessions = session_counts.shape[0]
    print(f"\nTotal number of unique sessions: {num_sessions}")

    # Now remove the sessions with more than MAX_IMPRESSIONS impressions from the behavior dataset and save it as a new file with suffix _9999percentile_removed.parquet
    df_9999percentile_removed = df[~df['session_id'].isin(
        high_impression_sessions['session_id'])]
    df_9999percentile_removed.to_parquet(
        f"{path}_9999percentile_removed.parquet")


if __name__ == "__main__":
    main()
