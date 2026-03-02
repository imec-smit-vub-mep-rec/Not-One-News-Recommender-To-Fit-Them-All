"""
Plot sessions-per-user distribution by cluster for a pipeline run.

Loads user_clusters.parquet and impressions_cleaned.parquet from a run directory,
computes sessions per user, and exports a probability histogram colored by cluster
as an image. The x-axis is clipped to the 99th percentile for readability.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sessions-per-user distribution by cluster for a run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the pipeline run directory containing user_clusters.parquet and impressions_cleaned.parquet",
    )
    return parser.parse_args()


def main() -> None:
    """Load run data, plot sessions-per-user by cluster, and export the figure as an image."""
    args = parse_args()
    run_dir = args.run_dir

    clusters_path = run_dir / "user_clusters.parquet"
    impressions_path = run_dir / "impressions_cleaned.parquet"

    clusters_df = pd.read_parquet(clusters_path)
    impressions_df = pd.read_parquet(impressions_path)

    sessions_per_user = (
        impressions_df.groupby("user_id")["session_id"]
        .nunique()
        .reset_index(name="num_sessions")
    )

    plot_df = pd.merge(clusters_df, sessions_per_user, on="user_id")
    x_max = plot_df["num_sessions"].quantile(0.99)

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=plot_df,
        x="num_sessions",
        hue="cluster_id",
        element="step",
        stat="probability",
        common_norm=False,
        bins=30,
        palette="tab10",
    )

    plt.title("Distribution of Number of Sessions per User by Cluster")
    plt.xlabel("Number of Sessions")
    plt.ylabel("Probability")
    plt.xlim(0, x_max)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = run_dir / "sessions_per_user_by_cluster.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()