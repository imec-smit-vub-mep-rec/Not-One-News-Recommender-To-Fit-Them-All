"""
Plot sessions-per-user distribution by cluster for a pipeline run.

Loads user_clusters.parquet and impressions_cleaned.parquet from a run directory,
computes sessions per user, and exports a probability histogram colored by cluster
as an image. Uses integer-width bins within the visible range (99th percentile),
unfilled step lines for clarity, and optionally a log-scale x-axis.
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
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use log scale for the x-axis (number of sessions)",
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
    x_min = 1 if args.log_x else 0

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=plot_df,
        x="num_sessions",
        hue="cluster_id",
        element="step",
        fill=False,
        linewidth=2,
        stat="probability",
        common_norm=False,
        binwidth=1,
        binrange=(x_min, x_max),
        palette="tab10",
    )

    plt.title("Distribution of Number of Sessions per User by Cluster")
    plt.xlabel("Number of Sessions" + (" (log scale)" if args.log_x else ""))
    plt.ylabel("Probability")
    if args.log_x:
        plt.xscale("log")
    plt.xlim(x_min, x_max)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = run_dir / "sessions_per_user_by_cluster.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()