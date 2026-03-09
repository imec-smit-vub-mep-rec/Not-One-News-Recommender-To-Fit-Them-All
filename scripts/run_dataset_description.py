#!/usr/bin/env python3
"""
Dataset descriptive statistics script.

Generates an overview of the dataset before running the full pipeline,
including summary statistics, distributions, outlier analysis, and
visualizations.

Usage:
    # From config file (runs conversion automatically)
    python run_dataset_description.py --config config/config_ebnerd_small.json

    # From dataset preset
    python run_dataset_description.py --dataset ebnerd --input-dir data/ebnerd/ebnerd_small

    # From already-converted data directory
    python run_dataset_description.py --data-dir runs/ebnerd_20260302_142823/data

    # With custom output directory
    python run_dataset_description.py --config config/config_ebnerd_small.json --output-dir reports/ebnerd
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, load_config, PRESET_CONFIGS
from src.converters import ADConverter, AdressaConverter, EBNeRDConverter
from src.utils import setup_logging, get_logger, load_dataframe, ensure_dir
from src.utils.datetime import parse_timestamp_series

logger = get_logger("dataset_description")

PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
PALETTE = "Set2"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics for a RICON dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to pipeline config JSON")
    group.add_argument("--data-dir", type=str, help="Directory with converted articles.parquet / impressions.parquet")

    parser.add_argument("--dataset", type=str, choices=list(PRESET_CONFIGS.keys()), help="Dataset preset name (use with --config)")
    parser.add_argument("--input-dir", type=str, help="Raw data directory (used with --dataset)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for report and plots (default: auto)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args) -> tuple:
    """Load articles and impressions DataFrames.

    Returns (articles_df, impressions_df, output_dir).
    """
    if args.data_dir:
        data_dir = Path(args.data_dir)
        articles_df = load_dataframe(str(data_dir / "articles.parquet"))
        impressions_df = load_dataframe(str(data_dir / "impressions.parquet"))
        output_dir = args.output_dir or str(data_dir.parent / "dataset_description")
        return articles_df, impressions_df, output_dir

    if args.config:
        config = load_config(args.config)
    elif args.dataset:
        from dataclasses import replace
        preset = PRESET_CONFIGS[args.dataset]
        if args.input_dir:
            preset = replace(preset, input_path=args.input_dir)
        config = PipelineConfig(dataset=preset)
    else:
        logger.error("Provide --config, --data-dir, or --dataset + --input-dir")
        sys.exit(1)

    input_path = config.dataset.input_path
    if not input_path:
        logger.error("No input path specified. Use --input-dir or set dataset.input_path in config.")
        sys.exit(1)

    fmt = config.dataset.format
    if fmt == "jsonl" or config.dataset.name == "adressa":
        converter = AdressaConverter(config=config.dataset)
        impressions_df = converter.convert_impressions()
        articles_df = converter.convert_articles()
    elif fmt == "parquet" or config.dataset.name == "ebnerd":
        converter = EBNeRDConverter(config=config.dataset)
        articles_df = converter.convert_articles()
        impressions_df = converter.convert_impressions()
    elif fmt == "spark_csv" or config.dataset.name in ("ad", "hln", "vk"):
        converter = ADConverter(config=config.dataset)
        articles_df = converter.convert_articles()
        impressions_df = converter.convert_impressions()
    else:
        logger.error(f"Unsupported format '{fmt}' for descriptive stats. Convert first and use --data-dir.")
        sys.exit(1)

    output_dir = args.output_dir or f"reports/{config.dataset.name}_description_{datetime.now():%Y%m%d_%H%M%S}"
    return articles_df, impressions_df, output_dir


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_basic_stats(articles_df: pd.DataFrame, impressions_df: pd.DataFrame) -> dict:
    n_impressions = len(impressions_df)
    n_articles = len(articles_df)
    n_users = impressions_df["user_id"].nunique()
    n_articles_interacted = impressions_df["article_id"].nunique()
    n_sessions = impressions_df["session_id"].nunique() if "session_id" in impressions_df.columns else None

    is_homepage = impressions_df["article_id"].isna()
    n_homepage = int(is_homepage.sum())
    n_article_impressions = int((~is_homepage).sum())

    ts = parse_timestamp_series(impressions_df["impression_time"])
    time_min, time_max = ts.min(), ts.max()
    span_days = (time_max - time_min).total_seconds() / 86400 if pd.notna(time_min) and pd.notna(time_max) else None

    stats = {
        "n_articles_catalog": n_articles,
        "n_articles_interacted": n_articles_interacted,
        "n_users": n_users,
        "n_impressions": n_impressions,
        "n_homepage_impressions": n_homepage,
        "n_article_impressions": n_article_impressions,
        "homepage_ratio": round(n_homepage / n_impressions, 4) if n_impressions else 0,
        "n_sessions": n_sessions,
        "time_min": str(time_min),
        "time_max": str(time_max),
        "time_span_days": round(span_days, 2) if span_days else None,
        "impressions_per_user_mean": round(n_impressions / n_users, 2) if n_users else 0,
        "impressions_per_user_median": float(impressions_df.groupby("user_id").size().median()),
    }

    if "is_subscriber" in impressions_df.columns:
        subs = impressions_df.groupby("user_id")["is_subscriber"].any()
        stats["n_subscribers"] = int(subs.sum())
        stats["subscriber_ratio"] = round(float(subs.mean()), 4)

    return stats


def compute_missing_data(articles_df: pd.DataFrame, impressions_df: pd.DataFrame) -> dict:
    result = {}
    for name, df in [("articles", articles_df), ("impressions", impressions_df)]:
        total = len(df)
        missing = {}
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                missing[col] = {"count": n_miss, "pct": round(100 * n_miss / total, 2)}
        result[name] = missing
    return result


def compute_per_user_stats(impressions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregated metrics."""
    is_homepage = impressions_df["article_id"].isna()
    user_groups = impressions_df.groupby("user_id")

    user_stats = pd.DataFrame()
    user_stats["total_impressions"] = user_groups.size()
    user_stats["article_impressions"] = (~is_homepage).groupby(impressions_df["user_id"]).sum()
    user_stats["homepage_impressions"] = is_homepage.groupby(impressions_df["user_id"]).sum()

    if "session_id" in impressions_df.columns:
        user_stats["n_sessions"] = user_groups["session_id"].nunique()

    if "read_time" in impressions_df.columns:
        user_stats["total_read_time"] = user_groups["read_time"].sum()
        user_stats["avg_read_time"] = user_groups["read_time"].mean()

    article_impressions = impressions_df[~is_homepage]
    if "category_str" in article_impressions.columns:
        user_stats["n_categories"] = (
            article_impressions.groupby("user_id")["category_str"]
            .nunique()
            .reindex(user_stats.index, fill_value=0)
        )
    elif "category_str" in impressions_df.columns:
        user_stats["n_categories"] = user_groups["category_str"].nunique()

    user_stats["unique_articles"] = article_impressions.groupby("user_id")["article_id"].nunique().reindex(user_stats.index, fill_value=0)

    ts = parse_timestamp_series(impressions_df["impression_time"])
    user_ts_agg = ts.groupby(impressions_df["user_id"]).agg(["min", "max"])
    user_stats["engagement_span_days"] = (user_ts_agg["max"] - user_ts_agg["min"]).dt.total_seconds() / 86400

    return user_stats


def compute_category_stats(impressions_df: pd.DataFrame, articles_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute category distribution from article impressions."""
    if "category_str" not in impressions_df.columns and "category_str" not in articles_df.columns:
        return None

    article_impressions = impressions_df[impressions_df["article_id"].notna()].copy()
    if article_impressions.empty:
        return None

    if "category_str" not in article_impressions.columns:
        cat_lookup = articles_df[["article_id", "category_str"]].drop_duplicates(subset="article_id")
        article_impressions = article_impressions.merge(cat_lookup, on="article_id", how="left")

    valid = article_impressions[article_impressions["category_str"].notna() & (article_impressions["category_str"] != "")]
    if valid.empty:
        return None

    counts = valid["category_str"].value_counts().reset_index()
    counts.columns = ["category", "impressions"]
    counts["pct"] = (counts["impressions"] / counts["impressions"].sum() * 100).round(2)
    return counts


def compute_temporal_stats(impressions_df: pd.DataFrame) -> dict:
    ts = parse_timestamp_series(impressions_df["impression_time"])
    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    hourly = hour.value_counts().sort_index()
    daily = dow.value_counts().sort_index()

    pct_morning = float(((hour >= 6) & (hour < 12)).mean())
    pct_afternoon = float(((hour >= 12) & (hour < 18)).mean())
    pct_evening = float(((hour >= 18) & (hour < 24)).mean())
    pct_night = float(((hour >= 0) & (hour < 6)).mean())
    pct_weekend = float((dow >= 5).mean())

    return {
        "hourly_distribution": hourly.to_dict(),
        "daily_distribution": {
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k]: int(v)
            for k, v in daily.items()
        },
        "pct_morning_6_12": round(pct_morning * 100, 2),
        "pct_afternoon_12_18": round(pct_afternoon * 100, 2),
        "pct_evening_18_24": round(pct_evening * 100, 2),
        "pct_night_0_6": round(pct_night * 100, 2),
        "pct_weekend": round(pct_weekend * 100, 2),
    }


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(user_stats: pd.DataFrame) -> dict:
    """Detect outliers using IQR and z-score methods on per-user metrics."""
    metrics_to_check = [
        c for c in [
            "total_impressions", "article_impressions", "homepage_impressions",
            "n_sessions", "total_read_time", "avg_read_time",
            "unique_articles", "n_categories", "engagement_span_days",
        ] if c in user_stats.columns
    ]

    iqr_outliers = {}
    zscore_outliers = {}

    for metric in metrics_to_check:
        vals = user_stats[metric].dropna()
        if vals.empty:
            continue

        # IQR
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (vals < lower) | (vals > upper)
        iqr_count = int(iqr_mask.sum())

        # z-score
        mean = vals.mean()
        std = vals.std()
        if std > 0:
            z = ((vals - mean) / std).abs()
            z_mask = z > 3
            z_count = int(z_mask.sum())
        else:
            z_mask = pd.Series(False, index=vals.index)
            z_count = 0

        iqr_outliers[metric] = {
            "count": iqr_count,
            "pct": round(100 * iqr_count / len(vals), 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        }
        zscore_outliers[metric] = {
            "count": z_count,
            "pct": round(100 * z_count / len(vals), 2),
        }

    # Top extreme users (by total impressions)
    if "total_impressions" in user_stats.columns:
        top_users = user_stats.nlargest(10, "total_impressions")
        top_user_records = []
        for uid, row in top_users.iterrows():
            record = {"user_id": str(uid)}
            for col in metrics_to_check:
                if col in row.index:
                    record[col] = round(float(row[col]), 2)
            top_user_records.append(record)
    else:
        top_user_records = []

    return {
        "iqr_method": iqr_outliers,
        "zscore_method": zscore_outliers,
        "top_10_users_by_impressions": top_user_records,
    }


def compute_composite_outlier_scores(user_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute a multivariate outlier score per user (L2 norm of z-scores).

    This approximates the pipeline's ``remove_top`` mechanism, which removes
    users by L2 norm in the StandardScaler-transformed feature space.
    """
    metrics = [
        c for c in [
            "total_impressions", "article_impressions", "homepage_impressions",
            "n_sessions", "total_read_time", "avg_read_time",
            "unique_articles", "n_categories", "engagement_span_days",
        ] if c in user_stats.columns
    ]
    if not metrics:
        return pd.DataFrame()

    z = user_stats[metrics].copy()
    for col in metrics:
        mean = z[col].mean()
        std = z[col].std()
        z[col] = ((z[col] - mean) / std).fillna(0) if std > 0 else 0.0

    scores = np.sqrt((z ** 2).sum(axis=1))
    result = pd.DataFrame({"composite_score": scores}, index=user_stats.index)
    result = result.sort_values("composite_score", ascending=False)
    return result


def suggest_remove_top(scores: pd.Series, max_candidates: int = 50) -> dict:
    """Analyse the top composite scores to suggest a ``remove_top`` value.

    Uses the largest relative gap in the ranked score curve (within the first
    ``max_candidates``) as a natural breakpoint.
    """
    ranked = scores.sort_values(ascending=False).values
    n = min(max_candidates, len(ranked) - 1)
    if n < 2:
        return {"suggested_remove_top": 0, "gap_index": 0, "gap_ratio": 0.0, "removal_curve": []}

    gaps = []
    for i in range(n - 1):
        if ranked[i + 1] > 0:
            gap_ratio = ranked[i] / ranked[i + 1]
        else:
            gap_ratio = float("inf")
        gaps.append(gap_ratio)

    best_gap_idx = int(np.argmax(gaps))
    best_gap_ratio = float(gaps[best_gap_idx])
    suggested = best_gap_idx + 1 if best_gap_ratio > 1.3 else 0

    curve = [
        {
            "rank": i + 1,
            "composite_score": round(float(ranked[i]), 4),
            "gap_to_next": round(float(gaps[i]), 4) if i < len(gaps) else None,
        }
        for i in range(n)
    ]

    return {
        "suggested_remove_top": suggested,
        "gap_index": best_gap_idx,
        "gap_ratio": round(best_gap_ratio, 4),
        "removal_curve": curve,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _save_fig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def plot_impressions_per_user(user_stats: pd.DataFrame, output_dir: str):
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        vals = user_stats["total_impressions"]
        axes[0].hist(vals.clip(upper=vals.quantile(0.99)), bins=60, color=sns.color_palette(PALETTE)[0], edgecolor="white", linewidth=0.3)
        axes[0].set_title("Impressions per User (clipped at 99th pctl)")
        axes[0].set_xlabel("Total impressions")
        axes[0].set_ylabel("Number of users")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        axes[1].hist(np.log1p(vals), bins=60, color=sns.color_palette(PALETTE)[1], edgecolor="white", linewidth=0.3)
        axes[1].set_title("Impressions per User (log scale)")
        axes[1].set_xlabel("log(1 + total impressions)")
        axes[1].set_ylabel("Number of users")
        axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/impressions_per_user.png")


def plot_sessions_per_user(user_stats: pd.DataFrame, output_dir: str):
    if "n_sessions" not in user_stats.columns:
        return
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        vals = user_stats["n_sessions"]

        axes[0].hist(vals.clip(upper=vals.quantile(0.99)), bins=50, color=sns.color_palette(PALETTE)[2], edgecolor="white", linewidth=0.3)
        axes[0].set_title("Sessions per User (clipped at 99th pctl)")
        axes[0].set_xlabel("Number of sessions")
        axes[0].set_ylabel("Number of users")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        axes[1].hist(np.log1p(vals), bins=50, color=sns.color_palette(PALETTE)[3], edgecolor="white", linewidth=0.3)
        axes[1].set_title("Sessions per User (log scale)")
        axes[1].set_xlabel("log(1 + sessions)")
        axes[1].set_ylabel("Number of users")
        axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/sessions_per_user.png")


def plot_reading_time(user_stats: pd.DataFrame, output_dir: str):
    if "avg_read_time" not in user_stats.columns:
        return
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        vals = user_stats["avg_read_time"]

        clipped = vals.clip(upper=vals.quantile(0.99))
        axes[0].hist(clipped, bins=60, color=sns.color_palette(PALETTE)[4], edgecolor="white", linewidth=0.3)
        axes[0].set_title("Avg Reading Time per User (clipped at 99th pctl)")
        axes[0].set_xlabel("Avg reading time (s)")
        axes[0].set_ylabel("Number of users")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        axes[1].hist(np.log1p(vals), bins=60, color=sns.color_palette(PALETTE)[5], edgecolor="white", linewidth=0.3)
        axes[1].set_title("Avg Reading Time per User (log scale)")
        axes[1].set_xlabel("log(1 + avg reading time)")
        axes[1].set_ylabel("Number of users")
        axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/reading_time_distribution.png")


def plot_category_distribution(cat_stats: Optional[pd.DataFrame], output_dir: str):
    if cat_stats is None or cat_stats.empty:
        return
    with plt.rc_context(PLOT_STYLE):
        top_n = min(20, len(cat_stats))
        top = cat_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.35)))
        colors = sns.color_palette(PALETTE, n_colors=top_n)
        bars = ax.barh(range(top_n), top["pct"].values, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["category"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage of article impressions (%)")
        ax.set_title(f"Top {top_n} Categories by Impression Share")

        for bar, pct in zip(bars, top["pct"].values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/category_distribution.png")


def plot_temporal_patterns(impressions_df: pd.DataFrame, output_dir: str):
    ts = parse_timestamp_series(impressions_df["impression_time"])
    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        hourly = hour.value_counts().sort_index()
        axes[0].bar(hourly.index, hourly.values, color=sns.color_palette(PALETTE)[0], edgecolor="white", linewidth=0.3)
        axes[0].set_title("Impressions by Hour of Day")
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Number of impressions")
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = dow.value_counts().sort_index()
        axes[1].bar(range(7), [daily.get(i, 0) for i in range(7)], color=sns.color_palette(PALETTE)[1], edgecolor="white", linewidth=0.3)
        axes[1].set_title("Impressions by Day of Week")
        axes[1].set_xlabel("Day")
        axes[1].set_ylabel("Number of impressions")
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_labels)
        axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/temporal_patterns.png")


def plot_outlier_boxplots(user_stats: pd.DataFrame, output_dir: str):
    metrics = [
        c for c in [
            "total_impressions", "n_sessions", "avg_read_time",
            "unique_articles", "n_categories", "engagement_span_days",
        ] if c in user_stats.columns
    ]
    if not metrics:
        return

    with plt.rc_context(PLOT_STYLE):
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]

        colors = sns.color_palette(PALETTE, n_colors=n)
        for i, metric in enumerate(metrics):
            vals = user_stats[metric].dropna()
            bp = axes[i].boxplot(
                vals, vert=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                widths=0.6,
            )
            axes[i].set_title(metric.replace("_", " ").title(), fontsize=10)
            axes[i].set_xticks([])
            axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.suptitle("Per-User Metric Distributions (with outliers)", fontsize=13, y=1.02)
        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/outlier_boxplots.png")


def plot_outlier_scatter(user_stats: pd.DataFrame, output_dir: str):
    """Scatter plot of total impressions vs avg reading time, highlighting outliers."""
    if "avg_read_time" not in user_stats.columns:
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))

        impressions = user_stats["total_impressions"]
        read_time = user_stats["avg_read_time"]

        # IQR-based outlier flags
        imp_q1, imp_q3 = impressions.quantile(0.25), impressions.quantile(0.75)
        imp_iqr = imp_q3 - imp_q1
        rt_q1, rt_q3 = read_time.quantile(0.25), read_time.quantile(0.75)
        rt_iqr = rt_q3 - rt_q1

        is_outlier = (
            (impressions < imp_q1 - 1.5 * imp_iqr) | (impressions > imp_q3 + 1.5 * imp_iqr) |
            (read_time < rt_q1 - 1.5 * rt_iqr) | (read_time > rt_q3 + 1.5 * rt_iqr)
        )

        normal = ~is_outlier
        ax.scatter(impressions[normal], read_time[normal], s=5, alpha=0.3, color=sns.color_palette(PALETTE)[0], label="Normal")
        ax.scatter(impressions[is_outlier], read_time[is_outlier], s=10, alpha=0.5, color=sns.color_palette(PALETTE)[3], label="Outlier (IQR)")
        ax.set_xlabel("Total Impressions")
        ax.set_ylabel("Avg Reading Time (s)")
        ax.set_title("User Activity: Impressions vs Reading Time")
        ax.legend(markerscale=3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/outlier_scatter.png")


def plot_engagement_span(user_stats: pd.DataFrame, output_dir: str):
    if "engagement_span_days" not in user_stats.columns:
        return
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        vals = user_stats["engagement_span_days"]
        ax.hist(vals.clip(upper=vals.quantile(0.99)), bins=60, color=sns.color_palette(PALETTE)[2], edgecolor="white", linewidth=0.3)
        ax.set_title("User Engagement Span (clipped at 99th pctl)")
        ax.set_xlabel("Engagement span (days)")
        ax.set_ylabel("Number of users")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/engagement_span.png")


def plot_user_metric_correlations(user_stats: pd.DataFrame, output_dir: str):
    numeric_cols = user_stats.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return

    with plt.rc_context(PLOT_STYLE):
        corr = user_stats[numeric_cols].corr()
        n = len(numeric_cols)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
        )
        labels = [c.replace("_", " ").title() for c in numeric_cols]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        ax.set_title("Correlation Matrix of Per-User Metrics")
        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/user_metric_correlations.png")


def plot_removal_curve(composite_scores: pd.DataFrame, suggestion: dict, output_dir: str):
    """Plot the ranked composite outlier score curve with suggested cutoff."""
    if composite_scores.empty:
        return

    ranked = composite_scores["composite_score"].sort_values(ascending=False).values
    n_show = min(100, len(ranked))
    ranks = np.arange(1, n_show + 1)
    scores = ranked[:n_show]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(ranks, scores, color=sns.color_palette(PALETTE)[0], linewidth=1.5)
        axes[0].fill_between(ranks, scores, alpha=0.15, color=sns.color_palette(PALETTE)[0])
        cut = suggestion["suggested_remove_top"]
        if cut > 0 and cut <= n_show:
            axes[0].axvline(x=cut, color=sns.color_palette(PALETTE)[3], linestyle="--", linewidth=1.5, label=f"Suggested remove_top={cut}")
            axes[0].scatter([cut], [scores[cut - 1]], color=sns.color_palette(PALETTE)[3], s=60, zorder=5)
            axes[0].legend(fontsize=10)
        axes[0].set_title("Composite Outlier Score (ranked)")
        axes[0].set_xlabel("User rank (1 = most extreme)")
        axes[0].set_ylabel("Composite score (L2 norm of z-scores)")
        axes[0].set_xlim(0, n_show + 1)

        # Gap ratios for top users
        curve = suggestion.get("removal_curve", [])
        if curve:
            gap_ranks = [c["rank"] for c in curve if c["gap_to_next"] is not None]
            gap_vals = [c["gap_to_next"] for c in curve if c["gap_to_next"] is not None]
            axes[1].bar(gap_ranks, gap_vals, color=sns.color_palette(PALETTE)[1], edgecolor="white", linewidth=0.3)
            axes[1].axhline(y=1.3, color="gray", linestyle=":", linewidth=1, label="Threshold (1.3x)")
            if cut > 0:
                axes[1].axvline(x=cut, color=sns.color_palette(PALETTE)[3], linestyle="--", linewidth=1.5, label=f"Suggested cut at rank {cut}")
            axes[1].set_title("Score Gap Ratios (score[i] / score[i+1])")
            axes[1].set_xlabel("User rank")
            axes[1].set_ylabel("Gap ratio")
            axes[1].legend(fontsize=9)

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/outlier_removal_curve.png")

    # Also plot the full distribution of composite scores
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        all_scores = composite_scores["composite_score"].values
        ax.hist(all_scores, bins=80, color=sns.color_palette(PALETTE)[2], edgecolor="white", linewidth=0.3)
        if cut > 0:
            threshold_score = ranked[cut - 1]
            ax.axvline(x=threshold_score, color=sns.color_palette(PALETTE)[3], linestyle="--", linewidth=1.5,
                        label=f"remove_top={cut} threshold ({threshold_score:.1f})")
            ax.legend(fontsize=10)
        ax.set_title("Distribution of Composite Outlier Scores (all users)")
        ax.set_xlabel("Composite score (L2 norm of z-scores)")
        ax.set_ylabel("Number of users")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/outlier_score_distribution.png")


def plot_article_popularity(impressions_df: pd.DataFrame, output_dir: str):
    article_impressions = impressions_df[impressions_df["article_id"].notna()]
    if article_impressions.empty:
        return

    with plt.rc_context(PLOT_STYLE):
        article_counts = article_impressions["article_id"].value_counts()
        # Use float to avoid Int64 dtype rejecting float quantile/clip values
        counts_f = article_counts.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(counts_f.clip(upper=counts_f.quantile(0.99)), bins=60, color=sns.color_palette(PALETTE)[4], edgecolor="white", linewidth=0.3)
        axes[0].set_title("Article Popularity (clipped at 99th pctl)")
        axes[0].set_xlabel("Number of impressions")
        axes[0].set_ylabel("Number of articles")
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        ranks = np.arange(1, len(article_counts) + 1)
        axes[1].plot(ranks, article_counts.values, color=sns.color_palette(PALETTE)[5], linewidth=0.8)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Article Popularity (rank-frequency, log-log)")
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Impressions")

        fig.tight_layout()
        _save_fig(fig, f"{output_dir}/article_popularity.png")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_console_report(
    basic: dict,
    missing: dict,
    user_stats: pd.DataFrame,
    temporal: dict,
    outliers: dict,
    cat_stats: Optional[pd.DataFrame],
    composite_suggestion: Optional[dict] = None,
) -> str:
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("DATASET DESCRIPTIVE STATISTICS")
    lines.append(sep)

    lines.append("\n--- Overview ---")
    lines.append(f"  Articles in catalog:      {basic['n_articles_catalog']:>12,}")
    lines.append(f"  Articles interacted with: {basic['n_articles_interacted']:>12,}")
    lines.append(f"  Unique users:             {basic['n_users']:>12,}")
    lines.append(f"  Total impressions:        {basic['n_impressions']:>12,}")
    lines.append(f"    - Homepage impressions: {basic['n_homepage_impressions']:>12,}")
    lines.append(f"    - Article impressions:  {basic['n_article_impressions']:>12,}")
    lines.append(f"  Homepage ratio:           {basic['homepage_ratio']:>12.2%}")
    if basic["n_sessions"] is not None:
        lines.append(f"  Total sessions:           {basic['n_sessions']:>12,}")
    lines.append(f"  Time range:               {basic['time_min']} -> {basic['time_max']}")
    if basic["time_span_days"] is not None:
        lines.append(f"  Time span:                {basic['time_span_days']:>12.1f} days")
    lines.append(f"  Avg impressions/user:     {basic['impressions_per_user_mean']:>12.1f}")
    lines.append(f"  Median impressions/user:  {basic['impressions_per_user_median']:>12.1f}")
    if "n_subscribers" in basic:
        lines.append(f"  Subscribers:              {basic['n_subscribers']:>12,}  ({basic['subscriber_ratio']:.2%})")

    # Missing data
    lines.append("\n--- Missing Data ---")
    any_missing = False
    for table_name, cols in missing.items():
        if cols:
            any_missing = True
            lines.append(f"  {table_name}:")
            for col, info in cols.items():
                lines.append(f"    {col:<30s}  {info['count']:>10,}  ({info['pct']:.1f}%)")
    if not any_missing:
        lines.append("  No missing values detected.")

    # Per-user summary
    lines.append("\n--- Per-User Statistics ---")
    describe_cols = [c for c in ["total_impressions", "n_sessions", "avg_read_time", "unique_articles", "n_categories", "engagement_span_days"] if c in user_stats.columns]
    desc = user_stats[describe_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    desc_str = desc.to_string(float_format=lambda x: f"{x:,.2f}")
    for line in desc_str.split("\n"):
        lines.append(f"  {line}")

    # Temporal
    lines.append("\n--- Temporal Patterns ---")
    lines.append(f"  Morning   (06-12):  {temporal['pct_morning_6_12']:>6.1f}%")
    lines.append(f"  Afternoon (12-18):  {temporal['pct_afternoon_12_18']:>6.1f}%")
    lines.append(f"  Evening   (18-24):  {temporal['pct_evening_18_24']:>6.1f}%")
    lines.append(f"  Night     (00-06):  {temporal['pct_night_0_6']:>6.1f}%")
    lines.append(f"  Weekend:            {temporal['pct_weekend']:>6.1f}%")

    # Categories
    if cat_stats is not None and not cat_stats.empty:
        lines.append("\n--- Top 10 Categories ---")
        for _, row in cat_stats.head(10).iterrows():
            lines.append(f"  {row['category']:<35s}  {row['impressions']:>10,}  ({row['pct']:.1f}%)")

    # Outliers
    lines.append("\n--- Outlier Analysis ---")
    lines.append("  IQR method (1.5 * IQR):")
    for metric, info in outliers["iqr_method"].items():
        lines.append(f"    {metric:<30s}  {info['count']:>8,} outliers  ({info['pct']:.1f}%)  bounds=[{info['lower_bound']:.1f}, {info['upper_bound']:.1f}]")

    lines.append("  Z-score method (|z| > 3):")
    for metric, info in outliers["zscore_method"].items():
        lines.append(f"    {metric:<30s}  {info['count']:>8,} outliers  ({info['pct']:.1f}%)")

    if outliers["top_10_users_by_impressions"]:
        lines.append("\n  Top 10 users by total impressions:")
        header_fields = list(outliers["top_10_users_by_impressions"][0].keys())
        lines.append("    " + "  ".join(f"{f:<20s}" for f in header_fields))
        for record in outliers["top_10_users_by_impressions"]:
            vals = [str(record.get(f, "")) for f in header_fields]
            lines.append("    " + "  ".join(f"{v:<20s}" for v in vals))

    # Composite outlier score / removal suggestion
    if composite_suggestion:
        lines.append("\n--- Composite Outlier Score (multivariate) ---")
        suggested = composite_suggestion["suggested_remove_top"]
        gap_ratio = composite_suggestion["gap_ratio"]
        lines.append(f"  Method: L2 norm of z-scores across all per-user metrics")
        lines.append(f"  (approximates pipeline's --remove-top mechanism)")
        lines.append(f"")
        if suggested > 0:
            lines.append(f"  >>> Suggested --remove-top: {suggested}  (gap ratio: {gap_ratio:.2f}x)")
        else:
            lines.append(f"  >>> No clear outlier separation found (max gap ratio: {gap_ratio:.2f}x, threshold: 1.3x)")
        lines.append(f"")
        lines.append(f"  Top ranked users by composite score:")
        lines.append(f"    {'Rank':<6s}  {'Score':<12s}  {'Gap to next':<12s}")
        for entry in composite_suggestion["removal_curve"][:20]:
            gap_str = f"{entry['gap_to_next']:.2f}x" if entry["gap_to_next"] is not None else "-"
            marker = "  <<<" if entry["rank"] == suggested else ""
            lines.append(f"    {entry['rank']:<6d}  {entry['composite_score']:<12.2f}  {gap_str:<12s}{marker}")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger.info("Starting dataset descriptive statistics")

    articles_df, impressions_df, output_dir = load_data(args)
    ensure_dir(output_dir)

    logger.info(f"Loaded {len(articles_df)} articles, {len(impressions_df)} impressions")
    logger.info(f"Output directory: {output_dir}")

    # Compute statistics
    logger.info("Computing basic statistics...")
    basic = compute_basic_stats(articles_df, impressions_df)

    logger.info("Analyzing missing data...")
    missing = compute_missing_data(articles_df, impressions_df)

    logger.info("Computing per-user statistics...")
    user_stats = compute_per_user_stats(impressions_df)

    logger.info("Computing category statistics...")
    cat_stats = compute_category_stats(impressions_df, articles_df)

    logger.info("Computing temporal patterns...")
    temporal = compute_temporal_stats(impressions_df)

    logger.info("Detecting outliers...")
    outliers = detect_outliers(user_stats)

    logger.info("Computing composite outlier scores...")
    composite_scores = compute_composite_outlier_scores(user_stats)
    composite_suggestion = suggest_remove_top(composite_scores["composite_score"]) if not composite_scores.empty else None

    # Console report
    report = format_console_report(basic, missing, user_stats, temporal, outliers, cat_stats, composite_suggestion)
    print("\n" + report)

    # Save JSON report
    full_report = {
        "generated_at": datetime.now().isoformat(),
        "basic_stats": basic,
        "missing_data": missing,
        "per_user_summary": {
            col: {
                "mean": round(float(user_stats[col].mean()), 4),
                "std": round(float(user_stats[col].std()), 4),
                "min": round(float(user_stats[col].min()), 4),
                "p25": round(float(user_stats[col].quantile(0.25)), 4),
                "median": round(float(user_stats[col].median()), 4),
                "p75": round(float(user_stats[col].quantile(0.75)), 4),
                "p90": round(float(user_stats[col].quantile(0.90)), 4),
                "p95": round(float(user_stats[col].quantile(0.95)), 4),
                "p99": round(float(user_stats[col].quantile(0.99)), 4),
                "max": round(float(user_stats[col].max()), 4),
            }
            for col in user_stats.select_dtypes(include=[np.number]).columns
        },
        "temporal_patterns": temporal,
        "category_distribution": cat_stats.head(30).to_dict(orient="records") if cat_stats is not None else None,
        "outliers": outliers,
        "composite_outlier_analysis": composite_suggestion,
    }

    json_path = f"{output_dir}/dataset_description.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Saved JSON report: {json_path}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_impressions_per_user(user_stats, output_dir)
    plot_sessions_per_user(user_stats, output_dir)
    plot_reading_time(user_stats, output_dir)
    plot_category_distribution(cat_stats, output_dir)
    plot_temporal_patterns(impressions_df, output_dir)
    plot_outlier_boxplots(user_stats, output_dir)
    plot_outlier_scatter(user_stats, output_dir)
    if not composite_scores.empty and composite_suggestion:
        plot_removal_curve(composite_scores, composite_suggestion, output_dir)
    plot_engagement_span(user_stats, output_dir)
    plot_user_metric_correlations(user_stats, output_dir)
    plot_article_popularity(impressions_df, output_dir)

    logger.info(f"Done. All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
