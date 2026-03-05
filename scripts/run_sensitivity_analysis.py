#!/usr/bin/env python3
"""
Run clustering sensitivity analyses for a completed pipeline run.

This script evaluates:
1) MiniBatchKMeans sensitivity (batch_size and n_init stability)
2) Scaler sensitivity (Standard/MinMax/Robust/Log1p+Standard)
3) Clustering diagnostics (entropy, feature importance, centroid radar)

Outputs are written under:
<run-dir>/sensitivity_analysis/{logs,visualizations,results}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for sensitivity analysis visualizations"
    ) from exc


# Add repo root to import src package when called as scripts/run_sensitivity_analysis.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.feature_engineering import create_user_features
from src.utils import ensure_dir, get_logger, load_dataframe, save_json, setup_logging


logger = get_logger("sensitivity_analysis")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sensitivity analysis script."""
    parser = argparse.ArgumentParser(
        description="Run clustering sensitivity analyses for an existing run directory",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., runs/ad_20260301_120000)",
    )
    parser.add_argument(
        "--skip-part-a",
        action="store_true",
        help="Skip Part A (MiniBatch sensitivity tests)",
    )
    parser.add_argument(
        "--skip-part-b",
        action="store_true",
        help="Skip Part B (scaler comparison tests)",
    )
    parser.add_argument(
        "--skip-part-c",
        action="store_true",
        help="Skip Part C (entropy/importance/radar diagnostics)",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help="Sample size for silhouette_score on large datasets",
    )
    parser.add_argument(
        "--radar-top-features",
        type=int,
        default=12,
        help="Top varying centroid features to show in radar chart",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _resolve_run_root(run_dir: Path) -> Path:
    """Resolve run root; support passing either run root or run/data path."""
    run_dir = run_dir.resolve()
    if (run_dir / "data").exists():
        return run_dir
    if run_dir.name == "data" and run_dir.parent.exists():
        return run_dir.parent
    raise FileNotFoundError(
        "Could not resolve run directory. Expected either <run>/data or <run> containing data/."
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int = 42,
) -> float:
    """Compute silhouette score with optional sampling to keep runtime tractable."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    sil_sample = sample_size if (sample_size > 0 and len(X) > sample_size) else None
    return float(
        silhouette_score(X, labels, sample_size=sil_sample, random_state=random_state)
    )


def _pythonize(value: Any) -> Any:
    """Recursively convert numpy/pandas objects to JSON-serializable Python objects."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _pythonize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_pythonize(v) for v in value]
    return value


def run_part_a(
    X_scaled: np.ndarray,
    n_clusters: int,
    viz_dir: Path,
    silhouette_sample_size: int,
) -> Dict[str, Any]:
    """
    Part A:
      A1) Batch-size stability for MiniBatchKMeans
      A2) n_init stability/variance sensitivity
    """
    logger.info("Running Part A: MiniBatch sensitivity analysis")

    results: Dict[str, Any] = {"batch_size_stability": {}, "n_init_variance": {}}

    # ---------- Test A1: Batch-size stability ----------
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192]
    batch_records: List[Dict[str, Any]] = []

    # Full KMeans baseline for quality reference (inertia/silhouette)
    t0 = time.perf_counter()
    baseline = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    baseline_labels = baseline.fit_predict(X_scaled)
    baseline_time = time.perf_counter() - t0
    baseline_inertia = float(baseline.inertia_)
    baseline_sil = _safe_silhouette(
        X_scaled, baseline_labels, sample_size=silhouette_sample_size, random_state=42
    )

    for batch_size in batch_sizes:
        t_start = time.perf_counter()
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            batch_size=batch_size,
        )
        labels = model.fit_predict(X_scaled)
        elapsed = time.perf_counter() - t_start
        sil = _safe_silhouette(
            X_scaled, labels, sample_size=silhouette_sample_size, random_state=42
        )
        record = {
            "batch_size": batch_size,
            "fit_time_seconds": float(elapsed),
            "inertia": float(model.inertia_),
            "silhouette_score": sil,
        }
        batch_records.append(record)
        logger.info(
            "A1 batch_size=%s | time=%.3fs | inertia=%.3f | silhouette=%.4f",
            batch_size,
            elapsed,
            model.inertia_,
            sil,
        )

    # Plot: dual-axis batch_size vs time and inertia
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = [r["batch_size"] for r in batch_records]
    y_time = [r["fit_time_seconds"] for r in batch_records]
    y_inertia = [r["inertia"] for r in batch_records]
    ln1 = ax1.plot(x, y_time, marker="o", color="#1f77b4", label="Fit Time (s)")
    ax1.set_xlabel("MiniBatchKMeans batch_size")
    ax1.set_ylabel("Fit Time (seconds)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(x, y_inertia, marker="s", color="#d62728", label="Inertia")
    ax2.axhline(
        baseline_inertia,
        linestyle="--",
        color="#9467bd",
        label="Full KMeans Inertia",
        alpha=0.8,
    )
    ax2.set_ylabel("Inertia", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ln1 + ln2 + ax2.lines[-1:]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title("A1: MiniBatch batch_size stability")
    plt.tight_layout()
    batch_plot = viz_dir / "batch_size_stability.png"
    plt.savefig(batch_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["batch_size_stability"] = {
        "baseline_kmeans": {
            "fit_time_seconds": baseline_time,
            "inertia": baseline_inertia,
            "silhouette_score": baseline_sil,
        },
        "mini_batch_results": batch_records,
        "plot_path": str(batch_plot),
    }

    # ---------- Test A2: n_init variance ----------
    n_inits = [1, 3, 5, 10, 20]
    seeds = list(range(1, 21))
    ninit_records: List[Dict[str, Any]] = []

    for n_init in n_inits:
        inertias = []
        for seed in seeds:
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=n_init,
                batch_size=2048,
            )
            model.fit(X_scaled)
            inertias.append(float(model.inertia_))
        inertias_np = np.array(inertias, dtype=float)
        rec = {
            "n_init": n_init,
            "mean_inertia": float(inertias_np.mean()),
            "std_inertia": float(inertias_np.std(ddof=1)),
            "min_inertia": float(inertias_np.min()),
            "max_inertia": float(inertias_np.max()),
            "all_inertias": inertias,
        }
        ninit_records.append(rec)
        logger.info(
            "A2 n_init=%s | mean=%.3f std=%.3f min=%.3f max=%.3f",
            n_init,
            rec["mean_inertia"],
            rec["std_inertia"],
            rec["min_inertia"],
            rec["max_inertia"],
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    x_n = [r["n_init"] for r in ninit_records]
    y_mean = [r["mean_inertia"] for r in ninit_records]
    y_std = [r["std_inertia"] for r in ninit_records]
    ax.errorbar(
        x_n,
        y_mean,
        yerr=y_std,
        marker="o",
        linestyle="-",
        capsize=4,
        color="#2ca02c",
    )
    ax.set_xlabel("n_init")
    ax.set_ylabel("Inertia (mean +/- 1 std across 20 seeds)")
    ax.set_title("A2: n_init sensitivity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ninit_plot = viz_dir / "n_init_variance.png"
    plt.savefig(ninit_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["n_init_variance"] = {
        "n_init_results": ninit_records,
        "plot_path": str(ninit_plot),
    }
    return results


def _build_scaler_variants(X_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """Build scaled variants for Part B."""
    variants: Dict[str, np.ndarray] = {}

    variants["StandardScaler"] = StandardScaler().fit_transform(X_raw)
    variants["MinMaxScaler"] = MinMaxScaler().fit_transform(X_raw)
    variants["RobustScaler"] = RobustScaler().fit_transform(X_raw)

    # Ensure log1p domain safety by shifting each feature to be >= 0 first.
    mins = X_raw.min(axis=0)
    shifts = np.where(mins < 0, -mins, 0.0)
    X_shifted = X_raw + shifts
    variants["Log1p+StandardScaler"] = StandardScaler().fit_transform(np.log1p(X_shifted))
    return variants


def run_part_b(
    impressions_df: pd.DataFrame,
    articles_df: Optional[pd.DataFrame],
    n_clusters: int,
    legacy_mode: bool,
    viz_dir: Path,
    silhouette_sample_size: int,
) -> Dict[str, Any]:
    """
    Part B:
      Compare clustering quality under different scaling strategies.
    """
    logger.info("Running Part B: scaler sensitivity analysis")

    # Recreate user features WITHOUT scaling so we can apply each scaler fairly.
    features_unscaled, metadata = create_user_features(
        impressions_df=impressions_df,
        articles_df=articles_df,
        legacy_mode=legacy_mode,
        scale=False,
        user_col="user_id",
    )
    feature_cols = [c for c in features_unscaled.columns if c != "user_id"]
    X_raw = features_unscaled[feature_cols].to_numpy(dtype=float)

    # Plot raw feature distributions (first 12 features to keep chart legible).
    max_plots = min(12, len(feature_cols))
    fig, axes = plt.subplots(
        nrows=int(np.ceil(max_plots / 3)),
        ncols=3,
        figsize=(15, 3.5 * int(np.ceil(max_plots / 3))),
    )
    axes = np.array(axes).reshape(-1)
    for i in range(max_plots):
        ax = axes[i]
        ax.hist(X_raw[:, i], bins=40, alpha=0.8, color="#4c72b0")
        ax.set_title(feature_cols[i], fontsize=9)
        ax.grid(True, alpha=0.2)
    for j in range(max_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    dist_plot = viz_dir / "feature_distributions.png"
    plt.savefig(dist_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    variants = _build_scaler_variants(X_raw)
    scaler_records: List[Dict[str, Any]] = []

    for scaler_name, X_scaled in variants.items():
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        rec = {
            "scaler": scaler_name,
            "inertia": float(model.inertia_),
            "silhouette_score": _safe_silhouette(
                X_scaled, labels, sample_size=silhouette_sample_size, random_state=42
            ),
            "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
        }
        scaler_records.append(rec)
        logger.info(
            "B scaler=%s | silhouette=%.4f | db=%.4f | ch=%.3f | inertia=%.3f",
            scaler_name,
            rec["silhouette_score"],
            rec["davies_bouldin_score"],
            rec["calinski_harabasz_score"],
            rec["inertia"],
        )

    # Grouped bar chart over three quality metrics.
    df_scores = pd.DataFrame(scaler_records)
    metrics_to_plot = [
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        ax.bar(df_scores["scaler"], df_scores[metric], color="#55a868")
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    scaler_plot = viz_dir / "scaler_comparison.png"
    plt.savefig(scaler_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_users": int(metadata.get("n_users", len(features_unscaled))),
        "n_features": int(metadata.get("n_features", len(feature_cols))),
        "feature_columns": feature_cols,
        "scaler_results": scaler_records,
        "distribution_plot_path": str(dist_plot),
        "scaler_plot_path": str(scaler_plot),
    }


def run_part_c(
    X_scaled: np.ndarray,
    feature_cols: List[str],
    labels: np.ndarray,
    n_clusters: int,
    viz_dir: Path,
    radar_top_features: int,
) -> Dict[str, Any]:
    """
    Part C:
      C1) Cluster entropy
      C2) Feature importance for cluster discrimination
      C3) Centroid radar visualization
    """
    logger.info("Running Part C: clustering diagnostics")

    results: Dict[str, Any] = {}

    # ---------- C1: Cluster entropy ----------
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-10)).sum())
    max_entropy = float(np.log(n_clusters))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Observed Entropy", "Max Entropy (log k)"], [entropy, max_entropy], color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Entropy")
    ax.set_title("C1: Cluster size entropy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    entropy_plot = viz_dir / "cluster_entropy.png"
    plt.savefig(entropy_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["cluster_entropy"] = {
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "entropy": entropy,
        "max_entropy_log_k": max_entropy,
        "normalized_entropy": normalized_entropy,
        "plot_path": str(entropy_plot),
    }
    logger.info(
        "C1 entropy=%.4f | max=%.4f | normalized=%.4f",
        entropy,
        max_entropy,
        normalized_entropy,
    )

    # ---------- C2: Feature importance ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    top_n = min(20, len(feat_imp))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp["feature"].head(top_n)[::-1], feat_imp["importance"].head(top_n)[::-1], color="#55a868")
    ax.set_xlabel("Importance")
    ax.set_title("C2: Top feature importances for predicting cluster_id")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fi_plot = viz_dir / "feature_importance.png"
    plt.savefig(fi_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["feature_importance"] = {
        "test_accuracy": test_accuracy,
        "top_features": feat_imp.head(20).to_dict(orient="records"),
        "all_features": feat_imp.to_dict(orient="records"),
        "plot_path": str(fi_plot),
    }
    logger.info("C2 random-forest test accuracy=%.4f", test_accuracy)

    # ---------- C3: Cluster centroid radar ----------
    # Fit KMeans to derive centroids on the same scaled feature matrix.
    centroid_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    centroid_model.fit(X_scaled)
    centers = centroid_model.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=feature_cols)

    # For readability, select most discriminative centroid dimensions by variance.
    radar_n = min(max(3, radar_top_features), len(feature_cols))
    top_features = (
        centers_df.var(axis=0)
        .sort_values(ascending=False)
        .head(radar_n)
        .index.tolist()
    )
    radar_data = centers_df[top_features].to_numpy()

    angles = np.linspace(0, 2 * np.pi, radar_n, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for cluster_idx in range(radar_data.shape[0]):
        vals = radar_data[cluster_idx]
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, linewidth=2, label=f"Cluster {cluster_idx}")
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=9)
    ax.set_title("C3: Cluster centroid radar (top varying features)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    radar_plot = viz_dir / "centroid_radar.png"
    plt.savefig(radar_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["centroid_radar"] = {
        "top_features_used": top_features,
        "centers_top_features": centers_df[top_features].to_dict(orient="records"),
        "plot_path": str(radar_plot),
    }

    return results


def main() -> None:
    args = parse_args()
    run_root = _resolve_run_root(Path(args.run_dir))
    data_dir = run_root / "data"

    analysis_root = run_root / "sensitivity_analysis"
    logs_dir = ensure_dir(analysis_root / "logs")
    viz_dir = ensure_dir(analysis_root / "visualizations")
    results_dir = ensure_dir(analysis_root / "results")

    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        log_file=logs_dir / "sensitivity_analysis.log",
    )

    logger.info("Starting clustering sensitivity analysis")
    logger.info("Run root: %s", run_root)

    # Required inputs from existing run outputs
    config_path = run_root / "config.json"
    user_features_path = data_dir / "user_features.parquet"
    user_clusters_path = data_dir / "user_clusters.parquet"
    impressions_path = data_dir / "impressions_cleaned.parquet"
    articles_path = data_dir / "articles_cleaned.parquet"

    for path in [config_path, user_features_path, user_clusters_path, impressions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = _load_json(config_path)
    clustering_cfg = config.get("clustering", {})
    n_clusters_cfg = clustering_cfg.get("n_clusters")
    legacy_mode = bool(clustering_cfg.get("legacy_features", False))

    features_df = load_dataframe(user_features_path)
    cluster_df = load_dataframe(user_clusters_path)
    impressions_df = load_dataframe(impressions_path)
    articles_df = load_dataframe(articles_path) if articles_path.exists() else None

    if "user_id" not in features_df.columns:
        raise ValueError(f"`user_id` missing in {user_features_path}")
    if not {"user_id", "cluster_id"}.issubset(cluster_df.columns):
        raise ValueError(f"`user_id`/`cluster_id` missing in {user_clusters_path}")

    merged = features_df.merge(
        cluster_df[["user_id", "cluster_id"]],
        on="user_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping users between user_features.parquet and user_clusters.parquet")

    feature_cols = [c for c in merged.columns if c not in {"user_id", "cluster_id"}]
    X_scaled = merged[feature_cols].to_numpy(dtype=float)
    labels = merged["cluster_id"].to_numpy(dtype=int)
    n_clusters = int(n_clusters_cfg) if n_clusters_cfg is not None else int(len(np.unique(labels)))

    logger.info(
        "Loaded data: users=%s features=%s clusters=%s legacy_mode=%s",
        len(merged),
        len(feature_cols),
        n_clusters,
        legacy_mode,
    )

    results: Dict[str, Any] = {
        "metadata": {
            "run_root": str(run_root),
            "n_users": int(len(merged)),
            "n_features": int(len(feature_cols)),
            "n_clusters": int(n_clusters),
            "legacy_mode": legacy_mode,
            "silhouette_sample_size": int(args.silhouette_sample_size),
        }
    }

    if not args.skip_part_a:
        results["part_a"] = run_part_a(
            X_scaled=X_scaled,
            n_clusters=n_clusters,
            viz_dir=viz_dir,
            silhouette_sample_size=args.silhouette_sample_size,
        )
    else:
        logger.info("Skipping Part A by request")

    if not args.skip_part_b:
        results["part_b"] = run_part_b(
            impressions_df=impressions_df,
            articles_df=articles_df,
            n_clusters=n_clusters,
            legacy_mode=legacy_mode,
            viz_dir=viz_dir,
            silhouette_sample_size=args.silhouette_sample_size,
        )
    else:
        logger.info("Skipping Part B by request")

    if not args.skip_part_c:
        results["part_c"] = run_part_c(
            X_scaled=X_scaled,
            feature_cols=feature_cols,
            labels=labels,
            n_clusters=n_clusters,
            viz_dir=viz_dir,
            radar_top_features=args.radar_top_features,
        )
    else:
        logger.info("Skipping Part C by request")

    out_json = results_dir / "sensitivity_results.json"
    save_json(_pythonize(results), out_json)
    logger.info("Saved aggregated results to %s", out_json)
    logger.info("Sensitivity analysis complete")


if __name__ == "__main__":
    main()
