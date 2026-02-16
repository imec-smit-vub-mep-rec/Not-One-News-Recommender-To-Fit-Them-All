"""
Topic-level diversity metrics for recommendation evaluation.

Computes Coverage and Gini at the topic (category) level, with support for
multi-topic items (exposure split across categories) and a topic popularity report.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger("evaluation.topic_diversity")


def _parse_categories_value(val: Any) -> Optional[List[str]]:
    """Parse categories from various formats (list, JSON string, etc.)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return [str(x).strip() for x in val if x is not None and str(x).strip()]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            if isinstance(parsed, str):
                return [parsed] if parsed else None
        except (ValueError, SyntaxError):
            return [val] if val else None
    return None


def _normalize_categories(
    articles_df: pd.DataFrame,
    item_mapping: Dict[str, int],
    article_col: str = "article_id",
    categories_col: str = "categories",
    category_str_col: str = "category_str",
) -> Tuple[Dict[int, List[str]], Set[str]]:
    """Build internal_id -> list of category strings for items with valid categories.

    Only includes items that appear in item_mapping (RecPack's filtered item set).
    Unknown items (no valid categories) are excluded.

    Args:
        articles_df: Articles DataFrame with article_id and category columns
        item_mapping: article_id -> internal_id (RecPack's mapping)
        article_col: Name of article ID column
        categories_col: Name of categories column (list of strings)
        category_str_col: Name of main category column (fallback)

    Returns:
        Tuple of (internal_id_to_categories, all_topics)
        - internal_id_to_categories: Dict mapping internal_id to list of category strings
        - all_topics: Set of all unique topics in the catalog
    """
    internal_id_to_categories: Dict[int, List[str]] = {}
    all_topics: Set[str] = set()

    if articles_df is None or len(articles_df) == 0:
        return internal_id_to_categories, all_topics

    has_categories = categories_col in articles_df.columns
    has_category_str = category_str_col in articles_df.columns

    articles_df = articles_df.copy()
    articles_df[article_col] = articles_df[article_col].astype(str)

    for _, row in articles_df.iterrows():
        article_id = str(row[article_col])
        if article_id not in item_mapping:
            continue

        internal_id = item_mapping[article_id]
        cats: Optional[List[str]] = None

        if has_categories:
            raw = row.get(categories_col)
            cats = _parse_categories_value(raw)

        if not cats and has_category_str:
            cat_str = row.get(category_str_col, "")
            if cat_str is not None and str(cat_str).strip():
                cats = [str(cat_str).strip()]

        if not cats:
            continue

        internal_id_to_categories[internal_id] = cats
        all_topics.update(cats)

    logger.debug(
        f"Built category mapping for {len(internal_id_to_categories)} items, "
        f"{len(all_topics)} unique topics"
    )
    return internal_id_to_categories, all_topics


def _aggregate_exposure_by_topic(
    exposure: np.ndarray,
    internal_id_to_categories: Dict[int, List[str]],
) -> Dict[str, float]:
    """Aggregate item exposure to topic exposure, splitting for multi-topic items.

    For each item i with exposure e_i and categories [c1, c2, ..., cn]:
        topic_exposure[c] += e_i / n  for each c in [c1, c2, ..., cn]

    Args:
        exposure: Item exposure array (shape n_items)
        internal_id_to_categories: internal_id -> list of category strings

    Returns:
        Dict mapping topic -> aggregated exposure
    """
    topic_exposure: Dict[str, float] = {}

    for internal_id in range(exposure.size):
        if internal_id not in internal_id_to_categories:
            continue
        cats = internal_id_to_categories[internal_id]
        if not cats:
            continue
        e = float(exposure[internal_id])
        if e <= 0:
            continue
        share = e / len(cats)
        for c in cats:
            topic_exposure[c] = topic_exposure.get(c, 0.0) + share

    return topic_exposure


def _compute_topic_coverage(topic_exposure: Dict[str, float], all_topics: Set[str]) -> float:
    """Topic coverage = topics with exposure > 0 / total topics in catalog."""
    if not all_topics:
        return 0.0
    n_recommended = sum(1 for t in all_topics if topic_exposure.get(t, 0) > 0)
    return float(n_recommended) / len(all_topics)


def _compute_topic_gini(topic_exposure: Dict[str, float], all_topics: Set[str]) -> float:
    """Gini coefficient on topic exposure vector."""
    if not all_topics:
        return 0.0

    exposure_arr = np.array(
        [topic_exposure.get(t, 0.0) for t in sorted(all_topics)],
        dtype=np.float64,
    )
    exposure_arr = np.clip(exposure_arr, 0.0, None)
    total = exposure_arr.sum()
    if total <= 0:
        return 0.0

    sorted_exposure = np.sort(exposure_arr)
    n = sorted_exposure.size
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(index * sorted_exposure)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_topic_diversity(
    exposure: np.ndarray,
    internal_id_to_categories: Dict[int, List[str]],
    all_topics: Set[str],
    k: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Compute topic-level Coverage and Gini for a given k.

    Returns:
        Tuple of (coverage, gini, topic_exposure_dict)
    """
    topic_exposure = _aggregate_exposure_by_topic(exposure, internal_id_to_categories)
    coverage = _compute_topic_coverage(topic_exposure, all_topics)
    gini = _compute_topic_gini(topic_exposure, all_topics)
    return coverage, gini, topic_exposure


def build_topic_report(
    topic_exposure: Dict[str, float],
    all_topics: Set[str],
    cluster_id: Optional[int] = None,
    algorithm: Optional[str] = None,
    k: Optional[int] = None,
    top_n: int = 20,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """Build topic popularity report with rank and exposure_pct.

    Args:
        topic_exposure: Topic -> exposure mapping
        all_topics: Set of all catalog topics (includes zero-exposure)
        cluster_id: Optional cluster identifier
        algorithm: Optional algorithm name
        k: Optional k value
        top_n: Number of most popular topics to highlight
        bottom_n: Number of least popular topics to highlight

    Returns:
        DataFrame with columns: cluster_id, algorithm, k, topic, exposure, rank,
        exposure_pct, is_most_popular, is_least_popular
    """
    total_exposure = sum(topic_exposure.get(t, 0) for t in all_topics)
    if total_exposure <= 0:
        total_exposure = 1.0

    sorted_topics = sorted(
        all_topics,
        key=lambda t: topic_exposure.get(t, 0.0),
        reverse=True,
    )
    rows = []
    for rank, topic in enumerate(sorted_topics, start=1):
        exp = topic_exposure.get(topic, 0.0)
        pct = 100.0 * exp / total_exposure
        is_most = rank <= top_n
        is_least = rank > len(sorted_topics) - bottom_n if len(sorted_topics) > bottom_n else False
        rows.append({
            "cluster_id": cluster_id,
            "algorithm": algorithm,
            "k": k,
            "topic": topic,
            "exposure": exp,
            "rank": rank,
            "exposure_pct": round(pct, 4),
            "is_most_popular": is_most,
            "is_least_popular": is_least,
        })

    return pd.DataFrame(rows)
