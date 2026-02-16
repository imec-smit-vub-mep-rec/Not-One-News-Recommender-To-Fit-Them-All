"""Tests for topic-level diversity metrics."""

import numpy as np
import pandas as pd

from src.evaluation.topic_diversity import (
    _normalize_categories,
    _parse_categories_value,
    _aggregate_exposure_by_topic,
    _compute_topic_coverage,
    _compute_topic_gini,
    compute_topic_diversity,
    build_topic_report,
)


def test_parse_categories_value():
    """Test parsing categories from various formats."""
    assert _parse_categories_value(None) is None
    assert _parse_categories_value([]) is None
    assert _parse_categories_value(["sport", "news"]) == ["sport", "news"]
    assert _parse_categories_value(["sport", "", "news"]) == ["sport", "news"]
    assert _parse_categories_value("sport") == ["sport"]
    assert _parse_categories_value('["sport", "news"]') == ["sport", "news"]


def test_normalize_categories():
    """Test building internal_id -> categories mapping."""
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3", "a4"],
        "categories": [["sport", "news"], ["sport"], [], ["tech"]],
        "category_str": ["sport", "sport", "unknown", "tech"],
    })
    item_mapping = {"a1": 0, "a2": 1, "a3": 2, "a4": 3}

    mapping, topics = _normalize_categories(articles, item_mapping)

    assert mapping[0] == ["sport", "news"]
    assert mapping[1] == ["sport"]
    assert mapping[2] == ["unknown"]  # empty categories, fallback to category_str
    assert mapping[3] == ["tech"]
    assert topics == {"sport", "news", "tech", "unknown"}


def test_normalize_categories_fallback_to_category_str():
    """When categories is missing, use category_str."""
    articles = pd.DataFrame({
        "article_id": ["a1"],
        "category_str": ["sport"],
    })
    item_mapping = {"a1": 0}
    mapping, topics = _normalize_categories(articles, item_mapping)
    assert mapping[0] == ["sport"]
    assert topics == {"sport"}


def test_aggregate_exposure_by_topic_multi_topic_split():
    """Multi-topic items split exposure equally."""
    exposure = np.array([2.0, 1.0])
    internal_id_to_categories = {0: ["sport", "news"], 1: ["sport"]}
    topic_exp = _aggregate_exposure_by_topic(exposure, internal_id_to_categories)
    assert topic_exp["sport"] == 2.0 / 2 + 1.0  # 1 from a1 + 1 from a2
    assert topic_exp["news"] == 2.0 / 2


def test_compute_topic_coverage():
    """Topic coverage = topics with exposure > 0 / total topics."""
    all_topics = {"a", "b", "c"}
    assert _compute_topic_coverage({}, all_topics) == 0.0
    assert _compute_topic_coverage({"a": 1.0, "b": 0.5}, all_topics) == 2 / 3
    assert _compute_topic_coverage({"a": 1.0, "b": 1.0, "c": 1.0}, all_topics) == 1.0


def test_compute_topic_gini():
    """Gini in [0, 1]."""
    all_topics = {"a", "b", "c"}
    assert 0 <= _compute_topic_gini({}, all_topics) <= 1
    gini_equal = _compute_topic_gini({"a": 1.0, "b": 1.0, "c": 1.0}, all_topics)
    assert gini_equal == 0.0
    gini_unequal = _compute_topic_gini({"a": 3.0, "b": 0.0, "c": 0.0}, all_topics)
    assert 0 < gini_unequal <= 1


def test_compute_topic_diversity():
    """Full topic diversity computation."""
    articles = pd.DataFrame({
        "article_id": ["a1", "a2"],
        "categories": [["sport", "news"], ["sport"]],
        "category_str": ["sport", "sport"],
    })
    item_mapping = {"a1": 0, "a2": 1}
    mapping, topics = _normalize_categories(articles, item_mapping)
    exposure = np.array([2.0, 1.0])

    coverage, gini, topic_exp = compute_topic_diversity(exposure, mapping, topics, k=10)
    assert 0 <= coverage <= 1
    assert 0 <= gini <= 1
    assert "sport" in topic_exp
    assert "news" in topic_exp


def test_build_topic_report():
    """Topic report has correct structure."""
    topic_exposure = {"sport": 2.0, "news": 1.0, "tech": 0.5}
    all_topics = {"sport", "news", "tech"}
    report = build_topic_report(
        topic_exposure, all_topics, cluster_id=0, algorithm="Popularity", k=10
    )
    assert len(report) == 3
    assert list(report.columns) == [
        "cluster_id", "algorithm", "k", "topic", "exposure", "rank",
        "exposure_pct", "is_most_popular", "is_least_popular",
    ]
    assert report["rank"].tolist() == [1, 2, 3]
    assert abs(report["exposure_pct"].sum() - 100.0) < 0.01
