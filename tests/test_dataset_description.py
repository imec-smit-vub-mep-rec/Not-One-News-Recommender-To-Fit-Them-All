"""Tests for dataset description outlier ranking helpers."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dataset_description import (  # noqa: E402
    build_outlier_ranking,
    build_total_impressions_removal_series,
)


@pytest.fixture
def user_stats():
    return pd.DataFrame(
        {
            "total_impressions": [10, 4, 20, 7],
            "article_impressions": [8, 3, 18, 5],
            "homepage_impressions": [2, 1, 2, 2],
            "n_sessions": [3, 2, 5, 2],
        },
        index=["u1", "u2", "u3", "u4"],
    )


def test_build_outlier_ranking_feature_mode_orders_by_selected_feature(user_stats):
    ranking = build_outlier_ranking(user_stats, "total_impressions")

    assert ranking["mode"] == "feature"
    assert ranking["basis"] == "total_impressions"
    assert ranking["ranked_user_ids"] == ["u3", "u1", "u4", "u2"]


def test_build_outlier_ranking_composite_mode_available(user_stats):
    ranking = build_outlier_ranking(user_stats, "composite_score")

    assert ranking["mode"] == "composite"
    assert ranking["basis"] == "composite_score"
    assert len(ranking["ranked_user_ids"]) == len(user_stats)
    assert not ranking["composite_scores"].empty


def test_build_outlier_ranking_rejects_unknown_basis(user_stats):
    with pytest.raises(ValueError, match="Unknown outlier-removal basis"):
        build_outlier_ranking(user_stats, "not_a_metric")


def test_total_impressions_removal_series_has_levels_zero_to_five(user_stats):
    ranked_user_ids = ["u3", "u1", "u4", "u2"]
    series_df = build_total_impressions_removal_series(
        user_stats, ranked_user_ids, removals=[0, 1, 2, 3, 4, 5]
    )

    assert set(series_df["remove_top"].unique()) == {0, 1, 2, 3, 4, 5}
    counts = (
        series_df[series_df["total_impressions"].notna()]
        .groupby("remove_top")["total_impressions"]
        .count()
        .to_dict()
    )
    assert counts[0] == 4
    assert counts[1] == 3
    assert counts[2] == 2
    assert counts[3] == 1
    assert counts[4] == 0
    assert counts[5] == 0
