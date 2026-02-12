"""Tests for AD S3 converter."""

import types
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import DatasetConfig
from src.converters.ad import ADConverter


class _FakeS3:
    def read_csv(self, path, **kwargs):
        if path.endswith("article_metadata.csv"):
            return pd.DataFrame(
                {
                    "ARTICLE_IDENTIFIER": ["a1", "a2"],
                    "title": ["T1", "T2"],
                    "main_section": ["regio", ""],
                    "categories": ["['regio' 'x']", "['sport' 'y']"],
                    "first_publication_timestamp": [
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T01:00:00+00:00",
                    ],
                    "bert_embedding": ["[0.1 0.2 0.3]", "[0.3 0.4 0.5]"],
                }
            )

        return pd.DataFrame(
            {
                "ARTICLE_IDENTIFIER": ["art1", "art2"],
                "IMPRESSION_ID": ["i1", "i2"],
                "START_TIME": ["2026-01-01T00:00:00.000Z", "2026-01-01T00:00:01.000Z"],
                "SESSION_ID": ["s1", "s2"],
                "IS_LOGGED_IN": ["true", "false"],
                "MAPPED_USER_IDENTIFIER": ["u1", "u2"],
                "TIME_ON_PAGE": [10, 20],
                "event_type": ["article_page_view", "home_page_view"],
            }
        )


def _install_fake_awswrangler(monkeypatch):
    fake = types.SimpleNamespace(s3=_FakeS3())
    monkeypatch.setitem(sys.modules, "awswrangler", fake)


def test_convert_articles_maps_columns(monkeypatch):
    _install_fake_awswrangler(monkeypatch)
    converter = ADConverter(
        DatasetConfig(name="ad", input_path="s3://bucket/ad", format="spark_csv")
    )

    articles = converter.convert_articles()

    assert set(["article_id", "title", "category_str", "time_published"]).issubset(articles.columns)
    assert articles.loc[0, "article_id"] == "a1"
    assert articles.loc[0, "category_str"] == "regio"
    # main_section is empty for second row, should fallback to categories
    assert articles.loc[1, "category_str"] == "sport"


def test_convert_impressions_homepage_article_null(monkeypatch):
    _install_fake_awswrangler(monkeypatch)
    converter = ADConverter(
        DatasetConfig(name="ad", input_path="s3://bucket/ad", format="spark_csv")
    )

    impressions = converter.convert_impressions()

    assert set(["user_id", "article_id", "impression_time", "session_id"]).issubset(impressions.columns)
    assert impressions["impression_time"].dtype.name in ("Int64", "int64")
    assert impressions.loc[0, "article_id"] == "art1"
    assert pd.isna(impressions.loc[1, "article_id"])
