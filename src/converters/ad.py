"""
AD dataset converter for S3 Spark-partitioned CSV exports.

Converts:
- ad/article_metadata.csv
- ad/impressions/event_type=.../*.csv

to the pipeline's standard schema.
"""

from __future__ import annotations

from typing import Any, Optional
import ast
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .base import BaseConverter
from ..config.settings import DatasetConfig


class ADConverter(BaseConverter):
    """Converter for AD dataset stored on S3 as Spark CSV output."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    @staticmethod
    def _load_dpg_env_from_dotenv() -> None:
        """
        Load DPG_* credentials from a local .env file if not exported.

        This is a lightweight fallback to avoid requiring python-dotenv.
        Existing process environment variables are never overwritten.
        """
        dotenv_path = Path(".env")
        if not dotenv_path.exists():
            return

        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in {
                    "DPG_ACCESS_KEY_ID",
                    "DPG_SECRET_ACCESS_KEY",
                    "DPG_SESSION_TOKEN",
                    "DPG_REGION",
                }:
                    continue
                value = value.strip()
                if (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
        except Exception:
            # Non-fatal: continue with default environment chain.
            return

    @staticmethod
    def _normalize_s3_root(path: str) -> str:
        """Ensure S3 root path has no trailing slash."""
        return str(path).rstrip("/")

    @staticmethod
    def _extract_bucket_from_s3_uri(s3_uri: str) -> Optional[str]:
        """Extract bucket name from an s3:// URI."""
        if not s3_uri:
            return None
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            return None
        return parsed.netloc or None

    def _build_boto3_session(self):
        """
        Configure AWS auth env vars from custom DPG env vars.

        Returns:
            None:
                awswrangler will use environment/default credential chain.
        """
        access_key = os.getenv("DPG_ACCESS_KEY_ID")
        secret_key = os.getenv("DPG_SECRET_ACCESS_KEY")
        session_token = os.getenv("DPG_SESSION_TOKEN")

        if not access_key or not secret_key:
            self._load_dpg_env_from_dotenv()
            access_key = os.getenv("DPG_ACCESS_KEY_ID")
            secret_key = os.getenv("DPG_SECRET_ACCESS_KEY")
            session_token = os.getenv("DPG_SESSION_TOKEN")

        # If custom vars are not set, use default credential chain.
        if not access_key or not secret_key:
            self.logger.info(
                "DPG_ACCESS_KEY_ID/DPG_SECRET_ACCESS_KEY not set; using default AWS credential chain"
            )
            return None

        # Try to auto-derive region from bucket location.
        region = None
        bucket = self._extract_bucket_from_s3_uri(self.config.input_path)
        try:
            import boto3

            base_session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
            )
            if bucket:
                try:
                    s3_client = base_session.client("s3", region_name="us-east-1")
                    location = s3_client.get_bucket_location(Bucket=bucket).get("LocationConstraint")
                    region = "us-east-1" if location in (None, "") else location
                    self.logger.info(f"Derived AWS region from bucket '{bucket}': {region}")
                except Exception as exc:
                    self.logger.warning(f"Could not derive region from bucket '{bucket}': {exc}")
        except Exception:
            # boto3 is optional here; we can still use fallback region/env credentials.
            pass

        if not region:
            region = (
                os.getenv("DPG_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
                or os.getenv("AWS_REGION")
                or "us-east-1"
            )
            self.logger.info(f"Using fallback AWS region: {region}")

        # Ray engine in awswrangler does not support boto3_session argument.
        # Expose credentials/region via standard AWS env vars instead.
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
        if session_token:
            os.environ.setdefault("AWS_SESSION_TOKEN", session_token)
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_REGION", region)
        self.logger.info("Configured AWS credentials from DPG_* environment variables")
        return None

    @staticmethod
    def _to_bool(series: pd.Series) -> pd.Series:
        """Convert mixed string/bool/int values to bool."""
        if series.dtype == bool:
            return series
        normalized = series.astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes", "y"})

    @staticmethod
    def _parse_categories_fallback(raw_value: Any) -> str:
        """
        Parse `categories` column and return first non-empty category.

        Expected inputs vary, e.g.:
        - "['regio' 'rijswijk']"
        - "['regio', 'rijswijk']"
        - JSON-like arrays
        """
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            return ""
        text = str(raw_value).strip()
        if not text:
            return ""

        # Try JSON first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except Exception:
            pass

        # Then try Python literal style.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
            if isinstance(parsed, tuple) and parsed:
                return str(parsed[0]).strip()
        except Exception:
            pass

        # Fallback for space-separated quote format.
        cleaned = text.strip("[]")
        for token in cleaned.replace(",", " ").split():
            token = token.strip("'\"")
            if token:
                return token
        return ""

    @staticmethod
    def _normalize_colname(name: Any) -> str:
        """Normalize source column names for robust matching."""
        return str(name).strip().upper()

    def convert_articles(self) -> pd.DataFrame:
        """Convert AD article metadata to standard format."""
        try:
            import awswrangler as wr
        except ImportError as exc:
            raise ImportError("awswrangler is required for AD S3 conversion.") from exc

        root = self._normalize_s3_root(self.config.input_path)
        articles_path = f"{root}/article_metadata.csv"
        self.logger.info(f"Loading AD article metadata from {articles_path}")
        self._build_boto3_session()

        read_kwargs = {
            "path": articles_path,
            "use_threads": True,
        }

        df = wr.s3.read_csv(**read_kwargs)
        self.logger.info(f"Loaded {len(df)} raw articles")
        article_col_map = {self._normalize_colname(col): col for col in df.columns}

        # Standard mappings
        if "ARTICLE_IDENTIFIER" in df.columns:
            df["article_id"] = df["ARTICLE_IDENTIFIER"].astype(str)
        else:
            raise ValueError("Missing required column 'ARTICLE_IDENTIFIER' in article_metadata.csv")

        df["title"] = df.get("title", "").fillna("").astype(str)

        if "main_section" in df.columns:
            df["category_str"] = df["main_section"].fillna("").astype(str).str.strip()
        else:
            df["category_str"] = ""

        # Optional fallback if main_section is empty.
        if "categories" in df.columns:
            empty_mask = df["category_str"].eq("")
            if empty_mask.any():
                df.loc[empty_mask, "category_str"] = df.loc[empty_mask, "categories"].map(
                    self._parse_categories_fallback
                )

        if "first_publication_timestamp" in df.columns:
            df["time_published"] = pd.to_datetime(
                df["first_publication_timestamp"], errors="coerce", utc=True
            ).dt.tz_convert(None)
        else:
            df["time_published"] = pd.NaT

        # Preserve optional raw embedding for session-local conversion later.
        if "bert_embedding" in df.columns:
            df["bert_embedding"] = df["bert_embedding"]

        if "article_word_count" in df.columns:
            df["article_length"] = pd.to_numeric(df["article_word_count"], errors="coerce").fillna(0)

        # Keep paywall marker when available for metered subscriber derivation.
        if "IS_PAYWALL" in article_col_map:
            df["is_paywall"] = self._to_bool(df[article_col_map["IS_PAYWALL"]]).astype(bool)
        else:
            df["is_paywall"] = False

        # Use schema default if absent later, but set explicit value for clarity.
        df["sentiment_score"] = 0.5

        return df

    def convert_impressions(self) -> pd.DataFrame:
        """Convert AD partitioned impressions to standard format."""
        try:
            import awswrangler as wr
        except ImportError as exc:
            raise ImportError("awswrangler is required for AD S3 conversion.") from exc

        root = self._normalize_s3_root(self.config.input_path)
        impressions_root = f"{root}/impressions/"
        self.logger.info(f"Loading AD impressions from {impressions_root}")
        self._build_boto3_session()

        expected_columns = [
            "ARTICLE_IDENTIFIER",
            "IMPRESSION_ID",
            "START_TIME",
            "SESSION_ID",
            "IS_LOGGED_IN",
            "MAPPED_USER_IDENTIFIER",
            "TIME_ON_PAGE",
        ]

        event_types = self.config.event_types or ["home_page_view", "article_page_view"]
        event_types_set = set(event_types)

        def partition_filter(partitions: dict[str, str]) -> bool:
            return partitions.get("event_type") in event_types_set

        read_kwargs = {
            "path": impressions_root,
            "dataset": True,
            "use_threads": True,
            # Use a callable so case/whitespace differences do not fail reads.
            "usecols": lambda c: self._normalize_colname(c) in set(expected_columns),
            "partition_filter": partition_filter,
        }

        df = wr.s3.read_csv(**read_kwargs)
        self.logger.info(f"AD impressions raw columns: {list(df.columns)}")

        column_map = {self._normalize_colname(col): col for col in df.columns}
        self.logger.info(f"AD impressions normalized columns: {list(column_map.keys())}")
        required = [
            "MAPPED_USER_IDENTIFIER",
            "SESSION_ID",
            "IMPRESSION_ID",
            "START_TIME",
            "IS_LOGGED_IN",
            "TIME_ON_PAGE",
            "ARTICLE_IDENTIFIER",
        ]
        missing = [name for name in required if name not in column_map]
        if missing:
            self.logger.error(
                "Missing required impression columns after normalization. "
                f"Missing={missing}, raw_columns={list(df.columns)}, "
                f"normalized_columns={list(column_map.keys())}"
            )
            raise ValueError(
                "Missing required impression columns after normalization: "
                f"{missing}. Available columns: {list(df.columns)}"
            )

        if "EVENT_TYPE" in column_map and "event_type" not in df.columns:
            df["event_type"] = df[column_map["EVENT_TYPE"]]

        if "event_type" not in df.columns:
            raise ValueError(
                "Partition column 'event_type' not found. "
                "Ensure impressions are read with dataset=True from the partition root."
            )

        # Optional timestamp filters for smoke tests.
        start_time_col = column_map["START_TIME"]
        user_col = column_map["MAPPED_USER_IDENTIFIER"]
        session_col = column_map["SESSION_ID"]
        impression_col = column_map["IMPRESSION_ID"]
        is_logged_in_col = column_map["IS_LOGGED_IN"]
        time_on_page_col = column_map["TIME_ON_PAGE"]
        article_col = column_map["ARTICLE_IDENTIFIER"]

        timestamps = pd.to_datetime(df[start_time_col], errors="coerce", utc=True)
        if self.config.start_time_min:
            min_ts = pd.Timestamp(self.config.start_time_min, tz="UTC")
            df = df.loc[timestamps >= min_ts].copy()
            timestamps = timestamps.loc[df.index]
        if self.config.start_time_max:
            max_ts = pd.Timestamp(self.config.start_time_max, tz="UTC")
            df = df.loc[timestamps <= max_ts].copy()
            timestamps = timestamps.loc[df.index]

        self.logger.info(f"Loaded {len(df)} raw impressions")

        out = pd.DataFrame()
        out["user_id"] = df[user_col].astype("string")
        out["session_id"] = df[session_col].astype("string")
        out["impression_id"] = df[impression_col].astype("string")
        out["read_time"] = pd.to_numeric(df[time_on_page_col], errors="coerce").fillna(0.0)
        out["is_logged_in"] = self._to_bool(df[is_logged_in_col]).astype(bool)

        # Convert RFC3339 timestamps to Unix milliseconds.
        ms = (timestamps.view("int64") // 10**6).astype("Int64")
        ms = ms.where(timestamps.notna(), pd.NA)
        out["impression_time"] = ms

        # Homepage rows must have null article_id for feature_engineering homepage detection.
        out["article_id"] = df[article_col].astype("string")
        homepage_mask = df["event_type"].astype(str).eq("home_page_view")
        out.loc[homepage_mask, "article_id"] = pd.NA

        # Drop rows missing required IDs/time after parsing.
        required_mask = (
            out["user_id"].notna()
            & out["session_id"].notna()
            & out["impression_id"].notna()
            & out["impression_time"].notna()
        )
        dropped = int((~required_mask).sum())
        if dropped:
            self.logger.warning(f"Dropping {dropped} impressions with missing required fields")
            out = out.loc[required_mask].copy()

        # Ensure article IDs are strings when present.
        if out["article_id"].notna().any():
            out.loc[out["article_id"].notna(), "article_id"] = (
                out.loc[out["article_id"].notna(), "article_id"].astype(str)
            )

        return out
