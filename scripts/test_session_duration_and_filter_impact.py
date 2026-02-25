#!/usr/bin/env python3
"""
Test script to compare the impact of:

1. avg_session_duration: read_time_sum vs timestamp_span
2. Session filter: drop entire sessions with >50 total impressions vs
   drop only article rows from sessions with >50 article impressions (legacy)
3. Extra: exclude entire sessions with >50 article impressions OR >100 total impressions
   

Run: python scripts/test_session_duration_and_filter_impact.py

Uses synthetic data if no real data path is provided.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root (for optional imports)
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_synthetic_data(n_users: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic impressions with edge cases for testing."""
    np.random.seed(seed)
    rows = []

    for u in range(n_users):
        n_sessions = np.random.randint(2, 15)
        for s in range(n_sessions):
            session_id = f"u{u}_s{s}"
            # Mix of article-heavy and homepage-heavy sessions
            n_article = np.random.randint(0, 80)  # Some sessions can have many articles
            n_homepage = np.random.randint(0, 30)
            n_total = n_article + n_homepage

            base_ts = 1_700_000_000_000 + u * 86400 * 1000 + s * 3600 * 1000  # ms

            # Article impressions
            for i in range(n_article):
                rows.append({
                    "user_id": f"u{u}",
                    "session_id": session_id,
                    "article_id": f"a{np.random.randint(0, 100)}",
                    "impression_time": base_ts + i * 60_000,  # 1 min apart
                    "read_time": np.random.uniform(5, 120),  # 5-120 sec read time
                    "category_str": np.random.choice(["sport", "politics", "tech"]),
                })

            # Homepage impressions (article_id is null)
            for i in range(n_homepage):
                rows.append({
                    "user_id": f"u{u}",
                    "session_id": session_id,
                    "article_id": np.nan,
                    "impression_time": base_ts + n_article * 60_000 + i * 30_000,
                    "read_time": np.random.uniform(2, 30),
                    "category_str": "",
                })

    df = pd.DataFrame(rows)
    # Ensure impression_time is numeric (ms)
    df["impression_time"] = df["impression_time"].astype(np.float64)
    return df


def filter_legacy(df: pd.DataFrame, max_article_impressions: int = 50) -> pd.DataFrame:
    """
    Legacy: Remove article rows from sessions with >50 article impressions.
    Homepage rows are kept.
    """
    with_article = df[df["article_id"].notna()]
    without_article = df[df["article_id"].isna()]

    session_article_counts = with_article.groupby("session_id").size()
    valid_sessions = session_article_counts[session_article_counts <= max_article_impressions].index
    with_article_filtered = with_article[with_article["session_id"].isin(valid_sessions)]

    return pd.concat([with_article_filtered, without_article], ignore_index=True)


def filter_new(df: pd.DataFrame, max_impressions: int = 50) -> pd.DataFrame:
    """
    New pipeline: Remove entire sessions with >50 total impressions (articles + homepage).
    """
    session_counts = df.groupby("session_id").size()
    valid_sessions = session_counts[session_counts <= max_impressions].index
    return df[df["session_id"].isin(valid_sessions)].copy()


def filter_article_or_total(
    df: pd.DataFrame,
    max_article_impressions: int = 50,
    max_total_impressions: int = 100,
) -> pd.DataFrame:
    """
    Exclude entire sessions with >50 article impressions OR >100 total impressions.
    """
    session_total = df.groupby("session_id").size()
    with_article = df[df["article_id"].notna()]
    session_article = with_article.groupby("session_id").size().reindex(session_total.index, fill_value=0)

    exclude = (session_article > max_article_impressions) | (session_total > max_total_impressions)
    valid_sessions = session_total.index[~exclude]
    return df[df["session_id"].isin(valid_sessions)].copy()


def avg_session_duration_read_time_sum(df: pd.DataFrame) -> pd.Series:
    """New: sum of read_time per session, averaged per user."""
    session_read_time = df.groupby(["user_id", "session_id"])["read_time"].sum()
    return session_read_time.groupby("user_id").mean()


def avg_session_duration_timestamp_span(df: pd.DataFrame) -> pd.Series:
    """Legacy: (max - min) impression_time per session in seconds, averaged per user."""
    # impression_time is in ms
    dt = pd.to_datetime(df["impression_time"], unit="ms")
    session_span = (
        df.assign(_dt=dt)
        .groupby(["user_id", "session_id"])["_dt"]
        .agg(lambda x: (x.max() - x.min()).total_seconds())
    )
    return session_span.groupby("user_id").mean()


def run_comparison(df: pd.DataFrame) -> None:
    """Run and print comparison."""
    print("=" * 70)
    print("IMPACT COMPARISON: Session filter + avg_session_duration definition")
    print("=" * 70)

    # --- Session filter impact ---
    df_legacy = filter_legacy(df)
    df_new = filter_new(df)
    df_article_or_total = filter_article_or_total(df)

    print("\n--- 1. SESSION FILTER IMPACT ---")
    print(f"Original:           {len(df):,} rows, {df['session_id'].nunique():,} sessions, {df['user_id'].nunique():,} users")
    print(f"Legacy filter:      {len(df_legacy):,} rows, {df_legacy['session_id'].nunique():,} sessions, {df_legacy['user_id'].nunique():,} users")
    print(f"New filter (>50):   {len(df_new):,} rows, {df_new['session_id'].nunique():,} sessions, {df_new['user_id'].nunique():,} users")
    print(f">50 art OR >100 tot:{len(df_article_or_total):,} rows, {df_article_or_total['session_id'].nunique():,} sessions, {df_article_or_total['user_id'].nunique():,} users")

    rows_diff = len(df_legacy) - len(df_new)
    sessions_diff = df_legacy["session_id"].nunique() - df_new["session_id"].nunique()
    users_diff = df_legacy["user_id"].nunique() - df_new["user_id"].nunique()
    print(f"\nLegacy keeps {rows_diff:+,} more rows, {sessions_diff:+,} more sessions, {users_diff:+,} more users vs new (>50)")

    # --- avg_session_duration impact (on same filtered data) ---
    print("\n--- 2. AVG_SESSION_DURATION: read_time_sum vs timestamp_span ---")
    print("(Computed on same data: legacy-filtered)")

    dur_read = avg_session_duration_read_time_sum(df_legacy)
    dur_ts = avg_session_duration_timestamp_span(df_legacy)

    users_common = dur_read.index.intersection(dur_ts.index)
    dur_read = dur_read.reindex(users_common, fill_value=np.nan)
    dur_ts = dur_ts.reindex(users_common, fill_value=np.nan)

    valid = dur_read.notna() & dur_ts.notna()
    diff = dur_read - dur_ts
    pct_diff = diff / (dur_ts + 1e-10) * 100

    print(f"\nPer-user comparison (n={valid.sum()} users with both values):")
    print(f"  read_time_sum: mean={dur_read[valid].mean():.1f}s, median={dur_read[valid].median():.1f}s")
    print(f"  timestamp_span: mean={dur_ts[valid].mean():.1f}s, median={dur_ts[valid].median():.1f}s")
    print(f"  Mean difference: {diff[valid].mean():.1f}s (read_time_sum - timestamp_span)")
    print(f"  Median abs diff: {diff[valid].abs().median():.1f}s")
    print(f"  Max abs diff: {diff[valid].abs().max():.1f}s")
    print(f"  Users with >50% relative diff: {(pct_diff[valid].abs() > 50).sum()}")

    # --- Combined impact: filter + duration on same data ---
    print("\n--- 3. COMBINED: Filter impact on avg_session_duration ---")
    dur_legacy_filter_read = avg_session_duration_read_time_sum(df_legacy)
    dur_new_filter_read = avg_session_duration_read_time_sum(df_new)
    dur_article_or_total_read = avg_session_duration_read_time_sum(df_article_or_total)
    dur_legacy_filter_ts = avg_session_duration_timestamp_span(df_legacy)
    dur_new_filter_ts = avg_session_duration_timestamp_span(df_new)
    dur_article_or_total_ts = avg_session_duration_timestamp_span(df_article_or_total)

    users_legacy = set(df_legacy["user_id"].unique())
    users_new = set(df_new["user_id"].unique())
    users_article_or_total = set(df_article_or_total["user_id"].unique())
    users_with_both = list(users_legacy & users_new)
    if users_with_both:
        u_common = list(users_with_both)[:10]
        print("\nSample users (legacy-filtered vs new-filtered):")
        for u in u_common:
            r_leg = dur_legacy_filter_read.get(u, np.nan)
            r_new = dur_new_filter_read.get(u, np.nan)
            t_leg = dur_legacy_filter_ts.get(u, np.nan)
            t_new = dur_new_filter_ts.get(u, np.nan)
            print(f"  {u}: read_sum={r_leg:.0f}/{r_new:.0f}s, ts_span={t_leg:.0f}/{t_new:.0f}s")

    print("\nAggregate avg_session_duration (all users):")
    print("  Legacy filter:      read_sum mean={:.1f}s median={:.1f}s | ts_span mean={:.1f}s median={:.1f}s".format(
        dur_legacy_filter_read.mean(), dur_legacy_filter_read.median(),
        dur_legacy_filter_ts.mean(), dur_legacy_filter_ts.median()))
    print("  New filter (>50):   read_sum mean={:.1f}s median={:.1f}s | ts_span mean={:.1f}s median={:.1f}s".format(
        dur_new_filter_read.mean(), dur_new_filter_read.median(),
        dur_new_filter_ts.mean(), dur_new_filter_ts.median()))
    print("  >50 art OR >100 tot:read_sum mean={:.1f}s median={:.1f}s | ts_span mean={:.1f}s median={:.1f}s".format(
        dur_article_or_total_read.mean(), dur_article_or_total_read.median(),
        dur_article_or_total_ts.mean(), dur_article_or_total_ts.median()))
    print("\n  (Users only in legacy: {}, only in new: {}, in >50art|>100tot: {})".format(
        len(users_legacy - users_new), len(users_new - users_legacy), len(users_article_or_total)))

    if users_with_both:
        print("\n  Users in BOTH filters (n={}):".format(len(users_with_both)))
        u_both = pd.Index(users_with_both)
        r_leg_both = dur_legacy_filter_read.reindex(u_both).dropna()
        r_new_both = dur_new_filter_read.reindex(u_both).dropna()
        t_leg_both = dur_legacy_filter_ts.reindex(u_both).dropna()
        t_new_both = dur_new_filter_ts.reindex(u_both).dropna()
        print("    Legacy: read_sum mean={:.1f}s | ts_span mean={:.1f}s".format(r_leg_both.mean(), t_leg_both.mean()))
        print("    New:    read_sum mean={:.1f}s | ts_span mean={:.1f}s".format(r_new_both.mean(), t_new_both.mean()))
    else:
        print("\nNo users in both filtered datasets (new filter may remove all data for some users).")

    print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare session filter and avg_session_duration impact")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to behaviors.parquet or impressions.parquet (optional)",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=500,
        help="Number of synthetic users for synthetic data (default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data",
    )
    args = parser.parse_args()

    if args.data:
        path = Path(args.data)
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        df = pd.read_parquet(path)
        if "impression_time" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "impression_time"})
        required = ["user_id", "session_id", "article_id", "impression_time", "read_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: missing columns {missing} in {path}")
            sys.exit(1)
        if "category_str" not in df.columns:
            df["category_str"] = ""
        print(f"Loaded {len(df):,} rows from {path}")
    else:
        print(f"Using synthetic data (n_users={args.n_users}, seed={args.seed})")
        df = make_synthetic_data(n_users=args.n_users, seed=args.seed)

    run_comparison(df)


if __name__ == "__main__":
    main()
