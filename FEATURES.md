# Clustering Features

This document lists all features used for user clustering, both in **legacy mode** and **new mode**.

## Mode Selection

- **Legacy mode**: Enable with `--legacy-features` (CLI) or `"legacy_features": true` in config.
- **New mode**: Default when `legacy_features` is false.
- **Log transform (optional)**: Disabled by default. Enable with `--log-transform` (CLI) or `"use_log_transform": true` in `clustering` config.

---

## Legacy Mode

Legacy mode uses a fixed set of 12 features that match the clustering pipeline from the original short paper. It excludes per-category proportions, time-of-day preferences, and diversity metrics (entropy, gini).

| #   | Feature                      | Type    | Processing                    | Description                                                               |
| --- | ---------------------------- | ------- | ----------------------------- | ------------------------------------------------------------------------- |
| 1   | `num_sessions`               | int     | StandardScaler (default), optional log1p + StandardScaler | Total number of sessions                                                  |
| 2   | `total_impressions`          | int     | StandardScaler (default), optional log1p + StandardScaler | Total number of impressions (homepage + articles)                         |
| 3   | `homepage_impressions`       | int     | StandardScaler (default), optional log1p + StandardScaler | Number of homepage impressions                                            |
| 4   | `article_impressions`        | int     | StandardScaler (default), optional log1p + StandardScaler | Number of article impressions                                             |
| 5   | `num_categories`             | int     | StandardScaler only            | Number of unique categories viewed                                        |
| 6   | `unique_articles`            | int     | StandardScaler (default), optional log1p + StandardScaler | Number of unique articles viewed                                          |
| 7   | `avg_reading_time`           | float   | StandardScaler (default), optional log1p + StandardScaler | Average reading time per impression (seconds)                             |
| 8   | `proportion_article_time`    | float   | StandardScaler only            | Proportion of total reading time spent on articles vs homepage            |
| 9   | `avg_session_length`         | float   | StandardScaler (default), optional log1p + StandardScaler | Average impressions per session                                           |
| 10  | `avg_categories_per_session` | float   | StandardScaler only            | Average unique categories per session                                     |
| 11  | `avg_category_switches`      | float   | StandardScaler only            | Average top category switches per session                                  |
| 12  | `avg_session_duration`       | float   | StandardScaler (default), optional log1p + StandardScaler | Average total read_time per session (sum of read_time across impressions) |

**Note:** The first iteration of the historical legacy code in `00_legacy/1-user-clustering/helpers/_old_user_clustering_addressa.py` also used time-of-day features (`percentage_morning`, `percentage_afternoon`, `percentage_evening`, `percentage_night`). The current legacy mode intentionally excludes these for consistency with the refactored pipeline.

---

## New Mode

New mode uses an extended feature set that includes per-category proportions, time-of-day preferences, and richer diversity metrics.

### 1. Activity Features

| Feature                   | Type  | Processing            | Description                                      |
| ------------------------- | ----- | --------------------- | ------------------------------------------------ |
| `unique_articles`         | int   | StandardScaler (default), optional log1p + StandardScaler | Number of unique articles viewed                 |
| `total_impressions`       | int   | StandardScaler (default), optional log1p + StandardScaler | Total number of impressions                      |
| `num_sessions`            | int   | StandardScaler (default), optional log1p + StandardScaler | Total number of sessions                         |
| `impressions_per_session` | float | StandardScaler (default), optional log1p + StandardScaler | Average impressions per session                  |
| `engagement_span_days`    | float | StandardScaler only    | Number of days between first and last impression |

### 2. Homepage Features

| Feature                     | Type  | Processing             | Description                                        |
| --------------------------- | ----- | ---------------------- | -------------------------------------------------- |
| `homepage_impressions`      | int   | StandardScaler (default), optional log1p + StandardScaler | Number of homepage impressions                     |
| `article_impressions`       | int   | StandardScaler (default), optional log1p + StandardScaler | Number of article impressions                      |
| `homepage_ratio`            | float | StandardScaler only    | Proportion of impressions on homepage              |
| `avg_reading_time_homepage` | float | StandardScaler only    | Average reading time on homepage (seconds)         |
| `avg_reading_time_articles` | float | StandardScaler only    | Average reading time on articles (seconds)         |
| `proportion_article_time`   | float | StandardScaler only    | Proportion of total reading time spent on articles |

### 3. Category Features (per-category proportions)

| Feature          | Type  | Processing         | Description                                                                                            |
| ---------------- | ----- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| `cat_{category}` | float | StandardScaler only | Proportion of impressions in each category (one column per category, e.g. `cat_sport`, `cat_politics`) |

_Not included in legacy mode._

### 4. Time Features (time-of-day preferences)

| Feature          | Type  | Processing         | Description                                              |
| ---------------- | ----- | ------------------ | -------------------------------------------------------- |
| `time_morning`   | float | StandardScaler only | Proportion of impressions during morning (6:00–12:00)    |
| `time_afternoon` | float | StandardScaler only | Proportion of impressions during afternoon (12:00–18:00) |
| `time_evening`   | float | StandardScaler only | Proportion of impressions during evening (18:00–24:00)   |
| `time_night`     | float | StandardScaler only | Proportion of impressions during night (0:00–6:00)       |
| `time_weekend`   | float | StandardScaler only | Proportion of impressions on weekend days                |

_Not included in legacy mode._

### 5. Diversity Features

| Feature            | Type  | Processing         | Description                                          |
| ------------------ | ----- | ------------------ | ---------------------------------------------------- |
| `category_entropy` | float | StandardScaler only | Entropy of category distribution (content diversity) |
| `num_categories`   | int   | StandardScaler only | Number of unique categories viewed                   |
| `category_gini`    | float | StandardScaler only | Gini coefficient of category distribution            |

_In legacy mode, only `num_categories` is used; entropy and gini are excluded._

### 6. Session Behavior Features

| Feature                      | Type  | Processing            | Description                           |
| ---------------------------- | ----- | --------------------- | ------------------------------------- |
| `avg_reading_time`           | float | StandardScaler (default), optional log1p + StandardScaler | Average reading time per impression   |
| `avg_session_length`         | float | StandardScaler (default), optional log1p + StandardScaler | Average impressions per session       |
| `avg_categories_per_session` | float | StandardScaler only    | Average unique categories per session |
| `avg_category_switches`      | float | StandardScaler only    | Average category switches per session |
| `avg_session_duration`       | float | StandardScaler (default), optional log1p + StandardScaler | Average total read_time per session   |

_In new mode, these are **not** included by default. They are only used in legacy mode._

### 7. Subscriber Feature (optional)

| Feature         | Type  | Processing         | Description                                 |
| --------------- | ----- | ------------------ | ------------------------------------------- |
| `is_subscriber` | int   | StandardScaler only | Whether the user is a paid subscriber (1/0) |

_Not used for clustering by default; available for post-hoc reporting. Can be enabled via `include_subscriber=True` in `UserFeatureExtractor`._

---

## Summary: Legacy vs New

| Feature Group             | Legacy Mode             | New Mode     |
| ------------------------- | ----------------------- | ------------ |
| Activity (core counts)    | ✓ (subset)              | ✓            |
| Homepage behavior         | ✓                       | ✓            |
| Per-category proportions  | ✗                       | ✓            |
| Time-of-day               | ✗                       | ✓            |
| Diversity (entropy, gini) | ✗ (only num_categories) | ✓            |
| Session behavior          | ✓                       | ✗            |
| Subscriber                | ✗                       | ✗ (optional) |

---

## Configuration

- **Default**: StandardScaler without log transform (`use_log_transform: false`)
- **CLI (enable log transform)**: `python scripts/run_full_pipeline.py ... --log-transform`
- **Config JSON (enable log transform)**: `"clustering": { "use_log_transform": true }`
- **Code**: `UserFeatureExtractor(..., use_log_transform=True)` or `create_user_features(..., use_log_transform=True)`
