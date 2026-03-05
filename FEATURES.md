# Clustering Features

This document lists all features used for user clustering, both in **legacy mode** and **new mode**.

## Mode Selection

- **Legacy mode**: Enable with `--legacy-features` (CLI) or `"legacy_features": true` in config.
- **New mode**: Default when `legacy_features` is false.

---

## Legacy Mode

Legacy mode uses a fixed set of 12 features that match the clustering pipeline from the original short paper. It excludes per-category proportions, time-of-day preferences, and diversity metrics (entropy, gini).

| #   | Feature                      | Description                                                               |
| --- | ---------------------------- | ------------------------------------------------------------------------- |
| 1   | `num_sessions`               | Total number of sessions                                                  |
| 2   | `total_impressions`          | Total number of impressions (homepage + articles)                         |
| 3   | `homepage_impressions`       | Number of homepage impressions                                            |
| 4   | `article_impressions`        | Number of article impressions                                             |
| 5   | `num_categories`             | Number of unique categories viewed                                        |
| 6   | `unique_articles`            | Number of unique articles viewed                                          |
| 7   | `avg_reading_time`           | Average reading time per impression (seconds)                             |
| 8   | `proportion_article_time`    | Proportion of total reading time spent on articles vs homepage            |
| 9   | `avg_session_length`         | Average impressions per session                                           |
| 10  | `avg_categories_per_session` | Average unique categories per session                                     |
| 11  | `avg_category_switches`      | Average top category switches per session                                 |
| 12  | `avg_session_duration`       | Average total read_time per session (sum of read_time across impressions) |

**Note:** The first iteration of the historical legacy code in `00_legacy/1-user-clustering/helpers/_old_user_clustering_addressa.py` also used time-of-day features (`percentage_morning`, `percentage_afternoon`, `percentage_evening`, `percentage_night`). The current legacy mode intentionally excludes these for consistency with the refactored pipeline.

---

## New Mode

New mode uses an extended feature set that includes per-category proportions, time-of-day preferences, and richer diversity metrics.

### 1. Activity Features

| Feature                   | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `unique_articles`         | Number of unique articles viewed                 |
| `total_impressions`       | Total number of impressions                      |
| `num_sessions`            | Total number of sessions                         |
| `impressions_per_session` | Average impressions per session                  |
| `engagement_span_days`    | Number of days between first and last impression |

### 2. Homepage Features

| Feature                     | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `homepage_impressions`      | Number of homepage impressions                     |
| `article_impressions`       | Number of article impressions                      |
| `homepage_ratio`            | Proportion of impressions on homepage              |
| `avg_reading_time_homepage` | Average reading time on homepage (seconds)         |
| `avg_reading_time_articles` | Average reading time on articles (seconds)         |
| `proportion_article_time`   | Proportion of total reading time spent on articles |

### 3. Category Features (per-category proportions)

| Feature          | Description                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `cat_{category}` | Proportion of impressions in each category (one column per category, e.g. `cat_sport`, `cat_politics`) |

_Not included in legacy mode._

### 4. Time Features (time-of-day preferences)

| Feature          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `time_morning`   | Proportion of impressions during morning (6:00–12:00)    |
| `time_afternoon` | Proportion of impressions during afternoon (12:00–18:00) |
| `time_evening`   | Proportion of impressions during evening (18:00–24:00)   |
| `time_night`     | Proportion of impressions during night (0:00–6:00)       |
| `time_weekend`   | Proportion of impressions on weekend days                |

_Not included in legacy mode._

### 5. Diversity Features

| Feature            | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `category_entropy` | Entropy of category distribution (content diversity) |
| `num_categories`   | Number of unique categories viewed                   |
| `category_gini`    | Gini coefficient of category distribution            |

_In legacy mode, only `num_categories` is used; entropy and gini are excluded._

### 6. Session Behavior Features

| Feature                      | Description                           |
| ---------------------------- | ------------------------------------- |
| `avg_reading_time`           | Average reading time per impression   |
| `avg_session_length`         | Average impressions per session       |
| `avg_categories_per_session` | Average unique categories per session |
| `avg_category_switches`      | Average category switches per session |
| `avg_session_duration`       | Average total read_time per session   |

_In new mode, these are **not** included by default. They are only used in legacy mode._

### 7. Subscriber Feature (optional)

| Feature         | Description                                 |
| --------------- | ------------------------------------------- |
| `is_subscriber` | Whether the user is a paid subscriber (1/0) |

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

- **CLI**: `python scripts/run_full_pipeline.py ... --legacy-features`
- **Config JSON**: `"clustering": { "legacy_features": true }`
- **Code**: `UserFeatureExtractor(legacy_mode=True)` or `create_user_features(..., legacy_mode=True)`
