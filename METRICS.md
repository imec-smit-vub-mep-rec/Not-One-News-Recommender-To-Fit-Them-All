# Metrics Reference

This document defines all column names and metrics produced by the pipeline, with their definitions and calculation methods.

---

## 1. Cluster Reports (cluster_profiles.xlsx)

The cluster profiles Excel file (`runs/<dataset>_<timestamp>/clusters/cluster_profiles.xlsx`) contains multiple sheets. Below are the column definitions for each sheet.

### 1.1 Cluster Summary Sheet

One row per cluster. Computed from raw impressions via `_compute_cluster_summary()`.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier (index) | Integer label from K-Means clustering |
| **Number of Users** | Count of users in the cluster | `users_unique.groupby('cluster_id')['user_id'].nunique()` |
| **Percentage of Users (%)** | Share of total users in this cluster | `(cluster_size / total_users) × 100` |
| **Number of Subscribers** *(or Number of Logged In Users)* | Count of subscribers/logged-in users in the cluster | Sum of `is_subscriber` (or logged-in flag) per user, aggregated by cluster. Dataset-specific label. |
| **Avg Reading Time (s)** | Mean read time per impression, averaged over users in cluster | Per user: `mean(read_time)` over all impressions. Per cluster: mean of per-user values. |
| **Proportion of Time on Articles** | Share of total reading time spent on articles (vs homepage) | Per user: `sum(read_time on articles) / sum(read_time)`. Per cluster: mean of per-user values. |
| **Avg Reading Time Homepage (s)** | Mean read time per homepage impression | Per user: `mean(read_time)` where `article_id` is null or 'homepage'. Per cluster: mean of per-user values. |
| **Avg Reading Time Articles (s)** | Mean read time per article impression | Per user: `mean(read_time)` where `article_id` is not null. Per cluster: mean of per-user values. |
| **Avg Impressions per Session** | Mean number of impressions per session | Per user: `mean(impressions per session_id)`. Per cluster: mean of per-user values. |
| **Avg Sessions per User** | Mean number of sessions per user | Per user: `nunique(session_id)`. Per cluster: mean of per-user values. |
| **Avg Sessions per Logged-in User** | Mean sessions for logged-in users only | Same as above, filtered to users with `is_subscriber == True` (or logged-in). |
| **Avg Sessions per Non-logged-in User** | Mean sessions for anonymous users only | Same as above, filtered to users with `is_subscriber == False`. |
| **Avg Categories Read** | Mean number of distinct categories per user | Per user: `nunique(category_str)` over impressions. Per cluster: mean of per-user values. |
| **Avg Session Duration (s)** | Mean total read time per session | Per user: `mean(sum(read_time) per session)`. Per cluster: mean of per-user values. |
| **Avg Category Switches per Session** | Mean number of category changes within a session | Per session: count transitions where `category_str != shift(category_str)`. Per user: mean over sessions. Per cluster: mean of per-user values. |
| **Morning (%)** | Share of impressions during 6:00–11:59 | Per user: proportion of impressions with `6 <= hour < 12`. Per cluster: mean × 100. |
| **Afternoon (%)** | Share of impressions during 12:00–17:59 | Per user: proportion with `12 <= hour < 18`. Per cluster: mean × 100. |
| **Evening (%)** | Share of impressions during 18:00–23:59 | Per user: proportion with `18 <= hour < 24`. Per cluster: mean × 100. |
| **Night (%)** | Share of impressions during 0:00–5:59 | Per user: proportion with `0 <= hour < 6`. Per cluster: mean × 100. |
| **Weekend (%)** | Share of impressions on Saturday/Sunday | Per user: proportion with `dayofweek >= 5`. Per cluster: mean × 100. |
| **Avg Engagement Span (days)** | Mean time span between first and last impression | Per user: `(max(impression_time) - min(impression_time))` in days. Per cluster: mean of per-user values. |
| **Avg Homepage Impressions** | Mean number of homepage impressions per user | Per user: count of impressions where `article_id` is null or 'homepage'. Per cluster: mean of per-user values. |
| **Avg Article Impressions** | Mean number of article impressions per user | Per user: count of impressions where `article_id` is not null. Per cluster: mean of per-user values. |
| **Homepage Ratio** | Proportion of impressions that are homepage views | Per user: `homepage_impressions / (homepage + article_impressions)`. Per cluster: mean of per-user values. |
| **Avg Category Entropy** | Mean entropy of category distribution per user | Per user: `-Σ p(c) log(p(c))` over categories. Per cluster: mean of per-user values. |
| **Avg Category Gini** | Mean Gini coefficient of category distribution per user | Per user: Gini on category impression counts. Per cluster: mean of per-user values. |
| **Desktop (%)** *(optional)* | Share of impressions from desktop devices | Per user: proportion of impressions with `device_type == 'desktop'`. Per cluster: mean × 100. Present if `device_type` in data. |
| **Mobile (%)** *(optional)* | Share of impressions from mobile devices | Same as above for `device_type == 'mobile'`. |
| **Tablet (%)** *(optional)* | Share of impressions from tablet devices | Same as above for `device_type == 'tablet'`. |
| **Avg Scroll Depth** *(optional)* | Mean scroll depth per user | Per user: `mean(scroll_depth)` over impressions. Per cluster: mean of per-user values. Present if `scroll_depth` in data. |

### 1.2 Cluster Statistics Sheet

Mean and standard deviation of each clustering feature per cluster. From `get_cluster_statistics()`.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier | Integer label |
| **size** | Number of users in cluster | `len(cluster_data)` |
| **{feature}_mean** | Mean of feature in cluster | `cluster_data[feature].mean()` |
| **{feature}_std** | Standard deviation of feature in cluster | `cluster_data[feature].std()` |

Features are those used for clustering (e.g. from `UserFeatureExtractor`).

### 1.3 Cluster Centers (Scaled) Sheet

Scaled centroid values from K-Means. Used for interpretation after inverse transform.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier | Integer label |
| **size** | Number of users in cluster | Count from `np.unique(labels, return_counts=True)` |
| **size_pct** | Percentage of users in cluster | `(size / total_users) × 100` |
| **{feature}** | Scaled centroid value | Value from `model.cluster_centers_` for each feature |

### 1.4 Category Profiles Sheet

Top-N categories per cluster by impression share. From `_compute_category_profiles()`.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier | From user–cluster mapping |
| **category** | Category name | From `category_str` (or merged from articles) |
| **impressions** | Number of impressions in this cluster–category | `groupby(['cluster_id','category_str']).size()` |
| **proportion_pct** | Share of cluster impressions in this category | `(impressions / cluster_total) × 100` |
| **rank** | Rank by impressions (1 = most) | `rank(ascending=False, method='min')` within cluster |

### 1.5 Cluster Distributions Sheet

Percentile breakdowns of key per-user metrics within each cluster. From `_compute_cluster_distributions()`.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier | From user metrics |
| **metric** | Metric name | One of: `avg_reading_time`, `avg_sessions_per_user`, `avg_impressions_per_session`, `avg_session_duration`, `avg_categories_read`, `avg_category_switches`, `engagement_span_days`, `category_entropy`, `category_gini`, `avg_homepage_impressions`, `avg_article_impressions`, `homepage_ratio` |
| **mean** | Mean of metric in cluster | `vals.mean()` |
| **std** | Standard deviation | `vals.std()` |
| **min** | Minimum | `vals.min()` |
| **p25** | 25th percentile | `vals.quantile(0.25)` |
| **median** | 50th percentile | `vals.quantile(0.50)` |
| **p75** | 75th percentile | `vals.quantile(0.75)` |
| **max** | Maximum | `vals.max()` |

### 1.6 Recommendation Performance Sheet

Same structure as the evaluation results CSV (see Section 2). Appended after evaluation.

---

## 2. Evaluation Results (cluster_X_results.csv)

Files in `runs/<dataset>_<timestamp>/evaluation_results/cluster_{id}_results.csv`. One row per algorithm. Each file corresponds to one cluster (cluster ID from filename). Produced by `run_evaluation()` in the RecPack pipeline using the LastItemPrediction scenario.

| Column | Definition | Calculation |
|--------|-------------|-------------|
| **cluster_id** | Cluster identifier | Present only in the Recommendation Performance sheet (concatenated results). In raw `cluster_X_results.csv`, the cluster is identified by the filename (`cluster_0_results.csv` → cluster 0). |
| **algorithm** | Recommendation algorithm name | e.g. `Popularity`, `ItemKNN`, `EASE`, `MultVAE`, `CB-ST` |
| **NDCGK_{k}** | Normalized Discounted Cumulative Gain at K | DCG@K / IDCG@K. DCG = Σ rel(i) / log₂(i+1); IDCG = ideal DCG. Relevance = 1 if held-out item in top-K, else 0. Averaged over test users. |
| **RecallK_{k}** | Recall at K | (Relevant items in top-K) / (Total relevant items). For LastItemPrediction: 1 if held-out item in top-K else 0. Averaged over test users. |
| **PrecisionK_{k}** | Precision at K | (Relevant items in top-K) / K. For LastItemPrediction: 1/K if held-out in top-K else 0. Averaged over test users. |
| **CoverageK_{k}** | Item catalog coverage at K | Proportion of items that appear in at least one user's top-K. `count_nonzero(exposure) / n_items`. |
| **GiniK_{k}** | Gini coefficient of item exposure at K | Inequality of exposure across items. `(2 Σ i·x_i) / (n·total) - (n+1)/n` on sorted exposure vector. 0 = equal, 1 = maximally unequal. |
| **CoverageK_topics_{k}** *(optional)* | Topic-level coverage at K | Proportion of topics (categories) with exposure > 0. Present when `articles_df` with categories is provided. |
| **GiniK_topics_{k}** *(optional)* | Gini coefficient of topic exposure at K | Same Gini formula applied to topic exposure vector. Present when `articles_df` with categories is provided. |

**Note:** `k` is from `k_values` (e.g. 10, 20, 50). Topic metrics are 0 when no article categories are available.

### Evaluation Scenario

- **LastItemPrediction**: Each user's last interaction is held out for testing; earlier interactions are used for training. One relevant item per user.
