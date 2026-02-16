# Topic-Level Coverage and Gini: Implementation Plan

## Overview

Add topic-level diversity metrics (Coverage and Gini) to the RecPack evaluation pipeline, with a separate topic report file listing most and least popular topics.

**Specifications:**
- **Topic source:** `categories` column (array of strings)
- **Multi-topic items:** Split exposure across all categories
- **Unknown items:** Exclude from topic metrics
- **Output:** Separate topic report file (per cluster, per algorithm)

---

## 1. Data Requirements

### 1.1 Categories Column

- **Schema:** `categories` is `object` (list of strings) per `ARTICLES_SCHEMA`
- **Source:** AD converter preserves raw `categories` from `article_metadata.csv` if present; HLN/VK may have different structures
- **Fallback:** When `categories` is missing or empty, use `[category_str]` as a single-element list (so items with only `category_str` still contribute to topic metrics)

### 1.2 Item Mapping

- RecPack uses internal 0-based item indices
- `preprocessing_info['item_mapping']`: `article_id` → `internal_id`
- Need reverse: `internal_id` → `article_id` to look up categories
- Articles with no valid categories (empty, null, or all empty strings) → **exclude from topic metrics**

---

## 2. Core Logic

### 2.1 Build Item → Categories Mapping

```
Input: articles_df with columns [article_id, categories, category_str]
Output: internal_id → list of category strings (valid, non-empty)

For each article:
  - If categories exists and is non-empty list of non-empty strings → use it
  - Else if category_str is non-empty → use [category_str]
  - Else → mark as unknown (exclude from topic metrics)

Build: internal_id_to_categories: Dict[int, List[str]]
  - Only include items that appear in item_mapping (RecPack's filtered item set)
  - Unknown items are simply absent from this dict
```

### 2.2 Aggregate Exposure by Topic (Split for Multi-Topic Items)

For each item `i` with exposure `e_i` and categories `[c1, c2, ..., cn]`:

```
topic_exposure[c] += e_i / n   for each c in [c1, c2, ..., cn]
```

So a multi-topic item splits its exposure equally across its categories.

### 2.3 Topic Coverage

```
n_topics_in_catalog = number of unique categories across all items in the evaluation catalog (that have valid categories)
n_topics_recommended = number of topics with topic_exposure > 0

Coverage_topics = n_topics_recommended / n_topics_in_catalog   (0 if n_topics_in_catalog == 0)
```

### 2.4 Topic Gini

Use the same `_gini_from_exposure` formula on the vector of topic exposures (one value per topic in the catalog). Topics with zero exposure are included (exposure = 0).

### 2.5 Topic Popularity Report

For each (cluster_id, algorithm, k):

- Sort topics by exposure (descending)
- Columns: `topic`, `exposure`, `rank`, `exposure_pct` (exposure / total_exposure * 100)
- Include both most popular (top N) and least popular (bottom N) in the report
- Optional: flag topics with zero exposure

---

## 3. File Changes

### 3.1 New Module: `src/evaluation/topic_diversity.py`

| Function | Purpose |
|----------|---------|
| `_normalize_categories(articles_df, item_mapping)` | Build `internal_id → List[str]` for items with valid categories; return also `all_topics` (set of catalog topics) |
| `_aggregate_exposure_by_topic(exposure, internal_id_to_categories)` | Aggregate item exposure to topic exposure (split for multi-topic) |
| `_compute_topic_coverage(topic_exposure, all_topics)` | Coverage = topics with exposure > 0 / total topics |
| `_compute_topic_gini(topic_exposure)` | Gini on topic exposure vector |
| `compute_topic_diversity(exposure, internal_id_to_categories, all_topics)` | Returns `{CoverageK_topics_{k}, GiniK_topics_{k}}` |
| `build_topic_report(topic_exposure, top_n=20, bottom_n=10)` | DataFrame with topic, exposure, rank, exposure_pct |

### 3.2 Changes to `src/evaluation/recpack_pipeline.py`

| Location | Change |
|----------|--------|
| `run_evaluation()` signature | Add `articles_df: Optional[pd.DataFrame] = None` |
| `run_evaluation()` body | After computing item-level exposure (lines ~583–588): if `articles_df` provided, build `internal_id_to_categories`, compute topic metrics, add `CoverageK_topics_{k}` and `GiniK_topics_{k}` to each row; build topic report per (algo, k) |
| `RecPackPipeline.__init__` | Add `articles_df: Optional[pd.DataFrame] = None` |
| `RecPackPipeline.load_data_from_dataframes()` | Accept optional `articles_df` |
| `RecPackPipeline.run()` | Pass `articles_df` to `run_evaluation` |
| `_evaluate_single_cluster()` signature | Add `articles_df: Optional[pd.DataFrame] = None` |
| `_evaluate_single_cluster()` body | Pass `articles_df` to pipeline; collect topic reports from pipeline |
| `run_cluster_evaluation()` signature | Add `articles_df: Optional[pd.DataFrame] = None` |
| `run_cluster_evaluation()` body | Pass `articles_df` to `_evaluate_single_cluster`; after all clusters, write topic report file(s) |

### 3.3 Changes to `scripts/run_full_pipeline.py`

| Location | Change |
|----------|--------|
| `run_evaluation()` | Load `articles_cleaned.parquet` before calling `run_cluster_evaluation`; pass as `articles_df`; do not delete `articles_df` until after evaluation (or load it inside `run_evaluation` from session path) |
| Data flow | Ensure `articles_cleaned.parquet` is available at evaluation time (it is saved in `run_preprocessing`) |

---

## 4. Topic Report File Format

### 4.1 Output Path

```
{session}/evaluation_results/topic_report_cluster_{cluster_id}.csv
```

Or, if per-algorithm reports are desired:

```
{session}/evaluation_results/topic_report_cluster_{cluster_id}_{algorithm}_k{k}.csv
```

**Recommendation:** Single file per cluster with columns `cluster_id`, `algorithm`, `k`, `topic`, `exposure`, `rank`, `exposure_pct`, `is_most_popular`, `is_least_popular` (or similar). This allows filtering/analysis in one place.

### 4.2 Report Structure (Single File Per Cluster)

| Column | Description |
|--------|-------------|
| cluster_id | Cluster identifier |
| algorithm | Algorithm name |
| k | K value (10, 20, 50) |
| topic | Category/topic string |
| exposure | Aggregated exposure (fractional for multi-topic items) |
| rank | 1 = most popular, N = least popular |
| exposure_pct | exposure / total_exposure * 100 |
| n_items_in_topic | (Optional) Number of items in catalog with this topic |

---

## 5. Edge Cases

| Case | Handling |
|------|----------|
| `categories` column missing | Fall back to `[category_str]` |
| `categories` is empty list `[]` | Fall back to `[category_str]` if non-empty; else exclude |
| `categories` contains empty strings | Filter out empty strings; if none left, use category_str or exclude |
| `categories` is string (e.g. JSON) | Parse; if parse fails, use category_str or exclude |
| No articles_df provided | Skip topic metrics; no topic report |
| All items unknown | Coverage_topics=0, Gini=0; report empty or with zero-exposure topics |
| Item in predictions but not in articles | Exclude from topic aggregation (unknown item) |

---

## 6. Implementation Order

1. **Create `topic_diversity.py`** with helper functions and tests for:
   - `_normalize_categories` (including fallback logic)
   - `_aggregate_exposure_by_topic` (split for multi-topic)
   - `_compute_topic_coverage`, `_compute_topic_gini`
   - `build_topic_report`

2. **Integrate into `run_evaluation`**:
   - Add `articles_df` parameter
   - Call topic helpers after item exposure is computed
   - Add `CoverageK_topics_{k}` and `GiniK_topics_{k}` to result rows

3. **Integrate into `RecPackPipeline`**:
   - Add `articles_df` to init and `load_data_from_dataframes`
   - Return topic report data from `run()` (or store on pipeline for retrieval)

4. **Integrate into `_evaluate_single_cluster` and `run_cluster_evaluation`**:
   - Thread `articles_df` through
   - Collect topic reports per cluster
   - Write `topic_report_cluster_{id}.csv` to `output_dir`

5. **Update `run_full_pipeline.py`**:
   - Load `articles_cleaned.parquet` before evaluation
   - Pass to `run_cluster_evaluation`

6. **Update README / docs** with new metrics and report format.

---

## 7. Testing Checklist

- [ ] Single-topic items: exposure fully assigned to one topic
- [ ] Multi-topic items: exposure split equally across categories
- [ ] Unknown items: excluded from topic metrics
- [ ] Empty categories: fallback to category_str
- [ ] No articles_df: topic metrics skipped, no errors
- [ ] Topic report: correct ranking, exposure_pct sums to 100
- [ ] Coverage_topics in [0, 1]
- [ ] Gini_topics in [0, 1]

---

## 8. Optional Enhancements (Future)

- Configurable topic column (e.g. `topics` vs `categories`)
- Topic report as Excel with multiple sheets (one per algorithm)
- Summary table: CoverageK_topics and GiniK_topics in main results CSV for quick comparison
