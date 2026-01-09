# RICON analysis

## Overview

![Overview of the flow](flow.png "Overview of the flow")

## Description

This is a monorepo for the analysis for the paper "Not One News Recommender To Fit Them All: How Different News Recommender Strategies Serve Different Engagement-based User Segments"
This repo contains the code for the user clustering and the subsequent cluster analysis using recpack.

We deliberately chose to split the code into two separate parts, to make a clear separation between the clustering itself and the cluster analysis.

## Datasets

The datasets can be downloaded from:

- Ekstra dataset (EB-NeRD): https://recsys.eb.dk/
- Adressa: https://reclab.idi.ntnu.no/dataset/

## How to run

1. Start with the 1-user-clustering folder and follow the instructions in the README there.
2. Then move the resulting clusters to the 2-cluster-analysis folder and follow the instructions in the README there.

## Data used

### Required data fields

To run both analyses in this repository, you need to collect the following data fields:

#### From behaviors/interactions data:
- **`user_id`** - Unique identifier for each user (used in both analyses)
- **`article_id`** - Unique identifier for each article (used in both analyses)
- **`impression_time`** - Timestamp of when the user viewed/interacted with the article (used in both analyses)
- **`session_id`** - Unique identifier for each user session (used in clustering)
- **`read_time`** - Time spent reading the article in seconds (used in clustering)
- **`is_subscriber`** - Boolean indicating if the user is a paid subscriber (used in clustering)
- **`impression_id`** - Unique identifier for each impression/event (used in clustering)

#### From articles data:
- **`article_id`** - Unique identifier for each article (used in both analyses)
- **`category_str`** - Category of the article as a string (used in both analyses: for clustering features and content-based recommendations)
- **`title`** - Title of the article (used in both analyses: for content-based recommendations)
- **`sentiment_score`** - Sentiment score of the article (used in clustering)

### Summary

**Fields used in BOTH analyses:**
- `user_id`
- `article_id`
- `impression_time`
- `category_str`
- `title`

**Fields used ONLY in clustering:**
- `session_id`
- `read_time`
- `is_subscriber`
- `impression_id` (just to count the number of impressions for each user - can be replaced by any other method to count the number of impressions)

**Note:** The recpack evaluation uses `interactions.csv` (created from behaviors data) and `articles_content.csv` (created from articles data with `category_str` and `title` concatenated). The clustering analysis uses the full behaviors and articles datasets to compute user engagement features.