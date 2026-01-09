"""
Script that loads datasets/adressa-large-0416/behaviors.parquet
and removes all rows where article_id is empty
then it sets for all the rows where article_id is "homepage", the article_id to an empty value
it writes that result as datasets/adressa-large-0416/behaviors_with_homepage.parquet
and then it removes all these homepage rows from the dataset
it writes that result as datasets/adressa-large-0416/behaviors_no_homepage.parquet
Give a summary of the statistics of the dataset before and after each step
"""

import pandas as pd

# Load the dataset
print("Loading dataset...")
df = pd.read_parquet('datasets/adressa-large-0416/behaviors.parquet')
print(f"Original dataset: {len(df)} rows")

# Report initial statistics
print("\nInitial Statistics:")
print(f"Total rows: {len(df)}")
print(f"Empty article_id: {df[df['article_id'].isna()].shape[0]}")
print(f"Homepage article_id: {df[df['article_id'] == 'homepage'].shape[0]}")

# Step 1: Remove rows where article_id is empty
df_no_empty = df[df['article_id'].notna() &
                 (df['article_id'] != 'empty')]
print(f"Rows removed: {len(df) - len(df_no_empty)}")
print(f"Remaining rows: {len(df_no_empty)}")

# Step 2: Set 'homepage' article_id to empty
df_with_homepage = df_no_empty.copy()
homepage_mask = df_with_homepage['article_id'] == 'homepage'
homepage_count = homepage_mask.sum()
df_with_homepage.loc[homepage_mask, 'article_id'] = None
print("\nAfter setting 'homepage' to empty:")
print(f"Homepage rows modified: {homepage_count}")
print(f"Total rows: {len(df_with_homepage)}")

# Save the result with homepage as empty
df_with_homepage.to_parquet(
    'datasets/adressa-large-0416/behaviors_with_homepage.parquet')
print("Saved as behaviors_with_homepage.parquet")

# Step 3: Remove the homepage (now empty) rows
df_no_homepage = df_with_homepage.dropna(subset=['article_id'])
print("\nAfter removing homepage rows:")
print(f"Rows removed: {len(df_with_homepage) - len(df_no_homepage)}")
print(f"Final row count: {len(df_no_homepage)}")

# Save the result without homepage
df_no_homepage.to_parquet(
    'datasets/adressa-large-0416/behaviors_no_homepage.parquet')
print("Saved as behaviors_no_homepage.parquet")

print("\nComplete! Summary:")
print(f"Original: {len(df)} rows")
print(f"After removing empty: {len(df_no_empty)} rows")
print(f"After removing homepage: {len(df_no_homepage)} rows")
