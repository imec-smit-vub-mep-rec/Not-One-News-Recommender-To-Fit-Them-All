import pandas as pd
import numpy as np
from datetime import datetime

# Load datasets
articles_df = pd.read_parquet('datasets/ekstra/articles.parquet')
behaviors_df = pd.read_parquet('datasets/ekstra/behaviors.parquet')

# Debug behaviors_df
print("\nDEBUG: Behaviors DataFrame Info:")
print("Shape:", behaviors_df.shape)
print("\nColumns:", behaviors_df.columns.tolist())
print("\nSample of article_ids_clicked:")
print(behaviors_df['article_ids_clicked'].head())
print("\nSample of behaviors data:")
print(behaviors_df.head())

# Filter articles with valid engagement metrics (published after Feb 16, 2023)
valid_articles = articles_df[
    (~articles_df['total_inviews'].isna()) &
    (~articles_df['total_pageviews'].isna()) &
    (~articles_df['total_read_time'].isna())
].copy()  # Use copy() to avoid SettingWithCopyWarning

# Create engagement categories for articles


def categorize_article_engagement(row):
    # Calculate engagement score based on normalized metrics
    pageview_ratio = row['total_pageviews'] / \
        row['total_inviews'] if row['total_inviews'] > 0 else 0
    avg_read_time = row['total_read_time'] / \
        row['total_pageviews'] if row['total_pageviews'] > 0 else 0

    # Engagement categories
    if pageview_ratio > 0.5 and avg_read_time > 120:
        return 'High_Engagement'
    elif pageview_ratio > 0.3 and avg_read_time > 60:
        return 'Medium_Engagement'
    else:
        return 'Low_Engagement'


valid_articles['engagement_category'] = valid_articles.apply(
    categorize_article_engagement, axis=1)

# Process behavior data to create event log for Disco
# 1. Homepage views
homepage_views = behaviors_df[behaviors_df['article_id'].isna()].copy()
homepage_views['activity'] = 'Homepage_View'
homepage_views['timestamp'] = homepage_views['impression_time']

# 2. Article impressions
article_impressions = []
for _, row in behaviors_df.iterrows():
    if isinstance(row['article_ids_inview'], list) and len(row['article_ids_inview']) > 0:
        for article_id in row['article_ids_inview']:
            article_impressions.append({
                'session_id': row['session_id'],
                'user_id': row['user_id'],
                'article_id': article_id,
                'activity': 'Article_Impression',
                'timestamp': row['impression_time'],
                'is_subscriber': row['is_subscriber']
            })
article_impressions_df = pd.DataFrame(article_impressions)

# 3. Article clicks
article_clicks = []
print("Processing behaviors data for clicks...")
click_count = 0
for _, row in behaviors_df.iterrows():
    # Debug first few rows
    if click_count == 0:
        print("\nDEBUG: First row article_ids_clicked type:", type(row['article_ids_clicked']))
        print("DEBUG: First row article_ids_clicked value:", row['article_ids_clicked'])
    
    # Check if article_ids_clicked exists and has values
    clicked_ids = row['article_ids_clicked']
    if isinstance(clicked_ids, np.ndarray) and len(clicked_ids) > 0:
        for article_id in clicked_ids:
            click_count += 1
            article_clicks.append({
                'session_id': row['session_id'],
                'user_id': row['user_id'],
                'article_id': article_id,
                'activity': 'Article_Click',
                'timestamp': row['impression_time'],
                'is_subscriber': row['is_subscriber'],
                'next_read_time': row['next_read_time'],
                'next_scroll_percentage': row['next_scroll_percentage']
            })

print(f"Found {click_count} article clicks")
print(f"Length of article_clicks list: {len(article_clicks)}")

if len(article_clicks) == 0:
    print("Warning: No article clicks found in the behaviors data!")
    # Create an empty DataFrame with the expected columns
    article_clicks_df = pd.DataFrame(columns=[
        'session_id', 'user_id', 'article_id', 'activity', 'timestamp',
        'is_subscriber', 'next_read_time', 'next_scroll_percentage'
    ])
else:
    article_clicks_df = pd.DataFrame(article_clicks)

print("Article clicks DataFrame shape:", article_clicks_df.shape)
print("Available columns in article_clicks_df:",
      article_clicks_df.columns.tolist())

# 4. Reading and scrolling behaviors (for clicked articles)
article_reads = article_clicks_df.copy()

# Only proceed with read/scroll processing if we have data
if len(article_reads) > 0:
    # Categorize read time
    def categorize_read_time(read_time):
        if read_time is None or np.isnan(read_time):
            return 'Read_Unknown'
        elif read_time < 30:
            return 'Read_Short'
        elif read_time < 120:
            return 'Read_Medium'
        else:
            return 'Read_Long'

    # Fix column name references - using the original column names from behaviors_df
    article_reads['read_activity'] = article_reads['next_read_time'].apply(
        categorize_read_time)
    article_reads['timestamp'] = pd.to_datetime(
        article_reads['timestamp']) + pd.to_timedelta(1, unit='second')

    # Categorize scroll percentage
    def categorize_scroll(scroll_pct):
        if scroll_pct is None or np.isnan(scroll_pct):
            return 'Scroll_Unknown'
        elif scroll_pct < 33:
            return 'Scroll_Low'
        elif scroll_pct < 75:
            return 'Scroll_Medium'
        else:
            return 'Scroll_High'

    article_reads['scroll_activity'] = article_reads['next_scroll_percentage'].apply(
        categorize_scroll)
    article_scrolls = article_reads.copy()
    article_scrolls['activity'] = article_scrolls['scroll_activity']
    article_scrolls['timestamp'] = pd.to_datetime(
        article_scrolls['timestamp']) + pd.to_timedelta(2, unit='second')
    article_reads['activity'] = article_reads['read_activity']
else:
    print("No article clicks to process for reading/scrolling behaviors")
    article_scrolls = article_reads.copy()

# Combine all events
events = []
for df, activity_col in [
    (homepage_views, 'activity'),
    (article_impressions_df, 'activity'),
    (article_clicks_df, 'activity'),
    (article_reads, 'activity'),
    (article_scrolls, 'activity')
]:
    # Make sure all required columns exist
    if 'session_id' in df.columns and 'user_id' in df.columns and activity_col in df.columns and 'timestamp' in df.columns:
        temp_df = df[['session_id', 'user_id',
                      activity_col, 'timestamp']].copy()
        temp_df.rename(columns={activity_col: 'activity'}, inplace=True)
        events.append(temp_df)
    else:
        missing_cols = [col for col in ['session_id', 'user_id',
                                        activity_col, 'timestamp'] if col not in df.columns]
        print(f"Skipping DataFrame because missing columns: {missing_cols}")

event_log = pd.concat(events, ignore_index=True)

# Sort by session_id and timestamp
event_log = event_log.sort_values(['session_id', 'timestamp'])

# Add article metadata where available - ensure article_id is in event_log
if 'article_id' in event_log.columns:
    # Use merge with indicator to see which rows had matches
    event_log = event_log.merge(
        articles_df[['article_id', 'category_str',
                     'premium', 'sentiment_label']],
        on='article_id',
        how='left',
        indicator=True
    )
    # Check how many matches occurred
    match_counts = event_log['_merge'].value_counts()
    print(f"Merge results: {match_counts}")
    # Remove the indicator column
    event_log = event_log.drop(columns=['_merge'])
else:
    print("Warning: 'article_id' not in event_log. Cannot merge article metadata.")

# Prepare CSV for Disco
event_log.to_csv('article_engagement_process.csv', index=False)

# Create a summary view with conversion metrics
session_summary = event_log.groupby('session_id').agg({
    'activity': lambda x: list(x),
    'user_id': 'first'
})

# Calculate conversion rates
session_summary['homepage_to_impression'] = session_summary['activity'].apply(
    lambda x: 'Homepage_View' in x and any('Article_Impression' in act for act in x)
)
session_summary['impression_to_click'] = session_summary['activity'].apply(
    lambda x: any('Article_Impression' in act for act in x) and any('Article_Click' in act for act in x)
)
session_summary['click_to_read'] = session_summary['activity'].apply(
    lambda x: any('Article_Click' in act for act in x) and any(
        any(read_type in act for read_type in ['Read_Short', 'Read_Medium', 'Read_Long']) for act in x
    )
)

print(f"Homepage to Impression conversion: {session_summary['homepage_to_impression'].mean():.2%}")
print(f"Impression to Click conversion: {session_summary['impression_to_click'].mean():.2%}")
print(f"Click to Read conversion: {session_summary['click_to_read'].mean():.2%}")
