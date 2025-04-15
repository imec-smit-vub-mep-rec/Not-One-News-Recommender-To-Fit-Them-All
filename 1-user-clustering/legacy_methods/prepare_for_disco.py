import pandas as pd
import numpy as np
from datetime import datetime

# Read the data
articles_df = pd.read_parquet('datasets/ekstra/articles.parquet')
history_df = pd.read_parquet('datasets/ekstra/history.parquet')

def expand_history_rows(row):
    """Expand the nested lists in history into individual rows"""
    events = []
    
    # Ensure all lists have the same length
    n_events = len(row['article_id_fixed'])
    
    for i in range(n_events):
        event = {
            'case_id': row['user_id'],
            'timestamp': row['impression_time_fixed'][i],
            'article_id': row['article_id_fixed'][i],
            'read_time': row['read_time_fixed'][i],
            'scroll_percentage': row['scroll_percentage_fixed'][i] if i < len(row['scroll_percentage_fixed']) else None
        }
        events.append(event)
    
    return events

# Expand the nested history data
expanded_events = []
for _, row in history_df.iterrows():
    expanded_events.extend(expand_history_rows(row))

# Convert to DataFrame
events_df = pd.DataFrame(expanded_events)

# Merge with article information
articles_slim = articles_df[['article_id', 'category_str', 'article_type', 'premium']]
events_df = events_df.merge(articles_slim, on='article_id', how='left')

# Create activity string combining category and premium status
events_df['activity'] = events_df['category_str'] + ' (' + events_df['premium'].map({True: 'Premium', False: 'Free'}) + ')'

# Prepare final Disco format
disco_df = events_df[[
    'case_id',
    'activity',
    'timestamp',
    'article_id',
    'read_time',
    'scroll_percentage',
    'article_type'
]].copy()

# Rename columns to Disco's preferred format
disco_df.columns = [
    'Case ID',
    'Activity',
    'Timestamp',
    'Article ID',
    'Read Time (seconds)',
    'Scroll Percentage',
    'Article Type'
]

# Sort by Case ID and Timestamp
disco_df = disco_df.sort_values(['Case ID', 'Timestamp'])

# Export to CSV
disco_df.to_csv('user_journeys_for_disco.csv', index=False)

print("Process mining data has been prepared and saved to 'user_journeys_for_disco.csv'")
print("\nDataset summary:")
print(f"Number of cases (users): {disco_df['Case ID'].nunique()}")
print(f"Number of events: {len(disco_df)}")
print(f"Number of unique activities: {disco_df['Activity'].nunique()}")
print("\nSample activities:")
print(disco_df['Activity'].value_counts().head()) 