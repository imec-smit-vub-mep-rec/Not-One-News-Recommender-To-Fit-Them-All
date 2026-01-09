"""
Example of a line in the adressa dataset:
 {
    "profile": [
      {
        "item": "0",
        "groups": [{ "count": 1, "group": "adressa-importance", "weight": 1.0 }]
      },
      {
        "item": "adressa.no",
        "groups": [{ "count": 1, "group": "site", "weight": 1.0 }]
      },
      {
        "item": "article",
        "groups": [{ "count": 1, "group": "pageclass", "weight": 1.0 }]
      },
      {
        "item": "bergsbakken",
        "groups": [{ "count": 1, "group": "entity", "weight": 0.88671875 }]
      },
      {
        "item": "elgeseter",
        "groups": [{ "count": 1, "group": "entity", "weight": 0.7265625 }]
      },
      {
        "item": "festningen",
        "groups": [{ "count": 1, "group": "entity", "weight": 0.8125 }]
      },
      {
        "item": "free",
        "groups": [{ "count": 1, "group": "adressa-access", "weight": 1.0 }]
      },
      {
        "item": "fyrverkeri",
        "groups": [{ "count": 1, "group": "concept", "weight": 1.0 }]
      },
      {
        "item": "gl\u00f8shaugen",
        "groups": [{ "count": 1, "group": "entity", "weight": 0.69921875 }]
      },
      {
        "item": "joakim slettebak wangen",
        "groups": [{ "count": 1, "group": "author", "weight": 1.0 }]
      },
      {
        "item": "marinen",
        "groups": [{ "count": 1, "group": "entity", "weight": 0.72265625 }]
      },
      {
        "item": "merethe mauland",
        "groups": [{ "count": 1, "group": "entity", "weight": 1.0 }]
      },
      {
        "item": "no",
        "groups": [{ "count": 1, "group": "language", "weight": 1.0 }]
      },
      {
        "item": "norunn bergesen",
        "groups": [{ "count": 1, "group": "author", "weight": 1.0 }]
      },
      {
        "item": "nyheter",
        "groups": [
          { "count": 1, "group": "category", "weight": 0.0859375 },
          { "count": 1, "group": "taxonomy", "weight": 0.08984375 }
        ]
      },
      {
        "item": "nyheter/trondheim",
        "groups": [{ "count": 1, "group": "taxonomy", "weight": 0.51953125 }]
      },
      {
        "item": "odd arne nilsen",
        "groups": [{ "count": 1, "group": "person", "weight": 0.90625 }]
      },
      {
        "item": "sports",
        "groups": [
          { "count": 1, "group": "classification", "weight": 0.43359375 }
        ]
      },
      {
        "item": "trondheim",
        "groups": [
          { "count": 1, "group": "category", "weight": 0.18359375 },
          { "count": 1, "group": "location", "weight": 0.16015625 }
        ]
      },
      {
        "item": "trondheim bydrift",
        "groups": [{ "count": 1, "group": "concept", "weight": 0.75 }]
      }
    ],
    "category1": "nyheter|trondheim",
    "canonicalUrl": "http://www.adressa.no/nyheter/trondheim/2016/12/31/Det-blir-fyrverkeri-14000281.ece",
    "userId": "cx:kfubh0ub7g8z3g5mgndoaljqd:1w4rvohza6x7d", <-- important: user_id
    "sessionStop": true,
    "referrerHostClass": "direct",
    "publishtime": "2016-12-31T15:48:48.000Z",
    "sessionStart": true,
    "keywords": "utenriks,innenriks,trondheim,E6,midtbyen,bybrann,bilulykker",
    "id": "2607fc7d7b4c0ede839a5ff6d499fa428237443e", <-- important: article_id
    "eventId": 906331936,
    "city": "trondheim",
    "title": "- Det blir fyrverkeri",
    "url": "http://adressa.no/nyheter/trondheim/2016/12/31/det-blir-fyrverkeri-14000281.ece",
    "country": "no",
    "region": "sor-trondelag",
    "author": ["norunn bergesen", "joakim slettebak wangen"],
    "deviceType": "Mobile",
    "time": 1483225203, <-- important: impression_time
    "os": "iPhone OS"
  }
"""

# 1. Load the adressa dataset (1 jsonl file per day, 7 files)
# 2. Turn them into one continuous jsonl file
# 3. Group by userId
# -> Initialize a new dataframe 'users' to store the users, with the following columns:
# - user_id = userId
# - is_subscriber = false by default
# - count_sessions = 0
# - count_total_impressions = 0
# - count_homepage_impressions = 0
# 4. Sort by time (UNIX timestamp in seconds)
# 5. Create a new dataframe 'behaviors' to store the impressions, and a dataframe 'articles' to store the article information
# 6. For each row, sorted by userId and sorted by time: process every line chronologically and insert a new row in impressions dataframe:
# - impression_id = eventId
# - session_id: random UUID, refreshes every time an event (a jsonl line) has a sessionStart = true, or if there is no sessionStart, but the previous event had a sessionStop = true
# - article_id: keep track of all articles that have been seen in the articles, based on the url. For each new article (url is not yet in the article dataframe), add it to the article dataframe and create a new article_id (unique UUID) also add a property publish_time = publishtime. Leave empty if it is homepage view (url === 'https://www.adressa.no/')
# - category_str = extract from url with regex pattern = r'https?://[^/]+/([^/]+)' -> store in the article dataframe, or retrieve from the article dataframe if it already exists
# - sentiment_score = default init .5 -> store in the article dataframe, or retrieve from the article dataframe if it already exists
# - impression_time = time
# - user_id = userId


# Update the user dataframe with the new data
# - count_sessions += 1
# - count_total_impressions += 1
# - count_homepage_impressions += 1 if url is a homepage
# - count_impressions_with_article += 1 if article_id is not None
# - count_impressions_without_article += 1 if article_id is None
# - is_subscriber = true if the url contains "/pluss/"

# 7. Save the users and articles dataframes to parquet

# 8. Now we need to update the behaviors with one extra column: is_subscriber
# -> For each row in the behaviors dataframe, fetch the is_subscriber status of the corresponding user in the users dataframe
# -> Add the is_subscriber column to each row in the behaviors dataframe

# 9. Save the behaviors dataframe to parquet

import json
import os
import re
import uuid
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import glob
import gc


def extract_category_from_url(url: str) -> str:
    """Extract category from URL using regex pattern."""
    if url == 'https://www.adressa.no/':
        return ''
    match = re.search(r'https?://[^/]+/([^/]+)', url)
    return match.group(1) if match else ''


def process_adressa_data(input_dir: str, output_dir: str):
    """
    Process adressa dataset and convert it to the required format.

    Args:
        input_dir: Directory containing adressa JSONL files
        output_dir: Directory to save output parquet files
    """
    # Initialize dataframes with optimized dtypes
    users = pd.DataFrame(columns=[
        'user_id',
        'is_subscriber',
        'count_sessions',
        'count_total_impressions',
        'count_homepage_impressions',
        'count_impressions_with_article',
        'percentage_morning',
        'percentage_afternoon',
        'percentage_evening',
        'percentage_night'
    ]).astype({
        'user_id': 'string',
        'is_subscriber': 'boolean',
        'count_sessions': 'int32',
        'count_total_impressions': 'int32',
        'count_homepage_impressions': 'int32',
        'count_impressions_with_article': 'int32',
        'percentage_morning': 'float32',
        'percentage_afternoon': 'float32',
        'percentage_evening': 'float32',
        'percentage_night': 'float32'
    })

    articles = pd.DataFrame(columns=[
        'article_id',
        'url',
        'publish_time',
        'category_str',
        'sentiment_score',
        'views',
        'total_reading_time',
    ]).astype({
        'article_id': 'string',
        'url': 'string',
        'publish_time': 'datetime64[ns]',
        'category_str': 'category',
        'sentiment_score': 'float32',
        'views': 'int32',
        'total_reading_time': 'float32'
    })

    behaviors = pd.DataFrame(columns=[
        'session_id',
        'impression_id',
        'article_id',
        'user_id',
        'impression_time',
        'read_time'
    ]).astype({
        'session_id': 'string',
        'impression_id': 'int64',
        'article_id': 'string',
        'user_id': 'string',
        'impression_time': 'int64',
        'read_time': 'float32'
    })

    # Track existing articles and user statistics
    existing_articles = {}
    user_stats = {}
    # Track last event time and session ID for each user
    user_sessions = {}

    # Process all JSONL files in chunks
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    chunk_size = 10000  # Adjust based on available memory
    total_files = len(jsonl_files)

    print(f"\nProcessing {total_files} files...")

    for file_idx, file_path in enumerate(jsonl_files, 1):
        print(
            f"\nProcessing file {file_idx}/{total_files}: {os.path.basename(file_path)}")

        # Get total lines in file for progress tracking
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        total_chunks = (total_lines + chunk_size - 1) // chunk_size

        for chunk_idx, chunk in enumerate(pd.read_json(file_path, lines=True, chunksize=chunk_size), 1):
            # Process each chunk
            chunk['datetime'] = pd.to_datetime(chunk['time'], unit='s')

            # Sort by user_id and time to ensure correct session tracking
            chunk = chunk.sort_values(['userId', 'time'])

            # Initialize session tracking
            chunk['session_id'] = None

            # Process each user's events in order
            for user_id, user_chunk in chunk.groupby('userId'):
                # Get or initialize user's session info
                last_time, current_session = user_sessions.get(
                    user_id, (None, None))

                # Process each event for this user
                for idx, row in user_chunk.iterrows():
                    # Check if we need a new session
                    if (row['sessionStart'] or
                        last_time is None or
                            (row['time'] - last_time) > 1800):  # 30 minutes timeout
                        current_session = str(uuid.uuid4())

                    # Update session ID for this event
                    chunk.at[idx, 'session_id'] = current_session

                    # Update last event time
                    last_time = row['time']

                # Update user's session info
                user_sessions[user_id] = (last_time, current_session)

            # Extract article information
            chunk['is_homepage'] = chunk['url'].isin(
                ['http://adressa.no', 'http://adressa.no/', 'https://www.adressa.no/', 'https://www.adressa.no'])
            chunk['category_str'] = chunk['url'].apply(
                extract_category_from_url)

            # Process non-homepage articles
            non_homepage = chunk[~chunk['is_homepage']]

            # Update article statistics
            article_stats = non_homepage.groupby('url').agg({
                'id': 'first',  # Get the article ID
                'publishtime': 'first',
                'category_str': 'first',
                'title': 'first',
                'activeTime': ['count', 'sum']
            }).reset_index()

            article_stats.columns = ['url', 'id', 'publish_time',
                                     'category_str', 'title', 'views', 'total_reading_time']

            # Create or update articles
            for _, row in article_stats.iterrows():
                article_id = row['id'] if 'id' in row and pd.notna(
                    row['id']) else "empty"

                if article_id not in existing_articles:
                    # Create new article
                    existing_articles[article_id] = {
                        'article_id': article_id,
                        'url': row['url'],
                        'publish_time': row['publish_time'],
                        'category_str': row['category_str'],
                        'title': row['title'],
                        'sentiment_score': 0.5,
                        'views': row['views'],
                        'total_reading_time': row['total_reading_time']
                    }
                else:
                    # Update existing article
                    existing_articles[article_id]['views'] += row['views']
                    existing_articles[article_id]['total_reading_time'] += row['total_reading_time']

            # Convert existing_articles to DataFrame
            articles = pd.DataFrame(list(existing_articles.values()))

            # Update behaviors dataframe
            new_behaviors = pd.DataFrame({
                'session_id': chunk['session_id'],
                'impression_id': chunk['eventId'],
                'article_id': chunk.apply(lambda x: 'homepage' if x['is_homepage'] else
                                          existing_articles.get(x.get('id'), existing_articles.get(x['url'], {})).get('article_id'), axis=1),
                'user_id': chunk['userId'],
                # Convert to milliseconds
                'impression_time': chunk['time'] * 1000,
                'read_time': chunk['activeTime']
            })

            behaviors = pd.concat(
                [behaviors, new_behaviors], ignore_index=True)

            # Calculate user statistics for this chunk
            chunk_user_stats = chunk.groupby('userId').agg({
                'session_id': 'nunique',
                'eventId': 'count',
                'is_homepage': 'sum',
                'url': lambda x: sum('/pluss/' in url for url in x),
                'datetime': lambda x: {
                    'morning': ((x.dt.hour >= 6) & (x.dt.hour < 12)).mean(),
                    'afternoon': ((x.dt.hour >= 12) & (x.dt.hour < 18)).mean(),
                    'evening': ((x.dt.hour >= 18) & (x.dt.hour < 24)).mean(),
                    'night': ((x.dt.hour >= 0) & (x.dt.hour < 6)).mean()
                }
            }).reset_index()

            # Update user statistics
            for _, row in chunk_user_stats.iterrows():
                user_id = row['userId']
                if user_id not in user_stats:
                    user_stats[user_id] = {
                        'user_id': user_id,
                        'count_sessions': row['session_id'],
                        'count_total_impressions': row['eventId'],
                        'count_homepage_impressions': row['is_homepage'],
                        'is_subscriber': row['url'] > 0,
                        'count_impressions_with_article': row['eventId'] - row['is_homepage'],
                        'percentage_morning': row['datetime']['morning'],
                        'percentage_afternoon': row['datetime']['afternoon'],
                        'percentage_evening': row['datetime']['evening'],
                        'percentage_night': row['datetime']['night']
                    }
                else:
                    user_stats[user_id]['count_sessions'] += row['session_id']
                    user_stats[user_id]['count_total_impressions'] += row['eventId']
                    user_stats[user_id]['count_homepage_impressions'] += row['is_homepage']
                    user_stats[user_id]['is_subscriber'] = user_stats[user_id]['is_subscriber'] or (
                        row['url'] > 0)
                    user_stats[user_id]['count_impressions_with_article'] += (
                        row['eventId'] - row['is_homepage'])
                    # Update time percentages by weighted average
                    total_impressions = user_stats[user_id]['count_total_impressions']
                    chunk_impressions = row['eventId']
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        old_value = user_stats[user_id][f'percentage_{period}']
                        new_value = row['datetime'][period]
                        user_stats[user_id][f'percentage_{period}'] = (
                            (old_value * (total_impressions - chunk_impressions) +
                             new_value * chunk_impressions) / total_impressions
                        )

            # Convert user_stats to DataFrame
            users = pd.DataFrame(list(user_stats.values()))

            # Show progress
            progress = (chunk_idx / total_chunks) * 100
            print(
                f"\rProcessing chunk {chunk_idx}/{total_chunks} ({progress:.1f}%)", end='', flush=True)

            # Clean up memory
            del chunk
            gc.collect()

    print("\nProcessing complete!")

    # Add is_subscriber to behaviors after all chunks are processed
    behaviors['is_subscriber'] = behaviors['user_id'].map(
        dict(zip(users['user_id'], users['is_subscriber']))
    )

    # Give a top 5 of longest sessions and the number of rows in the session
    print("Top 5 longest sessions:")
    print(behaviors.groupby('session_id').size().nlargest(5))

    # Save to parquet files with compression
    os.makedirs(output_dir, exist_ok=True)
    users.to_parquet(os.path.join(
        output_dir, 'users.parquet'), compression='gzip')
    articles.to_parquet(os.path.join(
        output_dir, 'articles.parquet'), compression='gzip')
    behaviors.to_parquet(os.path.join(
        output_dir, 'behaviors.parquet'), compression='gzip')

    # Save sample data for debugging
    export_row_count = 2000
    users.head(export_row_count).to_csv(
        os.path.join(output_dir, 'users.csv'), index=False)
    articles.head(export_row_count).to_csv(
        os.path.join(output_dir, 'articles.csv'), index=False)
    behaviors[['session_id', 'impression_id', 'article_id', 'user_id', 'is_subscriber', 'impression_time', 'read_time']].head(export_row_count).to_csv(
        os.path.join(output_dir, 'behaviors.csv'), index=False)

    # Print summary statistics
    print("Users: ")
    print(users.head())
    print("\nArticles: ")
    print(articles.head())
    print("\nBehaviors: ")
    print(behaviors.head())


def extract_user_data(input_dir: str, output_dir: str, user_id: str):
    """
    Write the corresponding rows to a new jsonl file (just keep the exact rows of this user)

    Args:
        input_dir (str): Directory containing the input JSONL file
        output_dir (str): Directory where the output JSONL file will be written
        user_id (str): The user ID to extract events for
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the input filename from the input directory
    input_file = os.path.join(input_dir, "small.jsonl")
    output_file = os.path.join(output_dir, f"user_{user_id}.jsonl")

    # Open input and output files
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # Process each line in the input file
        for line in infile:
            try:
                # Parse the JSON line
                event = json.loads(line.strip())

                # Check if this event belongs to the target user
                if event.get('userId') == user_id:
                    # Write the exact line to the output file
                    outfile.write(line)

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                continue


if __name__ == '__main__':
    # Directory containing adressa JSONL files
    input_dir = 'adressa/datasets/one_week'
    # Directory to save output parquet files
    output_dir = 'adressa/datasets/one_week/converted'
    process_adressa_data(input_dir, output_dir)
