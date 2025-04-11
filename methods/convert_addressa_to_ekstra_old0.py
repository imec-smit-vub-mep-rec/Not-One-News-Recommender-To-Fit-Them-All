import pandas as pd
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import os
import tempfile
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def string_to_int_id(string_id):
    """Convert a string ID to an integer using a hash function."""
    # Use a hash function to convert the string to an integer
    hash_object = hashlib.md5(string_id.encode())
    # Take the first 8 characters of the hex digest and convert to integer
    # This gives us a 32-bit integer (8 hex chars = 32 bits)
    return int(hash_object.hexdigest()[:8], 16)


def extract_category(profile):
    """Extract category from profile data."""
    if not profile:
        return ""

    for profile_item in profile:
        if 'groups' in profile_item:
            for group in profile_item['groups']:
                if 'group' in group and group['group'] == 'category':
                    return profile_item.get('item', '')

    return ""


def process_json_file(json_file, articles_dict, chunk_size=10000):
    """Process a single JSON file in chunks."""
    behaviors_chunk = []
    chunk_count = 0

    with open(json_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Skip records without 'id'
                if 'id' not in record:
                    continue

                # Check if this is an article record (has 'id' and 'title')
                if 'title' in record and record['id'] not in articles_dict:
                    # Extract category from profile
                    category = extract_category(record.get('profile', []))

                    # Extract topics from profile if available
                    topics = []
                    if 'profile' in record:
                        for profile_item in record['profile']:
                            if 'groups' in profile_item:
                                for group in profile_item['groups']:
                                    if 'group' in group and group['group'] in ['category', 'taxonomy', 'concept']:
                                        topics.append(
                                            profile_item.get('item', ''))

                    # Add keywords if available
                    if 'keywords' in record and record['keywords']:
                        topics.extend(record['keywords'].split(','))

                    # Remove duplicates
                    topics = list(set(topics))

                    # Handle publish time
                    publish_time = record.get('publishtime', '')
                    if publish_time:
                        try:
                            # Try to parse ISO format
                            dt = datetime.fromisoformat(
                                publish_time.replace('Z', '+00:00'))
                            publish_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            # If parsing fails, keep the original string
                            pass

                    article = {
                        # Use hash for article ID too
                        'article_id': string_to_int_id(record['id']),
                        'title': record.get('title', ''),
                        'subtitle': '',  # Not available
                        'body': '',  # Not available
                        'category_id': 0,  # Not available
                        'category_str': category,  # Extract category
                        'subcategory_ids': [],  # Not available
                        'premium': False,  # Not available
                        'published_time': publish_time,
                        'last_modified_time': publish_time,
                        'image_ids': [],  # Not available
                        'article_type': 'article_default',
                        'url': record.get('canonicalUrl', record.get('url', '')),
                        'ner_clusters': [],  # Not available
                        'entity_groups': [],  # Not available
                        'topics': topics,
                        'total_inviews': None,
                        'total_pageviews': None,
                        'total_read_time': None,
                        'sentiment_score': None,
                        'sentiment_label': None
                    }
                    articles_dict[record['id']] = article

                # Check if this is a behavior record (has 'eventId' and 'userId')
                if 'eventId' in record and 'userId' in record:
                    # Always use the hash function for user IDs to handle all formats
                    user_id = string_to_int_id(record['userId'])

                    # Extract article ID if available
                    article_id = None
                    if 'id' in record:
                        article_id = string_to_int_id(record['id'])

                    # Convert timestamp to datetime string
                    impression_time = ''
                    if 'time' in record:
                        try:
                            # First try to convert from Unix timestamp
                            dt = datetime.fromtimestamp(record['time'])
                            impression_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, TypeError):
                            # If that fails, try to parse as string
                            try:
                                dt = datetime.strptime(
                                    str(record['time']), '%Y-%m-%d %H:%M:%S')
                                impression_time = dt.strftime(
                                    '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                # If all parsing fails, use a default value
                                impression_time = '1970-01-01 00:00:00'

                    behavior = {
                        'impression_id': int(record['eventId']),
                        'user_id': user_id,
                        'article_id': article_id,
                        'session_id': int(record['eventId']),
                        'article_ids_inview': [],
                        'article_ids_clicked': [article_id] if article_id else [],
                        'impression_time': impression_time,
                        'read_time': float(record.get('activeTime', 0)),
                        'scroll_percentage': None,
                        'device_type': 0,
                        'is_sso_user': False,
                        'gender': None,
                        'postcode': None,
                        'age': None,
                        'is_subscriber': False,
                        'next_read_time': None,
                        'next_scroll_percentage': None
                    }

                    # Map device type
                    device_type = record.get('deviceType', '').lower()
                    if 'mobile' in device_type:
                        behavior['device_type'] = 2
                    elif 'tablet' in device_type:
                        behavior['device_type'] = 3
                    elif 'desktop' in device_type:
                        behavior['device_type'] = 1

                    behaviors_chunk.append(behavior)

                # Process chunk when it reaches the size limit
                if len(behaviors_chunk) >= chunk_size:
                    # Convert to DataFrame and ensure impression_time is datetime
                    chunk_df = pd.DataFrame(behaviors_chunk)
                    # Convert impression_time to datetime
                    chunk_df['impression_time'] = pd.to_datetime(
                        chunk_df['impression_time'], errors='coerce')
                    # Convert back to string in the correct format
                    chunk_df['impression_time'] = chunk_df['impression_time'].dt.strftime(
                        '%Y-%m-%d %H:%M:%S')
                    yield chunk_df
                    behaviors_chunk = []
                    chunk_count += 1

            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line in {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing record in {json_file}: {e}")
                continue

    # Yield the last chunk if not empty
    if behaviors_chunk:
        # Convert to DataFrame and ensure impression_time is datetime
        chunk_df = pd.DataFrame(behaviors_chunk)
        # Convert impression_time to datetime
        chunk_df['impression_time'] = pd.to_datetime(
            chunk_df['impression_time'], errors='coerce')
        # Convert back to string in the correct format
        chunk_df['impression_time'] = chunk_df['impression_time'].dt.strftime(
            '%Y-%m-%d %H:%M:%S')
        yield chunk_df
        chunk_count += 1

    logger.info(f"Processed {chunk_count} chunks from {json_file}")


def main():
    addressa_path = "addressa/datasets"  # Adjust path as needed
    output_path = "datasets"

    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Dictionary to store unique articles
    articles_dict = {}

    # Process each JSON file and write behaviors in chunks
    data_path = Path(addressa_path) / "one_week"
    if not data_path.exists():
        logger.error(f"Data path {data_path} does not exist")
        return

    # Create a list to store all behavior DataFrames
    behavior_dfs = []
    total_behaviors = 0

    for json_file in sorted(data_path.glob("*.json")):
        logger.info(f"Processing {json_file}")

        for chunk_df in process_json_file(json_file, articles_dict):
            # Store the chunk DataFrame
            behavior_dfs.append(chunk_df)
            total_behaviors += len(chunk_df)
            logger.info(f"Processed {total_behaviors} behaviors so far")

    # Convert articles dictionary to DataFrame and save
    logger.info("Saving articles...")
    articles_df = pd.DataFrame(list(articles_dict.values()))
    articles_df.to_parquet(f"{output_path}/articles.parquet", index=False)

    # Combine all behavior DataFrames and save
    logger.info("Saving behaviors...")
    if behavior_dfs:
        behaviors_df = pd.concat(behavior_dfs, ignore_index=True)
        behaviors_df.to_parquet(
            f"{output_path}/behaviors.parquet", index=False)

    logger.info("Conversion complete!")
    logger.info(
        f"Processed {len(articles_df)} articles and {total_behaviors} behaviors")


if __name__ == "__main__":
    main()
