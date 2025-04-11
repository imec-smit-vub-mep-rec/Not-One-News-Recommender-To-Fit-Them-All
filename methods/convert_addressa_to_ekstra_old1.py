import json
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone
import uuid
import logging
import os
# import re # Not strictly needed in this version

# --- Configuration ---
INPUT_FILE = 'addressa/datasets/one_week/20170101.jsonl'  # Your input file path
ARTICLES_OUTPUT_FILE = 'articles_addressa.csv'
BEHAVIORS_OUTPUT_FILE = 'behaviors_addressa.csv'
CHUNK_SIZE = 100000  # Process and write data in chunks
SESSION_TIMEOUT_MINUTES = 30

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def safe_get(data, key, default=None):
    """Safely get a value from a dict."""
    return data.get(key, default)

def parse_profile(profile_list):
    """Parses the 'profile' list to extract structured data.
       Ensures list fields are initialized as empty lists.
    """
    data = {
        'category_str': None,
        'subcategory_ids': [], # Initialize as empty list
        'premium': False,
        'article_type': None,
        'ner_clusters': [], # Initialize as empty list
        'entity_groups': [], # Initialize as empty list
        'topics': [], # Initialize as empty list
        'sentiment_score': None,
        'sentiment_label': None
    }
    ner_items = set()
    entity_items = set()
    topic_items = set()
    # adressa_tags = set() # Removed if only used for topics
    categories = set()
    taxonomies = set()
    # concepts = set() # Removed if only used for topics

    if not isinstance(profile_list, list):
        logging.warning("Profile data is not a list, skipping profile parsing.")
        return data # Return default empty structure

    for item_profile in profile_list:
        if not isinstance(item_profile, dict): continue # Skip malformed profile items
        item = safe_get(item_profile, 'item')
        groups = safe_get(item_profile, 'groups', [])
        if not isinstance(groups, list): continue # Skip malformed groups

        for group_info in groups:
            if not isinstance(group_info, dict): continue # Skip malformed group info
            group = safe_get(group_info, 'group')
            weight = safe_get(group_info, 'weight')
            if group and item is not None: # Ensure item is not None before adding
                # Premium Status
                if group == 'adressa-access':
                    data['premium'] = (item != 'free')

                # Article Type
                if group == 'pageclass':
                    data['article_type'] = str(item) # Ensure string

                # Categories/Taxonomy
                if group == 'category':
                     categories.add(str(item))
                if group == 'taxonomy':
                     taxonomies.add(str(item))

                # Entities / NER
                if group in ['entity', 'person', 'location', 'company']:
                    ner_items.add(str(item))
                if group == 'entity':
                     entity_items.add(str(item)) # Separate list if needed

                # Topics
                if group in ['concept', 'classification', 'adressa-tag']:
                     topic_items.add(str(item))

                # Sentiment
                if group == 'sentiment':
                    data['sentiment_label'] = str(item)
                    # Attempt to convert weight to float, handle errors
                    try:
                        data['sentiment_score'] = float(weight) if weight is not None else None
                    except (ValueError, TypeError):
                        logging.warning(f"Could not convert sentiment weight '{weight}' to float.")
                        data['sentiment_score'] = None


    # Consolidate extracted items into lists/strings
    data['ner_clusters'] = sorted(list(ner_items)) # Sort for consistency
    data['entity_groups'] = sorted(list(entity_items)) # Sort for consistency
    data['topics'] = sorted(list(topic_items)) # Sort for consistency

    # Determine primary category string
    if taxonomies:
        sorted_tax = sorted(list(taxonomies), key=len, reverse=True)
        data['category_str'] = sorted_tax[0]
    elif categories:
        data['category_str'] = "|".join(sorted(list(categories)))

    # Ensure list fields exist, even if empty (redundant due to init, but safe)
    data['subcategory_ids'] = data.get('subcategory_ids', [])
    data['ner_clusters'] = data.get('ner_clusters', [])
    data['entity_groups'] = data.get('entity_groups', [])
    data['topics'] = data.get('topics', [])

    return data

def parse_publish_time(time_str):
    """Safely parses various ISO-like timestamp formats into UTC datetime."""
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        # Handle 'Z' timezone directly
        if time_str.endswith('Z'):
            time_str = time_str[:-1] + '+00:00'
        # Handle potential milliseconds before parsing
        if '.' in time_str:
            time_str = time_str.split('.')[0]

        # Try standard ISO format
        dt = datetime.fromisoformat(time_str)
        # Ensure timezone is set, default to UTC if naive
        if dt.tzinfo is None:
             return dt.replace(tzinfo=timezone.utc)
        # Convert to UTC if it has other timezone
        return dt.astimezone(timezone.utc)

    except ValueError:
        logging.warning(f"Could not parse timestamp: {time_str}")
        return None


def get_session_id(user_id, event_time_dt, session_start_flag, user_sessions, timeout):
    """Generates or retrieves a session ID for a user event."""
    session_info = user_sessions.get(user_id)
    new_session = False

    if not session_info:
        new_session = True
    else:
        last_event_time, _ = session_info
        # Check if event_time_dt is timezone-aware before comparison
        if last_event_time.tzinfo is None and event_time_dt.tzinfo is not None:
            last_event_time = last_event_time.replace(tzinfo=timezone.utc) # Assume UTC if naive
        elif last_event_time.tzinfo is not None and event_time_dt.tzinfo is None:
             event_time_dt = event_time_dt.replace(tzinfo=timezone.utc) # Assume UTC if naive

        if session_start_flag or (event_time_dt - last_event_time > timeout):
            new_session = True

    if new_session:
        session_uuid = uuid.uuid4()
        user_sessions[user_id] = (event_time_dt, session_uuid)
        return str(session_uuid)
    else:
        # Update last event time for the existing session
        _, session_uuid = session_info
        user_sessions[user_id] = (event_time_dt, session_uuid)
        return str(session_uuid)

# --- Main Processing Logic ---

processed_article_ids = set()
articles_data = []
behaviors_data = []
user_sessions = {} # {user_id: (last_event_timestamp_dt, session_uuid)}
session_timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
line_count = 0
processed_events = 0 # Count successfully processed events for behaviors

# --- Define PyArrow Schemas ---
# Explicit schemas help prevent type inference issues across chunks

# Remove PyArrow schemas as they're not needed for CSV
# articles_schema = pa.schema([...])
# behaviors_schema = pa.schema([...])

# --- Use CSV writing approach ---
# For CSV, we'll use a different approach since CSV doesn't support appending in the same way as Parquet
articles_first_chunk = True
behaviors_first_chunk = True

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            line_count += 1
            try:
                # Sanitize input slightly - remove control characters that can break JSON
                clean_line = ''.join(c for c in line if c.isprintable() or c in '\n\r\t')
                event_data = json.loads(clean_line.strip())
                if not isinstance(event_data, dict):
                    logging.warning(f"Skipping non-dictionary JSON object on line {line_count}")
                    continue
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON on line {line_count}: {e}")
                continue

            # --- Process Article Data (if profile exists) ---
            has_profile = 'profile' in event_data and isinstance(event_data['profile'], list)
            article_id_from_event = safe_get(event_data, 'id') if has_profile else None

            if article_id_from_event and article_id_from_event not in processed_article_ids:
                profile_info = parse_profile(event_data['profile'])

                # Ensure all list fields are actually lists
                ner_clusters = profile_info.get('ner_clusters', [])
                entity_groups = profile_info.get('entity_groups', [])
                topics = profile_info.get('topics', [])
                subcategory_ids = profile_info.get('subcategory_ids', [])
                # image_ids needs to be handled if extracted

                article_row = {
                    'article_id': str(article_id_from_event), # Ensure string
                    'title': str(safe_get(event_data, 'title','')), # Ensure string, handle None
                    'subtitle': None, # Not available
                    'body': None, # Not available
                    'category_id': None, # Requires mapping
                    'category_str': str(profile_info['category_str']) if profile_info['category_str'] else None, # Ensure string or None
                    'subcategory_ids': subcategory_ids, # Already a list
                    'premium': profile_info.get('premium', False), # Already bool
                    'published_time': parse_publish_time(safe_get(event_data, 'publishtime')), # Use parser
                    'last_modified_time': None, # Not available
                    'image_ids': [], # Placeholder - ensure empty list if not available
                    'article_type': str(profile_info['article_type']) if profile_info['article_type'] else None, # Ensure string or None
                    'url': str(safe_get(event_data, 'canonicalUrl') or safe_get(event_data, 'url','')), # Ensure string, handle None
                    'ner_clusters': ner_clusters, # Already a list
                    'entity_groups': entity_groups, # Already a list
                    'topics': topics, # Already a list
                    'total_inviews': None,
                    'total_pageviews': None,
                    'total_read_time': None,
                    'sentiment_score': profile_info.get('sentiment_score'), # Already float or None
                    'sentiment_label': str(profile_info['sentiment_label']) if profile_info['sentiment_label'] else None # Ensure string or None
                }
                articles_data.append(article_row)
                processed_article_ids.add(article_id_from_event)

            # --- Process Behavior Data (for every event) ---
            user_id = safe_get(event_data, 'userId')
            event_id = safe_get(event_data, 'eventId')
            event_time_unix = safe_get(event_data, 'time')
            active_time = safe_get(event_data, 'activeTime')

            # Behavior links to article_id if the event has 'id' (even if no profile/not new)
            behavior_article_id = safe_get(event_data, 'id')

            # Check essential fields for behavior record
            if not all([user_id, event_id, event_time_unix is not None]):
                 logging.warning(f"Skipping event on line {line_count} due to missing userId, eventId, or time.")
                 continue # Skip if core info missing

            try:
                event_time_dt = datetime.fromtimestamp(int(event_time_unix), tz=timezone.utc)
            except (ValueError, TypeError, OverflowError) as e:
                 logging.warning(f"Could not parse timestamp '{event_time_unix}' for event {event_id} on line {line_count}: {e}")
                 continue # Skip behavior if timestamp is invalid

            session_start = safe_get(event_data, 'sessionStart', False)
            session_id = get_session_id(str(user_id), event_time_dt, session_start, user_sessions, session_timeout) # Ensure user_id is string

            # Determine clicked articles - simplified
            clicked_ids = [str(behavior_article_id)] if behavior_article_id else []

            # Handle read_time conversion safely
            read_time_val = None
            if active_time is not None:
                try:
                    read_time_val = int(active_time)
                except (ValueError, TypeError):
                    logging.warning(f"Could not convert activeTime '{active_time}' to int for event {event_id} on line {line_count}")

            behavior_row = {
                'impression_id': int(event_id), # Ensure int
                'user_id': str(user_id), # Ensure string
                'article_id': str(behavior_article_id) if behavior_article_id else None, # Ensure string or None
                'session_id': session_id, # Already string
                'article_ids_inview': None, # Not available, ensure None or [] as per schema intent
                'article_ids_clicked': clicked_ids, # Already list of strings or empty list
                'impression_time': event_time_dt, # Already datetime
                'read_time': read_time_val, # int or None
                'scroll_percentage': None, # Not available
                'device_type': str(safe_get(event_data, 'deviceType','Unknown')), # Ensure string, handle None
                'is_sso_user': None,
                'gender': None,
                'postcode': None,
                'age': None,
                'is_subscriber': None,
                'next_read_time': None,
                'next_scroll_percentage': None
            }
            behaviors_data.append(behavior_row)
            processed_events += 1


            # --- Write Chunk periodically ---
            if line_count % CHUNK_SIZE == 0:
                logging.info(f"Processed {line_count} lines. Writing chunk...")

                # Write Articles Chunk
                if articles_data:
                    try:
                        articles_df = pd.DataFrame(articles_data)
                        # --- Explicitly set dtypes for articles_df before converting to Arrow ---
                        articles_df = articles_df.astype({
                            'article_id': 'string', # Pandas nullable string
                            'title': 'string',
                            'subtitle': 'string',
                            'body': 'string',
                            'category_id': 'Int64', # Pandas nullable integer
                            'category_str': 'string',
                            'subcategory_ids': object, # Use object for lists
                            'premium': bool, # Standard bool
                            'published_time': 'datetime64[ns, UTC]',
                            'last_modified_time': 'datetime64[ns, UTC]',
                            'image_ids': object, # Use object for lists
                            'article_type': 'string',
                            'url': 'string',
                            'ner_clusters': object, # Use object for lists
                            'entity_groups': object, # Use object for lists
                            'topics': object, # Use object for lists
                            'total_inviews': 'Int64',
                            'total_pageviews': 'Int64',
                            'total_read_time': 'Float64', # Pandas nullable float
                            'sentiment_score': 'Float64',
                            'sentiment_label': 'string'
                        }, errors='ignore') # ignore errors if a column is missing (shouldn't happen here)

                        # Write to CSV with header only for the first chunk
                        articles_df.to_csv(ARTICLES_OUTPUT_FILE, mode='w' if articles_first_chunk else 'a', 
                                          header=articles_first_chunk, index=False)
                        articles_first_chunk = False
                        articles_data = [] # Clear chunk data
                    except Exception as e:
                        logging.error(f"Error writing articles chunk at line {line_count}: {e}", exc_info=True)
                        # Decide whether to stop or continue
                        # raise # Re-raise to stop execution
                        # continue # Or just log and try to continue with next chunk


                # Write Behaviors Chunk
                if behaviors_data:
                    try:
                        behaviors_df = pd.DataFrame(behaviors_data)
                        # --- Explicitly set dtypes for behaviors_df ---
                        behaviors_df = behaviors_df.astype({
                            'impression_id': 'int64', # Assuming it fits
                            'user_id': 'string',
                            'article_id': 'string',
                            'session_id': 'string',
                            'article_ids_inview': object,
                            'article_ids_clicked': object,
                            'impression_time': 'datetime64[ns, UTC]',
                            'read_time': 'Int64',
                            'scroll_percentage': 'Float64',
                            'device_type': 'string',
                            'is_sso_user': 'boolean', # Pandas nullable boolean
                            'gender': 'string',
                            'postcode': 'string',
                            'age': 'Int64',
                            'is_subscriber': 'boolean',
                            'next_read_time': 'Int64',
                            'next_scroll_percentage': 'Float64'
                        }, errors='ignore')

                        # Write to CSV with header only for the first chunk
                        behaviors_df.to_csv(BEHAVIORS_OUTPUT_FILE, mode='w' if behaviors_first_chunk else 'a', 
                                           header=behaviors_first_chunk, index=False)
                        behaviors_first_chunk = False
                        behaviors_data = [] # Clear chunk data
                    except Exception as e:
                        logging.error(f"Error writing behaviors chunk at line {line_count}: {e}", exc_info=True)
                        # Decide whether to stop or continue
                        # raise # Re-raise to stop execution
                        # continue # Or just log and try to continue with next chunk

                logging.info(f"Chunk written. Total events processed for behavior: {processed_events}")

    # --- Write Final Remaining Chunk ---
    logging.info("Writing final chunk...")
    if articles_data:
        try:
            articles_df = pd.DataFrame(articles_data)
            articles_df = articles_df.astype({
                'article_id': 'string', 'title': 'string', 'subtitle': 'string', 'body': 'string',
                'category_id': 'Int64', 'category_str': 'string', 'subcategory_ids': object, 'premium': bool,
                'published_time': 'datetime64[ns, UTC]', 'last_modified_time': 'datetime64[ns, UTC]',
                'image_ids': object, 'article_type': 'string', 'url': 'string', 'ner_clusters': object,
                'entity_groups': object, 'topics': object, 'total_inviews': 'Int64', 'total_pageviews': 'Int64',
                'total_read_time': 'Float64', 'sentiment_score': 'Float64', 'sentiment_label': 'string'
            }, errors='ignore')
            
            # Write to CSV with header only if this is the first chunk
            articles_df.to_csv(ARTICLES_OUTPUT_FILE, mode='w' if articles_first_chunk else 'a', 
                              header=articles_first_chunk, index=False)
        except Exception as e:
            logging.error(f"Error writing final articles chunk: {e}", exc_info=True)


    if behaviors_data:
        try:
            behaviors_df = pd.DataFrame(behaviors_data)
            behaviors_df = behaviors_df.astype({
                'impression_id': 'int64', 'user_id': 'string', 'article_id': 'string', 'session_id': 'string',
                'article_ids_inview': object, 'article_ids_clicked': object, 'impression_time': 'datetime64[ns, UTC]',
                'read_time': 'Int64', 'scroll_percentage': 'Float64', 'device_type': 'string',
                'is_sso_user': 'boolean', 'gender': 'string', 'postcode': 'string', 'age': 'Int64',
                'is_subscriber': 'boolean', 'next_read_time': 'Int64', 'next_scroll_percentage': 'Float64'
            }, errors='ignore')
            
            # Write to CSV with header only if this is the first chunk
            behaviors_df.to_csv(BEHAVIORS_OUTPUT_FILE, mode='w' if behaviors_first_chunk else 'a', 
                               header=behaviors_first_chunk, index=False)
        except Exception as e:
            logging.error(f"Error writing final behaviors chunk: {e}", exc_info=True)


    logging.info(f"Processing finished. Total lines read: {line_count}. Total events processed for behavior: {processed_events}")
    logging.info(f"Unique articles found: {len(processed_article_ids)}")

except FileNotFoundError:
    logging.error(f"Error: Input file '{INPUT_FILE}' not found.")
except Exception as e:
    logging.error(f"An unexpected error occurred during file processing: {e}", exc_info=True)
finally:
    # No need to close CSV writers as they're handled by pandas
    logging.info(f"Articles CSV file saved: {ARTICLES_OUTPUT_FILE}")
    logging.info(f"Behaviors CSV file saved: {BEHAVIORS_OUTPUT_FILE}")