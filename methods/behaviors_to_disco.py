import pandas as pd
from datetime import datetime

# Load the behaviors.parquet dataset
file_path = "datasets/ekstra/behaviors.parquet"
df = pd.read_parquet(file_path)


def get_scroll_category(scroll_depth):
    if pd.isna(scroll_depth):
        return "Article not scrolled"
    scroll_depth = float(scroll_depth)
    if scroll_depth == 100:
        return "Article scrolled 100%"
    elif scroll_depth > 50:
        return "Article scrolled > 50%"
    else:
        return "Article scrolled <= 50%"


def get_time_of_day(timestamp):
    hour = pd.to_datetime(timestamp).hour
    if 0 <= hour < 6:
        return "Night (0-6)"
    elif 6 <= hour < 12:
        return "Morning (6-12)"
    elif 12 <= hour < 18:
        return "Afternoon (12-18)"
    else:
        return "Evening (18-24)"


def extract_events(df):
    event_log = []

    # Group by user_id and session_id to process sessions
    grouped = df.groupby(['user_id', 'session_id'])

    for (user_id, session_id), session_group in grouped:
        # Create a unique Case ID per session
        case_id = f"{user_id}_{session_id}"

        # Get first and last timestamps for the session
        first_timestamp = session_group['impression_time'].min()
        last_timestamp = session_group['impression_time'].max()

        # Get device type for the session
        device_type = session_group['device_type'].iloc[0]
        device_map = {1: "Desktop", 2: "Mobile", 3: "Tablet", 0: "Unknown"}
        device = device_map.get(device_type, "Unknown")

        # Get subscriber status for the session
        is_subscriber = session_group['is_subscriber'].iloc[0]

        # Get time of day classification for session start
        time_of_day = get_time_of_day(first_timestamp)

        # Add session start event
        event_log.append(
            (case_id, f"Session Started on {device}", first_timestamp, is_subscriber, time_of_day))

        # Process each interaction in the session
        for _, row in session_group.iterrows():
            timestamp = row["impression_time"]
            article_id = row["article_id"]
            clicked_articles = row["article_ids_clicked"]
            scroll_depth = row.get("scroll_percentage", None)

            # Record page view events
            if pd.isna(article_id):
                event_log.append(
                    (case_id, f"Homepage Viewed on {device}", timestamp, is_subscriber, time_of_day))
            else:
                event_log.append((case_id, f"Article Viewed", timestamp, is_subscriber, time_of_day))
                # Add scroll depth event immediately after article view
                scroll_category = get_scroll_category(scroll_depth)
                event_log.append((case_id, scroll_category, timestamp, is_subscriber, time_of_day))

                # Record click events
                if isinstance(clicked_articles, list) and article_id in clicked_articles:
                    event_log.append((case_id, f"Article Clicked", timestamp, is_subscriber, time_of_day))

        # Add session end event
        event_log.append(
            (case_id, f"Session Ended on {device}", last_timestamp, is_subscriber, time_of_day))

    # Convert event_log to DataFrame
    event_df = pd.DataFrame(
        event_log, columns=["Case ID", "Activity Name", "Timestamp", "Is Subscriber", "Session Start Time"])

    # Add human readable timestamp
    event_df["Human Readable Time"] = pd.to_datetime(
        event_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    event_df.sort_values(by=["Case ID", "Timestamp"], inplace=True)

    return event_df


# Process the event log
event_df = extract_events(df)

# Save to CSV for Disco
event_df.to_csv("disco_behaviors_log.csv", index=False)
print("Event log saved as disco_behaviors_log.csv")
