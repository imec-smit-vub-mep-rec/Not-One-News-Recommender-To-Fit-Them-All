# General Data Format
## INPUT DATA (LOGS):

### ARTICLES:

| Name           | Data Type     | Description                                                                                                                                                       |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| article_id     | string        | Unique identifier for each article                                                                                                                                |
| time_published | timestamp     | Timestamp of when the article was published                                                                                                                       |
| title          | string        | String of the article title                                                                                                                                       |
| category_str   | string        | String of the (main) article category -> **is there always a main category? YES, but sometimes 2 main categories -> (can be cleaned, eg "algemeen" and "sport")** |
| categories     | array[string] | List of strings of the article categories                                                                                                                         |
| article_length | number        | Number of words in the article                                                                                                                                    |

### IMPRESSIONS:

| Name           | Data Type | Description                                                                                                                                                                                                          |
| -------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| impression_id  | string    | Unique identifier for each impression                                                                                                                                                                                |
| start_time     | timestamp | Timestamp of when the impression occurred                                                                                                                                                                            |
| active_seconds | number    | Number of seconds the user spent active on the page                                                                                                                                                                  |
| session_id     | string    | Unique identifier for each user session (one session can contain multiple impressions) -> **What is your definition of 'a session'?**                                                                                |
| user_id        | string    | Unique identifier for the user -> **What about anonymous users? Do they get unique 'throwaway' user_ids? No ids? Recurring fingerprint-based ids? -> Only device. Logged in user subscription status in de events.** |
| article_id     | string    | Unique identifier for the article that was viewed -> -1 if homepage view. **Are homepage views also included in the dataset?**                                                                                       |
| read_time      | number    | Number of seconds the user spent reading the article                                                                                                                                                                 |
| scroll_depth   | number    | Percentage of the article that the user scrolled through **mogelijks buggy**                                                                                                                                         |
| device_type    | string    | String of the device type that the user used to view the article                                                                                                                                                     |
| is_subscriber  | boolean   | Boolean indicating if the user is a paid subscriber                                                                                                                                                                  |
| day_of_week    | string    | String of the day of the week that the impression occurred                                                                                                                                                           |

---

## OUTPUT DATA:

### USERS TABLE:

| Name                              | Data Type | Description                                                                                                     |
| --------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------- |
| user_id                           | string    | Unique identifier for the user                                                                                  |
| count_sessions                    | number    | Total number of sessions for the user                                                                           |
| has_account                       | boolean   | Boolean indicating if the user has an account                                                                   |
| count_total_impressions           | number    | Total number of impressions for the user                                                                        |
|  proportion_morning_impressions   | number    | Proportion of impressions during morning hours(all impression during morning hours / total impressions)         |
| proportion_afternoon_impressions  | number    | Proportion of impressions during afternoon hours(all impression during afternoon hours / total impressions)     |
| proportion_evening_impressions    | number    | Proportion of impressions during evening hours(all impression during evening hours / total impressions)         |
| proportion_night_impressions      | number    | Proportion of impressions during night hours(all impression during night hours / total impressions)             |
| count_total_homepage_impressions  | number    | Total number of homepage impressions for the user                                                               |
| count_total_article_impressions   | number    | Total number of article impressions for the user                                                                |
| count_total_unique_categories     | number    | Total number of unique categories viewed by the user                                                            |
| count_total_unique_articles       | number    | Total number of unique articles viewed by the user                                                              |
| total_reading_time                | number    | Total reading time in seconds                                                                                   |
| proportion_morning_reading_time   | number    | Proportion of reading time during morning hours(all reading time during morning hours / total reading time)     |
| proportion_afternoon_reading_time | number    | Proportion of reading time during afternoon hours(all reading time during afternoon hours / total reading time) |
| proportion_evening_reading_time   | number    | Proportion of reading time during evening hours(all reading time during evening hours / total reading time)     |
| proportion_night_reading_time     | number    | Proportion of reading time during night hours(all reading time during night hours / total reading time)         |
| avg_reading_time                  | number    | Average reading time in seconds                                                                                 |
| avg_session_length                | number    | Average session length in number of impressions                                                                 |
| avg_session_duration              | number    | Average session duration in seconds                                                                             |
| avg_categories_per_session        | number    | Average number of categories per session                                                                        |
| avg_category_switches_per_session | number    | Average number of category switches per session                                                                 |
| avg_reading_time_homepage         | number    | Average reading time in seconds on homepage                                                                     |
| avg_reading_time_articles         | number    | Average reading time in seconds on articles                                                                     |
| is_subscriber                     | boolean   | Boolean indicating if the user is a paid subscriber                                                             |
| cluster_id                        | string    | Identifier of the cluster the user belongs to                                                                   |

---

### ARTICLES TABLE:

| Name                              | Data Type     | Description                                                                                                 |
| --------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| article_id                        | string        | Unique identifier for each article                                                                          |
| time_published                    | timestamp     | Timestamp of when the article was published                                                                 |
| title                             | string        | String of the article title                                                                                 |
| category_str                      | string        | String of the (main) article category -> **is there always a main category?**                               |
| categories                        | array[string] | List of strings of the article categories                                                                   |
| article_length                    | number        | Number of words in the article                                                                              |
| count_impressions                 | number        | Total number of impressions for the article                                                                 |
| sentiment_score                   | number        | Sentiment score of the article                                                                              |
| total_reading_time                | number        | Total reading time in seconds across all impressions                                                        |
| avg_reading_time                  | number        | Average reading time in seconds per impression                                                              |
| proportion_morning_reading_time   | number        | Proportion of reading time during morning hours                                                             |
| proportion_afternoon_reading_time | number        | Proportion of reading time during afternoon hours                                                           |
| proportion_evening_reading_time   | number        | Proportion of reading time during evening hours                                                             |
| proportion_night_reading_time     | number        | Proportion of reading time during night hours                                                               |
| proportion_subscribers            | number        | Proportion of subscribers that read the article                                                             |
| proportion_morning_impressions    | number        | Proportion of impressions during morning hours(all impression during morning hours / total impressions)     |
| proportion_afternoon_impressions  | number        | Proportion of impressions during afternoon hours(all impression during afternoon hours / total impressions) |
| proportion_evening_impressions    | number        | Proportion of impressions during evening hours(all impression during evening hours / total impressions)     |
| proportion_night_impressions      | number        | Proportion of impressions during night hours(all impression during night hours / total impressions)         |

### IMPRESSIONS TABLE:

| Name          | Data Type | Description                                                                                                                     |
| ------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------- |
| impression_id | string    | Unique identifier for each impression                                                                                           |
| timestamp     | timestamp | Timestamp of when the impression occurred                                                                                       |
| session_id    | string    | Unique identifier for each user session                                                                                         |
| user_id       | string    | Unique identifier for the user                                                                                                  |
| article_id    | string    | Unique identifier for the article that was viewed (-1 if homepage view: other non-article views should be removed from dataset) |
| read_time     | number    | Number of seconds the user spent reading the article                                                                            |
| scroll_depth  | number    | Percentage of the article that the user scrolled through                                                                        |
| device_type   | string    | String of the device type that the user used to view the article                                                                |
| is_subscriber | boolean   | Boolean indicating if the user is a paid subscriber                                                                             |
| day_of_week   | string    | String of the day of the week that the impression occurred                                                                      |

## CLUSTERING RESULTS:

### CLUSTERS TABLE:

| Name                              | Data Type | Description                                                                                                 |
| --------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------- |
| cluster_id                        | string    | Unique identifier for the cluster                                                                           |
| cluster_name                      | string    | Name of the cluster                                                                                         |
| cluster_description               | string    | Description of the cluster                                                                                  |
| cluster_size_absolute             | number    | Absolute number of users in the cluster                                                                     |
| cluster_size_percentage           | number    | Percentage of total users in the cluster                                                                    |
| avg_reading_time                  | number    | Average reading time in seconds for users in the cluster                                                    |
| avg_scroll_depth                  | number    | Average scroll percentage for users in the cluster                                                          |
| avg_session_length_articles       | number    | Average number of articles per session in the cluster                                                       |
| avg_categories_per_session        | number    | Average number of categories per session in the cluster                                                     |
| avg_category_switches_per_session | number    | Average number of category switches per session in the cluster                                              |
| avg_session_duration_seconds      | number    | Average session duration in seconds for the cluster                                                         |
| proportion_morning_reading_time   | number    | Proportion of reading time during morning hours                                                             |
| proportion_afternoon_reading_time | number    | Proportion of reading time during afternoon hours                                                           |
| proportion_evening_reading_time   | number    | Proportion of reading time during evening hours                                                             |
| proportion_night_reading_time     | number    | Proportion of reading time during night hours                                                               |
| proportion_morning_impressions    | number    | Proportion of impressions during morning hours(all impression during morning hours / total impressions)     |
| proportion_afternoon_impressions  | number    | Proportion of impressions during afternoon hours(all impression during afternoon hours / total impressions) |
| proportion_evening_impressions    | number    | Proportion of impressions during evening hours(all impression during evening hours / total impressions)     |
| proportion_night_impressions      | number    | Proportion of impressions during night hours(all impression during night hours / total impressions)         |
| proportion_users_with_account     | number    | Proportion of users with an account in the cluster                                                          |
| proportion_subscribers            | number    | Proportion of subscribers in the cluster                                                                    |
| proportion_homepage_impressions   | number    | Proportion of homepage impressions in the cluster                                                           |
| proportion_homepage_time          | number    | Proportion of time spent on homepage in the cluster                                                         |
| most_active_hour                  | string    | Most active hour of the day for the cluster                                                                 |
| most_active_day_of_week           | string    | Most active day of the week for the cluster                                                                 |
