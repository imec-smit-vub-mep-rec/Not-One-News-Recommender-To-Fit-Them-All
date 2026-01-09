# Create venv

`python -m venv venv`

# Activate venv

`source venv/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# Workflow

## 1. Convert adressa to Ekstra format

`python 1.adressa_to_ekstra_format.py`

## 2. User clustering for each dataset

`python 2.user_clustering.py`

## 3. Convert behaviors to interactions

`python 3.behaviors_to_interactions.py`

## 4. Create content templates for articles

These will be embedded using sentence transformers in the content-based algorithm.
`python 4.articles_to_content.py`

## 5. Recpack evaluation

Copy the outputted clusters (in csv format) to the 2-recpack-evaluation/{dataset}/1.input.clusters folder.
Run the recpack evaluation pipeline for each dataset.

# Example folder structure after running the workflow

```
adressa/
├── datasets/one-week
datasets/
├── ekstra-large/
│ ├── articles.parquet
│ ├── behaviors.parquet
│ ├── interactions.csv
├── adressa/
│ ├── articles.parquet
│ ├── behaviors.parquet
│ ├── interactions.csv
```

# EB-NeRD Dataset format

- Each dataset bundle—demo, small, and large—consists of a training set and validation set, together with the articles (articles.parquet) present in the bundle. The official test set is to be downloaded separately from these. Each data split has two files: 1) the behavior logs for the 7-day data split period (behaviors.parquet) and 2) the users' click histories (history.parquet), i.e., 21 days of clicked news articles prior to the data split's behavior logs. The click histories are fixed to the period prior to the behavior logs; i.e., they are not updated within the data split period.
- Explore it easily using https://www.tablab.app/view
- Preview this markdown with https://markdownlivepreview.com/

## 1. Dataset: articles.parquet

All articles available during the period.

### Column overview

| Column             | Context                                                                                                                                                                                                                | Example                                          | dtype        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------ | -------------------------- |
| 1 Article ID       | The unique ID of a news article.                                                                                                                                                                                       | 8987932                                          | i32          |
| 2 Title            | The article's Danish title.                                                                                                                                                                                            | Se billederne: Zlatans paradis til salg          | str          |
| 3 Subtitle         | The article's Danish subtitle.                                                                                                                                                                                         | Zlatan Ibrahimovic har sat (. . . ).             | str          |
| 4 Body             | The article's full Danish text body.                                                                                                                                                                                   | Drømmer du om en eksklusiv (. . . ).             | str          |
| 5 Category ID      | The category ID.                                                                                                                                                                                                       | 142                                              | i16          |
| 6 Category string  | The category as a string.                                                                                                                                                                                              | sport                                            | str          |
| 7 Subcategory IDs  | The subcategory IDs.                                                                                                                                                                                                   | [196, 271]                                       | list[i16]    |
| 8 Premium          | Whether the content is behind a paywall.                                                                                                                                                                               | False                                            | bool         |
| 9 Time published   | The time the article was published. The format is "YYYY/MM/DD HH:MM:SS".                                                                                                                                               | 2021-11-15 03:56:56                              | datetime[𝜇𝑠] |
| 10 Time modified   | The timestamp for the last modification of the article, e.g., updates as the story evolves or spelling corrections. The format is "YYYY/MM/DD HH:MM:SS".                                                               | 2023-06-29 06:38:41                              | datetime[𝜇𝑠] |
| 11 Image IDs       | The image IDs used in the article.                                                                                                                                                                                     | [8988118]                                        | list[i64]    |
| 12 Article type    | The type of article, such as a feature, gallery, video, or live blog.                                                                                                                                                  | article_default                                  | str          |
| 13 URL             | The article's URL.                                                                                                                                                                                                     | https://ekstrabladet.dk/.../8987932              | str          |
| 14 NER             | The tags retrieved from a proprietary named-entityrecognition model at Ekstra Bladet are based on the concatenated title, abstract, and body.                                                                          | ['Aftonbladet', 'Sverige', 'Zlatan Ibrahimovic'] | list[str]    |
| 15 Entities        | The tags retrieved from a proprietary entity-recognition model at Ekstra Bladet are based on the concatenated title, abstract, and body.                                                                               | ['ORG', 'LOC', 'PER']                            | list[str]    |
| 16 Topics          | The tags retrieved from a proprietary topic-recognition model at Ekstra Bladet are based on the concatenated title, abstract, and body.                                                                                | []                                               | list[str]    |
| 17 Total in views  | The total number of times an article has been in view (registered as seen) by users within the first 7 days after it was published. This feature only applies to articles that were published after February 16, 2023. | null                                             | i32          |
| 18 Total pageviews | The total number of times an article has been clicked by users within the first 7 days after it was published. This feature only applies to articles that were published after February 16, 2023.                      | null                                             | i32          | --> filter out null values |
| 19 Total read-time | The accumulated read-time of an article within the first 7 days after it was published. This feature only applies to articles that were published after February 16, 2023.                                             | null                                             | f32          | --> filter out null values |
| 20 Sentiment label | The assigned sentiment label from a proprietary sentiment model at Ekstra Bladet is based on the concatenated title and abstract. The labels are negative, neutral, and positive.                                      | Neutral                                          | str          |
| 21 Sentiment score | The sentiment score from a proprietary sentiment model at Ekstra Bladet is based on the concatenated title and abstract. The score is the corresponding probability to the sentiment label.                            | 0.5299                                           | f32          |

### Head:

```
['article_id', 'title', 'subtitle', 'last_modified_time', 'premium', 'body', 'published_time', 'image_ids', 'article_type', 'url', 'ner_clusters', 'entity_groups', 'topics', 'category', 'subcategory', 'category_str', 'total_inviews', 'total_pageviews', 'total_read_time', 'sentiment_score', 'sentiment_label']
article_id title subtitle last_modified_time ... total_pageviews total_read_time sentiment_score sentiment_label
0 3037230 Ishockey-spiller: Jeg troede jeg skulle dø ISHOCKEY: Ishockey-spilleren Sebastian Harts h... 2023-06-29 06:20:57 ... NaN NaN 0.9752 Negative
1 3044020 Prins Harry tvunget til dna-test Hoffet tvang Prins Harry til at tage dna-test ... 2023-06-29 06:21:16 ... NaN NaN 0.7084 Negative
2 3057622 Rådden kørsel på blå plader Kan ikke straffes: Udenlandske diplomater i Da... 2023-06-29 06:21:24 ... NaN NaN 0.9236 Negative
3 3073151 Mærsk-arvinger i livsfare FANGET I FLODBØLGEN: Skibsrederens oldebørn må... 2023-06-29 06:21:38 ... NaN NaN 0.9945 Negative
4 3193383 Skød svigersøn gennem babydyne 44-årig kvinde tiltalt for drab på ekssvigersø... 2023-06-29 06:22:57 ... NaN NaN 0.9966 Negative
```

## 2. Dataset: behaviors.parquet

Between 25/5/2023 and 1/06/2023
Logs for the 7-day period

### Column overview

| Column                    | Context                                                                                                                                                                                      | Example                          | dtype        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| 1 Impression ID           | The unique ID of an impression.                                                                                                                                                              | 153                              | u32          |
| 2 User ID                 | The anonymized user ID.                                                                                                                                                                      | 44038                            | u32          |
| 3 Article ID              | The unique ID of a news article. An empty field means the impression is from the front page.                                                                                                 | 9650148                          | i32          |
| 4 Session ID              | A unique ID for a user's browsing session.                                                                                                                                                   | 1153                             | u32          |
| 5 In view article IDs     | List of in view article IDs in the impression (news articles that were registered as seen by the user). The order of the IDs have been shuffled.                                             | [9649538, 9649689, ..., 9649569] | list[i32]    |
| 6 Clicked article IDs     | List of article IDs clicked in the impression.                                                                                                                                               | [9649689]                        | list[i32]    |
| 7 Time                    | The impression timestamp. The format is "YYYY/MM/DD HH:MM:SS".                                                                                                                               | 2023-02-25 06:41:40              | datetime[𝜇𝑠] |
| 8 Read-time               | The amount of time, in seconds, a user spends on a given page.                                                                                                                               | 14.0                             | f32          | --> on what page?: On the article_id or homepage (when article_id is empty)                 |
| 9 Scroll percentage       | The percentage of an article that a user scrolls through.                                                                                                                                    | 100.0                            | f32          | -> Of what article: the one of the article_id (scroll and read time are empty for homepage) |
| 10 Device type            | The type of device used to access the content, such as desktop (1) mobile (2), tablet (3), or unknown (0).                                                                                   | 1                                | i8           |
| 11 SSO status             | Indicates whether a user is logged in through Single Sign-On (SSO) authentication.                                                                                                           | True                             | bool         |
| 12 Subscription status    | The user's subscription status indicates whether they are a paid subscriber. Note that the subscription is fixed throughout the period and was set when the dataset was created.             | True                             | bool         | --> So no data on conversion                                                                |
| 13 Gender                 | The user's gender, either male (0) or female (1), as specified in their profile.                                                                                                             | null                             | i8           |
| 14 Postcode               | The user's postcode, aggregated at the district level as specified in their profile, categorized as metropolitan (0), rural district (1), municipality (2), provincial (3), or big city (4). | 2                                | i8           |
| 15 Age                    | The user's age, as specified in their profile, categorized into 10-year bins (e.g., 20-29, 30-39, etc.)                                                                                      | 50                               | i8           |
| 16 Next read-time         | The time a user spends on the next clicked article, i.e., the article in the clicked article IDs.                                                                                            | 8.0                              | f32          | --> Only the first one? Does that link to a new row?                                        |
| 17 Next scroll percentage | The scroll percentage for a user's next article interaction, i.e., the article in clicked article IDs.                                                                                       | 41.0                             | f32          |

### Head:

```
['impression_id', 'article_id', 'impression_time', 'read_time', 'scroll_percentage', 'device_type', 'article_ids_inview', 'article_ids_clicked', 'user_id', 'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'next_read_time', 'next_scroll_percentage']
impression_id article_id impression_time read_time scroll_percentage device_type ... postcode age is_subscriber session_id next_read_time next_scroll_percentage
0 48401 NaN 2023-05-21 21:06:50 21.0 NaN 2 ... NaN NaN False 21 16.0 27.0
1 152513 9778745.0 2023-05-24 07:31:26 30.0 100.0 1 ... NaN NaN False 298 2.0 48.0
2 155390 NaN 2023-05-24 07:30:33 45.0 NaN 1 ... NaN NaN False 401 215.0 100.0
3 214679 NaN 2023-05-23 05:25:40 33.0 NaN 2 ... NaN NaN False 1357 40.0 47.0
4 214681 NaN 2023-05-23 05:31:54 21.0 NaN 2 ... NaN NaN False 1358 5.0 49.0
```

## 3. Dataset: history.parquet

Users’ click histories (Table 8), which contain 21 days of clicked news articles prior to the data split’s impression logs

### Column overview

| Column              | Context                                                                                | Example                                            | dtype              |
| ------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------ |
| 1 User ID           | The anonymized user ID.                                                                | 44038                                              | u32                |
| 2 Article IDs       | The article IDs clicked by the user.                                                   | [9618533, . . . 9646154]                           | list[i32]          |
| 3 Timestamps        | The timestamps of when the articles were clicked. The format is "YYYY/MM/DD HH:MM:SS". | [2023-02-02 16:37:42, . . . , 2023-02-22 18:28:38] | list[datetime[𝜇𝑠]] |
| 4 Read-times        | The read-times of the clicked articles.                                                | [425.0, . . . 12.0]                                | list[f32]          |
| 5 Scroll percentage | The scroll percentage of the clicked articles.                                         | [null, . . . 100.0]                                | list[f32]          |

### Head:

```
['user_id', 'impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed']
user_id impression_time_fixed ... article_id_fixed read_time_fixed
0 13538 [2023-04-27T10:17:43.000000, 2023-04-27T10:18:... ... [9738663, 9738569, 9738663, 9738490, 9738663, ... [17.0, 12.0, 4.0, 5.0, 4.0, 9.0, 5.0, 46.0, 11...
1 58608 [2023-04-27T18:48:09.000000, 2023-04-27T18:48:... ... [9739362, 9739179, 9738567, 9739344, 9739202, ... [2.0, 24.0, 72.0, 65.0, 11.0, 4.0, 101.0, 0.0,...
2 95507 [2023-04-27T15:20:28.000000, 2023-04-27T15:20:... ... [9739035, 9738646, 9634967, 9738902, 9735495, ... [18.0, 29.0, 51.0, 12.0, 10.0, 10.0, 13.0, 24....
3 106588 [2023-04-27T08:29:09.000000, 2023-04-27T08:29:... ... [9738292, 9738216, 9737266, 9737556, 9737657, ... [9.0, 15.0, 42.0, 9.0, 3.0, 58.0, 26.0, 214.0,...
4 617963 [2023-04-27T14:42:25.000000, 2023-04-27T14:43:... ... [9739035, 9739088, 9738902, 9738968, 9738760, ... [45.0, 29.0, 116.0, 26.0, 34.0, 42.0, 58.0, 59...
```

# adressa Dataset format

JSON file with user and article interactions.
Example:

```JSON
[{"eventId": 672342954, "city": "oslo", "activeTime": 8, "url": "http://adressa.no", "referrerHostClass": "direct", "region": "oslo", "time": 1483225202, "userId": "cx:13116981155471447840854:z9xm0v0hh3qy", "sessionStart": false, "deviceType": "Mobile", "sessionStop": false, "country": "no", "os": "iPhone OS"},
{"profile": [{"item": "0", "groups": [{"count": 1, "group": "adressa-importance", "weight": 1.0}]}, {"item": "adressa", "groups": [{"count": 1, "group": "concept", "weight": 0.6796875}]}, {"item": "adressa.no", "groups": [{"count": 1, "group": "site", "weight": 1.0}]}, {"item": "adresseavisen", "groups": [{"count": 1, "group": "entity", "weight": 0.578125}]}, {"item": "article", "groups": [{"count": 1, "group": "pageclass", "weight": 1.0}]}, {"item": "bildet", "groups": [{"count": 1, "group": "concept", "weight": 0.78125}]}, {"item": "camilla kilnes", "groups": [{"count": 1, "group": "author", "weight": 1.0}]}, {"item": "espen rasmussen", "groups": [{"count": 1, "group": "author", "weight": 1.0}]}, {"item": "free", "groups": [{"count": 1, "group": "adressa-access", "weight": 1.0}]}, {"item": "hobbies-and-interests", "groups": [{"count": 1, "group": "classification", "weight": 0.48828125}]}, {"item": "lesernes nytt\u00e5rsbilder", "groups": [{"count": 1, "group": "concept", "weight": 1.0}]}, {"item": "negative", "groups": [{"count": 1, "group": "sentiment", "weight": 1.0}]}, {"item": "no", "groups": [{"count": 1, "group": "language", "weight": 1.0}]}, {"item": "nyheter", "groups": [{"count": 1, "group": "category", "weight": 0.1171875}, {"count": 1, "group": "taxonomy", "weight": 0.125}]}], "activeTime": 79, "canonicalUrl": "http://www.adressa.no/nyheter/2016/12/31/Se-lesernes-nytt%c3%a5rsbilder-14000400.ece", "referrerHostClass": "search", "sessionStop": false, "userId": "cx:2fs9x8i7jvcjyckoxqfa6l4lw:3rr1gvpcbzx8w", "publishtime": "2016-12-31T17:13:57.000Z", "sessionStart": false, "referrerUrl": "http://adressa.no", "keywords": "utenriks,innenriks,trondheim,E6,midtbyen,bybrann,bilulykker", "id": "9f3999bd1a1a8d67bcb073ad54840f15cb30f014", "eventId": 335855522, "city": "trondheim", "title": "Se lesernes nytt\u00e5rsbilder", "url": "http://adressa.no/nyheter/2016/12/31/se-lesernes-nytt%c3%a5rsbilder-14000400.ece", "country": "no", "region": "sor-trondelag", "author": ["camilla kilnes", "espen rasmussen"], "referrerSearchEngine": "Internal", "deviceType": "Tablet", "time": 1483225202, "os": "Android"}]
```
