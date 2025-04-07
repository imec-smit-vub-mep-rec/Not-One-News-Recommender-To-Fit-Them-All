import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def get_time_of_day(hour):
    if 0 <= hour < 6:
        return "Night (0-6)"
    elif 6 <= hour < 12:
        return "Morning (6-12)"
    elif 12 <= hour < 18:
        return "Afternoon (12-18)"
    else:
        return "Evening (18-24)"

# Read the parquet file
df = pd.read_parquet('datasets/ekstra/articles.parquet')

# Extract hour from the publication timestamp and categorize
df['hour'] = pd.to_datetime(df['published_time']).dt.hour
df['time_of_day'] = df['hour'].apply(get_time_of_day)

# Count articles by time of day
article_counts = df['time_of_day'].value_counts().sort_index()

# Display results
print("\nArticle Publication Distribution by Time of Day:")
print("=" * 45)
for time_of_day, count in article_counts.items():
    print(f"{time_of_day}: {count} articles")

# Calculate percentages
percentages = (article_counts / len(df) * 100).round(1)
print("\nPercentage Distribution:")
print("=" * 45)
for time_of_day, percentage in percentages.items():
    print(f"{time_of_day}: {percentage}%")

# Create proportional distribution of article types per time of day
article_type_props = pd.crosstab(
    df['time_of_day'], 
    df['article_type'], 
    normalize='index'
) * 100

# Create stacked bar chart
ax = article_type_props.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    rot=0
)

plt.title('Article Type Distribution by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Percentage')
plt.legend(title='Article Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('article_type_distribution.png')


# Print the numerical percentages
print("\nDetailed Article Type Distribution by Time of Day:")
print("=" * 45)
print(article_type_props.round(1))
