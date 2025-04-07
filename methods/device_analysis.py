import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the behaviors data
df = pd.read_parquet('datasets/ekstra/behaviors.parquet')

# Create a device type mapping for better readability
device_mapping = {
    0: 'Unknown',
    1: 'Desktop',
    2: 'Mobile',
    3: 'Tablet'
}

# Convert device types
df['device_type'] = df['device_type'].map(device_mapping)

# Group by session_id and calculate metrics
session_metrics = df.groupby(['session_id', 'device_type']).agg({
    'article_id': lambda x: x.notna().sum(),  # Count actual article views (not homepage)
    'read_time': 'mean',
    'scroll_percentage': 'mean'
}).reset_index()

# Calculate bounces (sessions with no article views)
bounces = session_metrics[session_metrics['article_id'] == 0].groupby('device_type').size()
total_sessions = session_metrics.groupby('device_type').size()
bounce_rates = (bounces / total_sessions * 100).round(2)

# Calculate average metrics per device
device_metrics = session_metrics.groupby('device_type').agg({
    'article_id': 'mean',
    'read_time': 'mean',
    'scroll_percentage': 'mean'
}).round(2)

# Set the style to a built-in matplotlib style
plt.style.use('bmh')  # Using 'bmh' style instead of 'seaborn'

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Device-Specific User Behavior Analysis', fontsize=16, y=0.95)

# 1. Average Articles per Session
sns.barplot(data=device_metrics.reset_index(), 
           x='device_type', y='article_id',
           ax=axes[0,0])
axes[0,0].set_title('Average Articles Viewed per Session')
axes[0,0].set_ylabel('Number of Articles')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Average Read Time
sns.barplot(data=device_metrics.reset_index(),
           x='device_type', y='read_time',
           ax=axes[0,1])
axes[0,1].set_title('Average Read Time per Article')
axes[0,1].set_ylabel('Seconds')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Average Scroll Depth
sns.barplot(data=device_metrics.reset_index(),
           x='device_type', y='scroll_percentage',
           ax=axes[1,0])
axes[1,0].set_title('Average Scroll Depth')
axes[1,0].set_ylabel('Percentage')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Bounce Rates
sns.barplot(x=bounce_rates.index, y=bounce_rates.values,
           ax=axes[1,1])
axes[1,1].set_title('Bounce Rate by Device')
axes[1,1].set_ylabel('Percentage')
axes[1,1].tick_params(axis='x', rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig('device_analysis.png')

# Print detailed metrics
print("\nDetailed Metrics by Device:")
print("-" * 50)
print("\nAverage Articles per Session:")
print(device_metrics['article_id'])
print("\nAverage Read Time (seconds):")
print(device_metrics['read_time'])
print("\nAverage Scroll Depth (%):")
print(device_metrics['scroll_percentage'])
print("\nBounce Rates (%):")
print(bounce_rates)
