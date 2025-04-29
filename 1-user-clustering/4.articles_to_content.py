"""
This script reads an in/articles.parquet file and creates embeddings for each article by taking the category_str and title
and concatenating them with a separator.

It then uses the SentenceTransformer to create embeddings for the concatenated strings.

The embeddings are saved to a new file called out/articles_embeddings.parquet
"""

import pandas as pd
articles = pd.read_parquet('./in/articles.parquet')

# Prepare the list of texts to encode
texts_to_encode = [
    'query: ' + str(row['category_str']) + ': ' + str(row['title'])
    for index, row in articles.iterrows()
]

# Write a csv: articles_content.csv with columns article_id, content (content is the corresponding texts_to_encode)
# Create DataFrame with article_id and corresponding text content
articles_content = pd.DataFrame({
    'article_id': articles['article_id'],
    'content': texts_to_encode
})

# Save to CSV
articles_content.to_csv('./out/articles_content.csv', index=False)


# # Create embeddings in batch
# # Load the SentenceTransformer model
# model = SentenceTransformer('intfloat/multilingual-e5-large')
# # The result is typically a NumPy array or list of arrays. Convert to list for Parquet compatibility.
# embeddings = model.encode(
#     texts_to_encode, normalize_embeddings=True, convert_to_numpy=True).tolist()

# # Print the first 10 embeddings
# print(embeddings[:10])

# # Assign embeddings to the DataFrame column
# articles['embeddings'] = embeddings

# # Save the embeddings to a parquet file with the columns: article_id, title, category_str, embeddings
# articles[['article_id', 'title', 'category_str', 'embeddings']].to_parquet(
#     './out/articles_embeddings.parquet', index=False)
