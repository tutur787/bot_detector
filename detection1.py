### IMPORT LIBRARIES

import pandas as pd
import numpy as np
import re
import json
from nltk.tokenize import word_tokenize
from profanity_check import predict_prob
from textblob import TextBlob
from sklearn.cluster import KMeans
from collections import Counter
import emoji
import language_tool_python
import nltk
nltk.download('punkt')
import argparse

### SET SEED

np.random.seed(123)

### LOAD DATA

def load_data():
    with open('dataset.english.2024-03-22.augmented.B2.24.json', 'r') as file:
        data = json.load(file)

    # The data is loaded into a pandas dataframe
    tweet_df = pd.DataFrame(data)
    og_tweet_df = pd.DataFrame(data)

    with open('dataset.english.2024-03-22.users.augmented.B2.2.json', 'r') as file:
        data = json.load(file)

    # The data is loaded into a pandas dataframe
    users_df = pd.DataFrame(data)

    # Remove tweets where text contains 'Bokep Porn'
    tweet_df = tweet_df[~tweet_df['text'].str.contains('Bokep Porn')]
    users_df = users_df[users_df['id'].isin(tweet_df['author_id'])]

    # Print the length of the dataset
    print(f"Tweet dataset length: {len(tweet_df)}")

    # Print the length of the dataset
    print(f"User dataset length: {len(users_df)}")

    return tweet_df, users_df, og_tweet_df

### REMOVING PROFANITY USERS

def remove_profanity(tweet_df, users_df):

    # Predict the probability of profanity in each tweet
    tweet_df['profanity_prob'] = tweet_df['text'].apply(lambda x: predict_prob([x])[0])

    # Identifying users with any tweet having a profanity probability greater than 0.5
    users_with_high_profanity = tweet_df[tweet_df['profanity_prob'] >= 0.55]['author_id'].unique()

    # Filtering out all tweets from these users
    tweet_df = tweet_df[~tweet_df['author_id'].isin(users_with_high_profanity)]

    # Remove users_with_high_profanity from users_df
    users_df = users_df[~users_df['id'].isin(users_with_high_profanity)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after removing tweets with high probability of profanity: {len(tweet_df)}")
    print(f"User dataset length after removing users with high probability of profanity: {len(users_df)}")

    return tweet_df, users_df

### REMOVING USERS WITH LOW POLARITY

def remove_low_polarity(tweet_df, users_df):
    # Measure the polarity of the tweets
    tweet_df['polarity'] = tweet_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Compute average polarity for each user
    average_polarity_by_user = tweet_df.groupby('author_id')['polarity'].mean()

    # Identify users with low average polarity
    users_with_low_average_polarity = average_polarity_by_user[(average_polarity_by_user > 0.05) & (average_polarity_by_user < 0.2)].index

    # Filter the tweet_df to keep only tweets from these users
    tweet_df = tweet_df[tweet_df['author_id'].isin(users_with_low_average_polarity)]

    # Filter the users_df to keep only users with low average polarity
    users_df = users_df[users_df['id'].isin(users_with_low_average_polarity)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after removing users with high average polarity: {len(tweet_df)}")
    print(f"User dataset length after removing users with high average polarity: {len(users_df)}")

    return tweet_df, users_df

### CLUSTERING & COSINE SIMILARITY
    
def cluster_cosine(tweet_df, users_df):
    # Assuming tweet_df is your DataFrame containing 'author_id' and 'text'

    # Function to vectorize text using the GloVe embeddings
    def vectorize_text(text, glove_model):
        words = word_tokenize(text.lower())
        word_vectors = [glove_model.get(word, np.zeros(100)) for word in words]  # Handling out-of-vocabulary words
        return np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(100)

    def load_glove_model(glove_file):
        model = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding
        return model

    # Load the GloVe model
    glove_model = load_glove_model('glove.twitter.27B.100d.txt')  # Adjust the file path as needed

    # Aggregate tweets by user
    user_tweets = tweet_df.groupby('author_id')['text'].apply(' '.join).reset_index()

    # Vectorize the aggregated tweets
    user_tweets['vector'] = user_tweets['text'].apply(lambda x: vectorize_text(x, glove_model))

    # Clustering users based on tweet similarity
    X = np.array(user_tweets['vector'].tolist())
    kmeans = KMeans(n_clusters=5).fit(X)
    user_tweets['cluster'] = kmeans.labels_

    # Keep the users in the two clusters that contain the most users
    selected_clusters = user_tweets['cluster'].value_counts().nlargest(2).index
    selected_users = user_tweets[user_tweets['cluster'].isin(selected_clusters)]['author_id']

    # Filter the tweet_df and users_df DataFrames to keep only the selected users
    tweet_df = tweet_df[tweet_df['author_id'].isin(selected_users)]
    users_df = users_df[users_df['id'].isin(selected_users)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after cluster filtering: {len(tweet_df)}")
    print(f"User dataset length after cluster filtering: {len(users_df)}")

    return tweet_df, users_df

### EMOJI DETECTION
    
def emoji_detection(tweet_df, users_df, og_tweet_df):
    # Function to extract emojis from a text
    def extract_emojis(text):
        return [char['emoji'] for char in emoji.emoji_list(text)]

    # Count emoji frequency in the original dataset
    emoji_counts = Counter(emoji for tweet in og_tweet_df['text'] for emoji in extract_emojis(tweet))

    # Determine uncommon emojis (bottom 75%)
    threshold_index = int(len(emoji_counts) * 0.75)
    uncommon_emojis = set(sorted(emoji_counts, key=emoji_counts.get)[:threshold_index])

    # Analyze emoji usage in the reduced dataset
    def calculate_emoji_metrics(user_tweets, uncommon_emojis):
        user_emoji_count = Counter(emoji for tweet in user_tweets for emoji in extract_emojis(tweet))
        total_emojis = sum(user_emoji_count.values())
        user_uncommon_emoji_count = sum(user_emoji_count[emoji] for emoji in uncommon_emojis)
        uncommon_emoji_percentage = user_uncommon_emoji_count / total_emojis if total_emojis > 0 else 0
        return total_emojis * uncommon_emoji_percentage  # Relative metric

    # Apply the analysis to the reduced dataset
    tweet_df['emoji_metric'] = tweet_df.groupby('author_id')['text'].transform(lambda x: calculate_emoji_metrics(x, uncommon_emojis))

    # Now, sort the users by the 'emoji_metric'
    sorted_users = tweet_df.drop_duplicates('author_id').sort_values(by='emoji_metric', ascending=False)

    # Only keep those that have above 5 emoji_metric
    selected_users = sorted_users[sorted_users['emoji_metric'] > 5]['author_id']

    # Filter the tweet_df and users_df DataFrames to keep only the selected users
    tweet_df = tweet_df[tweet_df['author_id'].isin(selected_users)]
    users_df = users_df[users_df['id'].isin(selected_users)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after emoji filtering: {len(tweet_df)}")
    print(f"User dataset length after emoji filtering: {len(users_df)}")

    return tweet_df, users_df

### GRAMMAR CHECK

def grammar_check(tweet_df, users_df):
    # Initialize LanguageTool
    tool = language_tool_python.LanguageTool('en-US')

    # Function to clean the text (remove URLs and emojis)
    def clean_text(text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove emojis
        emojis = {char['emoji'] for char in emoji.emoji_list(text)}
        for em in emojis:
            text = text.replace(em, '')
        return text

    # Function to count grammatical errors in a tweet
    def count_errors(text):
        cleaned_text = clean_text(text)
        matches = tool.check(cleaned_text)
        return len(matches)

    # Count errors for each tweet
    tweet_df['error_count'] = tweet_df['text'].apply(count_errors)

    # Sum errors for each user
    user_error_counts = tweet_df.groupby('author_id')['error_count'].sum().reset_index()

    # Identify users with more than 15 total errors
    users_with_excessive_errors = user_error_counts[user_error_counts['error_count'] > 15]['author_id'].tolist()

    # Filter out these users
    tweet_df = tweet_df[~tweet_df['author_id'].isin(users_with_excessive_errors)]
    users_df = users_df[~users_df['id'].isin(users_with_excessive_errors)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after grammar check: {len(tweet_df)}")
    print(f"User dataset length after grammar check: {len(users_df)}")

    return tweet_df, users_df

### MAIN FUNCTION
    
def main():
    tweet_df, users_df, og_tweet_df = load_data()
    #tweet_df, users_df = remove_low_polarity(tweet_df, users_df)
    tweet_df, users_df = cluster_cosine(tweet_df, users_df)
    tweet_df, users_df = remove_profanity(tweet_df, users_df)
    tweet_df, users_df = emoji_detection(tweet_df, users_df, og_tweet_df)
    tweet_df, users_df = grammar_check(tweet_df, users_df)

    print(users_df['id'])

    # Save the users_df id column to a text file
    users_df['id'].to_csv('filtered_users_dataset1.txt', index=False)

if __name__ == "__main__":
    main()