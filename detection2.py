### IMPORT LIBRARIES

import pandas as pd
import numpy as np
import re
import json
from nltk.tokenize import word_tokenize
from profanity_check import predict_prob
from sklearn.cluster import KMeans
import emoji
import language_tool_python
import nltk
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

### SET SEED

np.random.seed(123)

### LOAD DATA

def load_data(tweet_path, user_path):
    with open(tweet_path, 'r') as file:
        data = json.load(file)

    # The data is loaded into a pandas dataframe
    tweet_df = pd.DataFrame(data)
    og_tweet_df = pd.DataFrame(data)

    with open(user_path, 'r') as file:
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
    kmeans = KMeans(n_clusters=10).fit(X)
    user_tweets['cluster'] = kmeans.labels_

    # Keep the users in the two clusters that contain the most users
    selected_clusters = user_tweets['cluster'].value_counts().nlargest(4).index
    selected_users = user_tweets[user_tweets['cluster'].isin(selected_clusters)]['author_id']

    # Filter the tweet_df and users_df DataFrames to keep only the selected users
    tweet_df = tweet_df[tweet_df['author_id'].isin(selected_users)]
    users_df = users_df[users_df['id'].isin(selected_users)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after cluster filtering: {len(tweet_df)}")
    print(f"User dataset length after cluster filtering: {len(users_df)}")

    return tweet_df, users_df

### GRAMMAR CHECK

def grammar_check(tweet_df, users_df):
    # Initialize LanguageTool
    tool = language_tool_python.LanguageTool('en-US')

    # Function to clean the text (remove URLs and emojis)
    def clean_text(text):
        # Remove URLs (both http and https)
        text = re.sub(r'https?:\S+', '', text)
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

    # Calculate error rate (errors per tweet)
    tweet_df['error_rate'] = tweet_df['error_count'] / tweet_df['text'].str.split().apply(len)

    # Sum error rates for each user
    user_error_rates = tweet_df.groupby('author_id')['error_rate'].sum().reset_index()

    # Identify users with an average error rate higher than a threshold (e.g., 0.2 errors per word)
    users_with_excessive_errors = user_error_rates[user_error_rates['error_rate'] >= 0.9]['author_id'].tolist()

    # Filter out these users
    tweet_df = tweet_df[~tweet_df['author_id'].isin(users_with_excessive_errors)]
    users_df = users_df[~users_df['id'].isin(users_with_excessive_errors)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after grammar check: {len(tweet_df)}")
    print(f"User dataset length after grammar check: {len(users_df)}")

    return tweet_df, users_df

def cluster_descriptions(tweets_df, users_df):
    # Preprocess descriptions
    def preprocess_description(text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, hashtags, mentions, and non-alphanumeric characters
        text = re.sub(r'http\S+|#[A-Za-z0-9_]+|@[A-Za-z0-9_]+|\W', ' ', text)
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    users_df['processed_description'] = users_df['description'].apply(preprocess_description)

    # Vectorize descriptions
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(users_df['processed_description'])

    # Apply clustering (you can tune n_clusters based on your dataset)
    kmeans = KMeans(n_clusters=10, random_state=42)
    users_df['cluster'] = kmeans.fit_predict(X)

    # Keep the users in the two clusters that contain the most users
    selected_clusters = users_df['cluster'].value_counts().nlargest(4).index
    selected_users = users_df[users_df['cluster'].isin(selected_clusters)]['id']

    # Filter the tweet_df and users_df DataFrames to keep only the selected users
    tweet_df = tweet_df[tweet_df['author_id'].isin(selected_users)]
    users_df = users_df[users_df['id'].isin(selected_users)]

    # Print the length of the dataset after filtering
    print(f"Tweet dataset length after cluster filtering: {len(tweet_df)}")
    print(f"User dataset length after cluster filtering: {len(users_df)}")

    return users_df

### MAIN FUNCTION
    
def main():
    '''# Load diff.json to a dataframe and check which users are in users_df
    with open('diff.json', 'r') as file:
        data = json.load(file)
    
    diff_df = pd.DataFrame(data)'''

    # Use argparse to get the tweet_path and user_path
    parser = argparse.ArgumentParser()
    parser.add_argument('tweet_path', type=str, help='Path to the tweet dataset')
    parser.add_argument('user_path', type=str, help='Path to the user dataset')
    args = parser.parse_args()
    tweet_path = args.tweet_path
    user_path = args.user_path

    tweet_df, users_df, og_tweet_df = load_data(tweet_path, user_path)

    tweet_df, users_df = cluster_cosine(tweet_df, users_df)
    #post_cluster_bots = len(diff_df[diff_df['id'].isin(users_df['id'])])

    tweet_df, users_df = remove_profanity(tweet_df, users_df)
    #post_prof_bots = len(diff_df[diff_df['id'].isin(users_df['id'])])

    tweet_df, users_df = grammar_check(tweet_df, users_df)
    #post_grammar_bots = len(diff_df[diff_df['id'].isin(users_df['id'])])

    tweet_df, users_df = cluster_descriptions(tweet_df, users_df)

    print(users_df['id'])

    # Save the users_df id column to a text file
    users_df['id'].to_csv('filtered_users_dataset2.txt', index=False)

    '''print(f"Cluster removes: {len(diff_df) - post_cluster_bots}, Profanity removes: {post_cluster_bots - post_prof_bots}, Grammar removes: {post_prof_bots - post_grammar_bots}")
    # Only print the users that are in users_df
    #print(diff_df[diff_df['id'].isin(users_df['id'])])
    print(f"We have found: {len(diff_df[diff_df['id'].isin(users_df['id'])])} bots")
    print(f"There are {len(users_df)} users in the users_df")
    print(f"{len(diff_df[diff_df['id'].isin(users_df['id'])])/len(users_df) * 100}% of the users in users_df are bots")
    print(f"{len(diff_df[diff_df['id'].isin(users_df['id'])]) / len(diff_df) * 100}% bots have been found")'''

if __name__ == "__main__":
    main()