
from faker import Faker
import json
import random

# Define a function to generate random tweets
def generate_random_tweets(n):
    # Initialize Faker to generate random user data
    fake = Faker()
    tweets = []
    for _ in range(n):
        username = fake.user_name()
        text = fake.text(max_nb_chars=280)
        timestamp = fake.date_time_between(start_date="-2y", end_date="now").strftime('%Y-%m-%d %H:%M:%S')
        tweet = {
            "Username": username,
            "Text": text,
            "Timestamp": timestamp
        }
        tweets.append(tweet)
    return tweets


def add_tweet(tweet_id, username, text, retweets, likes, timestamp):
    with open('/Users/annazaidi/Desktop/bot_detector/data/twitter_dataset_2.json', 'r') as file:
        data = json.load(file)
        data.append({
            'Tweet_ID': tweet_id,
            'Username': username,
            'Text': text,
            'Retweets': retweets,
            'Likes': likes,
            'Timestamp': timestamp
        })
        random.shuffle(data)
    with open('tweets.json', 'w') as file:
        json.dump(data, file)

#generate 10 tweeets and add them to /Users/annazaidi/Desktop/bot_detector/data/twitter_dataset_2.json in random places
tweets = generate_random_tweets(10)
for tweet in tweets:
    add_tweet(tweet['Username'], tweet['Text'], tweet['Timestamp'], 0, 0, tweet['Timestamp'])
