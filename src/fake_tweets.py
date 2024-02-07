from faker import Faker
import random
import datetime

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