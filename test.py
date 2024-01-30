import requests
from datetime import datetime, timedelta
import json

def search_tweets(hashtag, days=2, bearer_token="AAAAAAAAAAAAAAAAAAAAAPyksAEAAAAA7ZICh7F6AzggryY1YgimVzOsUxg%3D5UnNBc85E68iq1SG6ERrH9ja8IeKsn8kJKNllxreBkuM299JEd"):
    """
    Search for tweets containing the specified hashtag within the past 'days' days.

    Parameters:
    - hashtag: The hashtag to search for (e.g., "#taylorswift").
    - days: The number of past days to search within.
    - bearer_token: Your Twitter API Bearer Token.

    Returns:
    A list of tweets.
    """
    # Format the search query
    query = f"{hashtag} -is:retweet"
    
    # Calculate the start time for the search query
    start_time = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Twitter API v2 endpoint for recent search
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    
    # Parameters for the API request
    params = {
        "query": query,
        "start_time": start_time,
        "max_results": 100,  # Adjust as needed, up to 100
        "tweet.fields": "created_at,author_id,text"
    }
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    
    response = requests.get(search_url, headers=headers, params=params)
    
    if response.status_code == 200:
        # Parse the response JSON and return the data
        tweets = response.json().get("data", [])
        return tweets
    else:
        raise Exception(f"Failed to fetch tweets: {response.status_code} {response.text}")

def save_tweets_to_json(tweets, filename="tweets.json"):
    """
    Save the list of tweets to a JSON file.

    Parameters:
    - tweets: The list of tweets to save.
    - filename: The name of the file to save the tweets to.
    """
    with open(filename, "w") as file:
        json.dump(tweets, file, ensure_ascii=False, indent=4)

# Example usage
def main():
    bearer_token = "YOUR_BEARER_TOKEN_HERE"  # Replace with your Bearer Token
    tweets = search_tweets("#taylorswift", bearer_token=bearer_token)
    save_tweets_to_json(tweets)
