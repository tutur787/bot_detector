import pandas as pd
import data_modification as dm
import json

def test_remove_retweets_likes_ID():
    fname = 'data/twitter_dataset.json'
    tweets = dm.remove_retweets_likes(fname)

def main():
    test_remove_retweets_likes_ID()

if __name__ == '__main__':
    main()