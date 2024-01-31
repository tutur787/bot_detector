#remove retweets and likes from the 'twitter_dataset.json' file
#and save the changes to 'twitter_dataset_2.json'
#don't set them to 0 but completely remove them from the file

import json

def remove_retweets_likes():
    #open the file
    with open('data/twitter_dataset.json', 'r') as file:
        data = json.load(file)
        #remove retweets and likes
        for tweet in data:
            tweet.pop('Retweets', None)
            tweet.pop('Likes', None)
    #write the changes to the file
    with open('data/twitter_dataset_2.json', 'w') as file:
        json.dump(data, file, indent=4)

def main():
    remove_retweets_likes()

if __name__ == '__main__':
    main()