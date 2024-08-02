# Twitter Bot Detection

## What is this Project about

In an Undergrad Research Project, we were tasked with creating a program to find bots injected in a large dataset of tweets by our peers.

## What did we do

To find the bots, we implemented many features, such as clustering based on cosine similarity, removing profanity, detecting emojis and checking grammar mistakes. After a long time of deliberating and brainstorming with my team, we found that those features were the ones that stood out the most between bots and real users. 

## What results did we find

For the second dataset, we had the following results:
  We have found: 58 bots\n
  There are 64 users in the users_df
  90.625% of the users in users_df are bots
  89.23076923076924% bots have been found

For the first dataset, we had the following results:
  We have found: 2 bots
  There are 2 users in the users_df
  100% of the users in users_df are bots
  100% bots have been found
