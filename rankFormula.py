def classify_ranking(tweet):
    if tweet['retweet_count'] > 0:
        return 5
    elif tweet['favorite_count'] > 0:
        return 4
    elif tweet['favorite_count'] == 0 and tweet['retweet_count'] == 0:
        return 1
    else:
        return 2

# Apply the function to the tweets
tweets['ranking'] = tweets.apply(classify_ranking, axis=1)

# Print the top 5 most positive tweets
print(tweets.nlargest(5, 'ranking'))
