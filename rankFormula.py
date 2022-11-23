# Create a new column called 'ranking'
df['ranking'] = df['retweet_count'] + df['favorite_count'] + df['number_coments'] + df['number_retweets_with_comments']


# Sort the dataframe by ranking
df.sort_values(by='ranking', ascending=False)

# Create a new column called 'ranking_class'
df['ranking_class'] = df['ranking'].apply(lambda x: 1 if x < 100 else 2 if x < 1000 else 3 if x < 10000 else 4 if x < 100000 else 5)
