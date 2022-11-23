import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('tweets.csv')

df.head()

df.info()

df.describe()

df['text length'] = df['text'].apply(len)

df.head()

g = sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text length')

sns.boxplot(x='stars',y='text length',data=df,palette='rainbow')


sns.countplot(x='stars',data=df,palette='rainbow')

stars = df.groupby('stars').mean()
stars

stars.corr()

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


tweet_class = df[(df.ranking==1) | (df.ranking==5)]

X = tweet_class['text']
y = tweet_class['ranking']


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()

X = cv.fit_transform(X)


from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


from sklearn.naive_bayes import MultinomialNB


nb = MultinomialNB()

nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



from sklearn.feature_extraction.text import  TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # Strings to token integer counts
    ('tfidf', TfidfTransformer()),  # Integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # Train on TF-IDF vectors w/ Naive Bayes classifier
])


X = tweet_class['text']
y = tweet_class['ranking']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

# Waiting...

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))