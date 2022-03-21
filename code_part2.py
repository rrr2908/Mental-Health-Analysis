import tweepy
import emoji
# TextBlob - Python library for processing textual data
from textblob import TextBlob

# WordCloud - Python linrary for creating image wordclouds
from wordcloud import WordCloud

# Pandas - Data manipulation and analysis library
import pandas as pd

# NumPy - mathematical functions on multi-dimensional arrays and matrices
import numpy as np

# Regular Expression Python module
import re

# Matplotlib - plotting library to create graphs and charts
import matplotlib.pyplot as plt

# Settings for Matplotlib graphs and charts
from pylab import rcParams, text, scatter, show
rcParams['figure.figsize'] = 12, 8

#extract tweets and display
tweets = pd.read_csv('new_BoldMonk__tweets.csv')
tweets.head(20)

#clean the tweets
def cleanUpTweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', str(txt))
    # Remove hashtags
    txt = re.sub(r'#', '', str(txt))
    # Remove retweets:
    txt = re.sub(r'RT : ', '', str(txt))
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', str(txt))
    #txt = re.sub(r' ', '', txt)
    return txt
    # Remove emojis
def remove_emoji(txt):
        
    regex_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return regex_pattern.sub(r'', txt)
tweets['message'] = tweets['message'].apply(cleanUpTweet)
tweets['message'] = tweets['message'].apply(remove_emoji)
tweets.head(20)

# feature extraction
def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity'

#display with feature rating
tweets.head(20)

# classify output variables
# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

# apply on polarity of tweets
tweets['Score'] = tweets['Polarity'].apply(getTextAnalysis)
#display scores
tweets.head(20)

#generate positive,negative and neutral wordclouds
positive_words = ' '.join(list(tweets[tweets['Score'] == 'Positive']['message']))
positive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(positive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(positive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

depressive_words = ' '.join(list(tweets[tweets['Score'] == 'Negative']['message']))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(depressive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

neutral_words = ' '.join(list(tweets[tweets['Score'] == 'Neutral']['message']))
neutral_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(neutral_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(neutral_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

# plot output variables as bar graph
labels = tweets.groupby('Score').count().index.values

values = tweets.groupby('Score').size().values

plt.bar(labels, values)

#display % of tweets
positive = tweets[tweets['Score'] == 'Positive']

print(str(positive.shape[0]/(tweets.shape[0])*100) + " % of positive tweets")

