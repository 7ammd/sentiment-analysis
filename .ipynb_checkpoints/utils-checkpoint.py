import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import twitter_samples
def process_tweet(tweet):
    """Process tweet function: cleans, tokenizes, removes stopwords, and stems."""
    tweet2 = re.sub(r'^RT[\s]+', '', tweet)
    tweet2 = re.sub(r'https?://[^\s\t\r\n]+', '', tweet2)
    tweet2 = re.sub(r'#', '', tweet2)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet2)

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
            
    return tweets_clean

def build_freqs(tweets, ys):
    """Build frequencies dictionary mapping (word, label) to count."""
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs

def load_tweets():
    """Loads positive and negative tweets using NLTK twitter_samples."""
    # Download the twitter samples dataset if not already downloaded
    nltk.download("twitter_samples", quiet=True)

    # Load positive and negative tweets from NLTK dataset
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")

    return all_positive_tweets, all_negative_tweets    