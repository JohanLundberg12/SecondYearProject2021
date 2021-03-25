import numpy as np
import string
from nltk.corpus import stopwords
stopwords = stopwords.words('english')+list(string.punctuation)

def remove_stopwords(text):
    '''
    Input: A list of strings.
    Output: A list of strings.
    '''
    return [word for word in text if word not in stopwords]


def tokenize(text):
    '''
    Input: 
        - A string.
    Output:
        - A list of tokens.
    '''

    return text.split(" ")


def preprocessor(text, settings=None):
    '''
    Input: A string.
    Output: A list of tokens.
    '''
    text = tokenize(text)

    if settings['remove_stopwords']:
        text = remove_stopwords(text)

    return text


def transform(df, settings, preprocessor=preprocessor):
    '''A series of transformations on the df.
    Input: 
        - df: DataFrame.
        - settings: dictionary.
        - preprocessor: function.
    Output: X: List of lists of tokens, y: list.
    '''
    df = df.replace(np.nan, '', regex=True)

    if settings['include_summary']:
        df['reviewText'] = df.reviewText + ' ' + df.summary
    y = df.sentiment.tolist()
    data = [(preprocessor(string, settings), y) for string, y in zip(df['reviewText'].values, y)]
    X, y = zip(*data)

    return X, y
