from sklearn.model_selection import train_test_split

from helper_functions import bag_of_features
from Load import json_to_df
from models.NaiveBayes import NaiveBayes
from preprocess import transform

path = 'data/music_reviews_train.json'
df = json_to_df(path)

settings = {'include_summary':True, 'remove_stopwords': True, 'unigram_freq':False, 'unigram_pres':True}
X, y = transform(df, settings)

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)

train_features = [bag_of_features(sentence, settings) for sentence in X_train]

test_features = [bag_of_features(sentence, settings) for sentence in X_dev]


NB = NaiveBayes()
w = NB.fit(train_features, y_train)
preds = NB.predict(test_features, w)
print(NB.evaluate(y_dev, preds))
