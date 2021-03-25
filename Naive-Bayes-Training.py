from sklearn.model_selection import train_test_split

from helper_functions import bag_of_features
from Load import json_to_df
from models.NaiveBayes import NaiveBayes
from preprocess import transform

path = 'data/music_reviews_train.json'
path_test = 'data/music_reviews_dev.json'
df = json_to_df(path)

settings = {'include_summary': False, 'remove_stopwords': True,
            'unigram_freq': True, 'unigram_pres': False}

X, y = transform(df, settings)
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
train_features = [bag_of_features(sentence, settings) for sentence in X_train]
dev_features = [bag_of_features(sentence, settings) for sentence in X_dev]

NB = NaiveBayes()
w = NB.fit(train_features, y_train)
preds = NB.predict(dev_features, w)
print(NB.evaluate(y_dev, preds))

df_test = json_to_df(path_test)
X_test, y_test = transform(df_test, settings)
test_features = [bag_of_features(sentence, settings) for sentence in X_test]
preds = NB.predict(test_features, w)
<<<<<<< HEAD
print(NB.evaluate(y_test, preds))
=======
print(NB.evaluate(y_dev, preds))
>>>>>>> 7e16777b75236e1a5c4c7dd6d79b348c6510feac
