import numpy as np
import pandas as pd
import pickle

def clean_nan(df):
    df.replace(np.nan, '', regex=True, inplace=True)

    return df


def join_columns(df, col1, col2):

    df[col1] = df[col1] + ' ' + df[col2]

    return df


def encode_labels(df):
    df.replace({"sentiment":{"positive":1,"negative":0}}, inplace=True)

    return df


def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def score_model(model, df):
    return model.score(df['review_body'], df['sentiment'])


def casetype(df, casetype):
    return df.casetype == casetype


def calc_text_len(df):
    df["textlen"] = [len(text) for text in df.review_body]

    return df


def get_correct_incorrect_reviews(df, preds, actual):
    correct = {'review_body':[], 'casetype':[], 'textlen':[]}
    incorrect = {'review_body':[], 'casetype':[], 'textlen':[]}
    
    for idx, (pred, actual) in enumerate(zip(preds, actual)):
        if pred == actual:
            correct['review_body'].append(df['review_body'][idx])
            correct['casetype'].append(df['casetype'][idx])
            correct['textlen'].append(df['textlen'][idx])
        else:
            incorrect['review_body'].append(df['review_body'][idx])
            incorrect['casetype'].append(df['casetype'][idx])
            incorrect['textlen'].append(df['textlen'][idx])
            
    return pd.DataFrame.from_dict(correct), pd.DataFrame.from_dict(incorrect)