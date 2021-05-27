import sys
from keras.preprocessing import text
import ast
from collections import defaultdict 
import numpy as np
import pandas as pd
from scipy.stats import hmean
from scipy.stats import norm
from pathlib import Path
import ntpath

from helper_files import clean_nan, join_columns, encode_labels, calc_text_len
from plotting import box_plot, create_term_freq_df_plots
from casetypes import casetypes, get_casetype

def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

def fit_tokenizer(df):
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(df.review_body)

    return tokenizer


def create_counts(tokenizer):
    word2id = defaultdict(int)
    id2word = defaultdict(str)
    word2id.update(tokenizer.word_index)
    id2word.update(tokenizer.index_word)
    counts = tokenizer.get_config()['word_counts']
    counts = ast.literal_eval(counts)

    return counts, word2id, id2word


def create_term_freq_df(df):
    tokenizer = fit_tokenizer(df)
    counts, word2id, id2word = create_counts(tokenizer)

    term_freq_df = np.zeros((len(df),len(counts)+1))
    wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in df.review_body]
    for idx1 in range(len(term_freq_df)):
        for idx2 in range(len(wids[idx1])):
            col_index = wids[idx1][idx2]
            term_freq_df[idx1,col_index] += 1

    msk = df.sentiment==0
    pos_wids = term_freq_df[~msk]
    neg_wids = term_freq_df[msk]

    neg_tf = np.sum(neg_wids,axis=0,dtype=int)
    pos_tf = np.sum(pos_wids,axis=0,dtype=int)
    term_freq_df = pd.DataFrame([neg_tf,pos_tf])
    term_freq_df.rename(columns=id2word,inplace=True)
    term_freq_df = term_freq_df.transpose()
    term_freq_df = term_freq_df.iloc[1:]
    term_freq_df = term_freq_df.sort_index(axis=0)

    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']

    term_freq_df['pos_rate'] = term_freq_df['positive'] * 1./term_freq_df['total']
    term_freq_df['pos_freq_pct'] = term_freq_df['positive'] * 1./term_freq_df['positive'].sum()
    term_freq_df['pos_hmean'] = term_freq_df.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']]) if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
    term_freq_df['pos_rate_normcdf'] = normcdf(term_freq_df['pos_rate'])
    term_freq_df['pos_freq_pct_normcdf'] = normcdf(term_freq_df['pos_freq_pct'])
    term_freq_df['pos_normcdf_hmean'] = hmean([term_freq_df['pos_rate_normcdf'], term_freq_df['pos_freq_pct_normcdf']])

    term_freq_df['neg_rate'] = term_freq_df['negative'] * 1./term_freq_df['total']
    term_freq_df['neg_freq_pct'] = term_freq_df['negative'] * 1./term_freq_df['negative'].sum()
    term_freq_df['neg_hmean'] = term_freq_df.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']]) if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 else 0), axis=1)
    term_freq_df['neg_rate_normcdf'] = normcdf(term_freq_df['neg_rate'])
    term_freq_df['neg_freq_pct_normcdf'] = normcdf(term_freq_df['neg_freq_pct'])
    term_freq_df['neg_normcdf_hmean'] = hmean([term_freq_df['neg_rate_normcdf'], term_freq_df['neg_freq_pct_normcdf']])

    return term_freq_df



if __name__ == '__main__':
    file_path = sys.argv[1]
    df = pd.read_csv(file_path, index_col=0)
    data_set_name = ntpath.basename(file_path).split('.')[0]
    folder_path = './word_freq/general_analysis/'

    if len(df) > 100:
        df = clean_nan(df)
        df = join_columns(df, 'review_body', 'review_headline')
        check_listing = False
        folder_path += 'subsets'
    else:
        check_listing = True
        folder_path += 'hard_cases'

    df = encode_labels(df)
    df = calc_text_len(df)
    box_plot(df, folder_name=folder_path, name=f'{data_set_name}_Overall')
    term_freq_df = create_term_freq_df(df)
    create_term_freq_df_plots(term_freq_df, name=f'{data_set_name}_overall', folder_name=folder_path)

    if check_listing:
        for casetype in casetypes:
            casetype_df = get_casetype(df, casetype)
            box_plot(casetype_df, folder_name=folder_path+'/'+casetype, name=f'{data_set_name}_{casetype}')
            term_freq_df = create_term_freq_df(casetype_df)
            create_term_freq_df_plots(term_freq_df, name=f'{data_set_name}_{casetype}', folder_name=folder_path+'/'+casetype)
