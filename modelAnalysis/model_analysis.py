import pandas as pd
import sys
from functools import partial

from helper_files import clean_nan, encode_labels, join_columns, load_model, score_model, calc_text_len, get_correct_incorrect_reviews
from casetypes import casetypes, get_casetype
from plotting import box_plot, create_term_freq_df_plots
from word_frequency import create_term_freq_df

file_path = sys.argv[1]
model_path = sys.argv[2]

df = pd.read_csv(file_path, index_col=0)
model = load_model(model_path)
model_name = model['MultinomialNB']

if len(df) > 100:
    df = clean_nan(df)
    df = join_columns(df, 'review_body', 'review_headline')
    check_listing = False
else:
    check_listing = True

df = encode_labels(df)

print("General Score of the model")
print(f"{model}: ", score_model(model, df), "\n")

df = calc_text_len(df)
box_plot = partial(box_plot, folder_name=f'{model_name}')
box_plot(df, name=f'{model_name} Overall')
preds = model.predict(df['review_body'])
actual = df.sentiment
correct_df, incorrect_df = get_correct_incorrect_reviews(df, preds, actual)
box_plot(correct_df, name=f'{model_name} Correct')
box_plot(incorrect_df, name=f'{model_name} Incorrect')

if check_listing:
    casetypes_dfs = []
    casetype_model = partial(score_model, model)

    for casetype in casetypes:
        casetype_df = get_casetype(df, casetype)
        #casetypes_dfs.append(casetype_df)
        print(f"{model}, {casetype}: ", casetype_model(casetype_df))
        box_plot(casetype_df, name=f'{casetype}') #maybe also do boxplots per casetype per correct_df, incorrect_df

        term_freq_df = create_term_freq_df(casetype_df)
        create_term_freq_df_plots(term_freq_df, name=casetype, folder_name=f'{model_name}')


    