import numpy as np

def tokenizer(df, include_summary=False):
    '''
    Input: 
        - DataFrame.
    Output:
        - A list of lists of tokens.
    '''
    df = df.replace(np.nan, '', regex=True)
    if include_summary:
        df['reviewText'] = df.reviewText +  " " + df.summary
    
    reviews = [[word for word in sentence.split(" ")] for sentence in df.reviewText if len(sentence) > 1]
    
    return reviews