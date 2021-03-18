def tokenizer(df, include_summary=False):
    '''
    Input: 
        - DataFrame.
    Output:
        - A list of lists of tokens.
    '''
    if include_summary:
        df['reviewText'] = df.reviewText +  " " + df.summary
    
    df = df[df.reviewText.notna()]
    reviews = [[word for word in sentence.split(" ")] for sentence in df.reviewText if len(sentence) > 1]
    
    return reviews