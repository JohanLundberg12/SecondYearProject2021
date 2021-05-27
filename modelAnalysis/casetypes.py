casetypes = {
    'fairness':0,
    'invariance':1,
    'regular':2,
    'temporal':3,
    'negation':4
}

def get_casetype(df, casetype):
    return df[df.casetype == casetype]
