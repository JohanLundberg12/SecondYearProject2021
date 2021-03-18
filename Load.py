import gzip
import json
import pandas as pd

def compressed_json_to_df(path):
    return pd.concat([pd.json_normalize(json.loads(line)) for line in gzip.open(path)])

def json_to_df(path):
    data = [json.loads(line) for line in open(path)]
    return pd.json_normalize(data)