import gzip
import json
import pandas as pd

def compressed_json_to_df(path):
    return pd.concat([pd.json_normalize(json.loads(line)) for line in gzip.open(path)])