import json
import pandas as pd

with open('preprocessing_key.json') as f:
    preprocessing_key = json.load(f)

mappings = preprocessing_key['mappings']
scaling = preprocessing_key['scaling']

for col, map_dict in mappings.items():
    if col == 'YesNoColumns':
        yesno_map = map_dict
        yesno_cols = ['PhoneService', 'PaperlessBilling', 'Dependents', 'Partner']  # твої колонки з Yes/No
        for c in yesno_cols:
            if c in df.columns:
                df[c] = df[c].map(yesno_map).fillna(df[c])
    else:
        if col in df.columns:
            df[col] = df[col].map(map_dict).fillna(df[col])

for col, divisor in scaling.items():
    if col in df.columns:
        df[col] = df[col] / divisor
