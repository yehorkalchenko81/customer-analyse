import pandas as pd

def load_mappings(filepath='mappings.json'):
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_in_cols(df, columns, mapping):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df

def preprocess_input_data(df, mappings):
    # Замінюємо Yes/No на 1/0
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df = replace_in_cols(df, yes_no_cols, mappings['YesNo'])

    # Застосовуємо мапінг для інших категоріальних колонок
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaymentMethod']

    for col in categorical_cols:
        if col in df.columns and col in mappings:
            df[col] = df[col].map(mappings[col]).fillna(df[col])

    # Конвертація в числові типи
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # **Перекодовуємо всі категоріальні (object) колонки у числові коди**
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes

    df.fillna(0, inplace=True)

    return df
