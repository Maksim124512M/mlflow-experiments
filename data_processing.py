import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('bank.csv')

df_encoded = pd.get_dummies(data=df, columns=['job', 'default', 'housing', 'loan', 'marital', 'contact', 'poutcome', 'deposit'])

edu_encoder = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary', 'unknown']])

month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

df_encoded['month_num'] = df['month'].map(month_map)

df_encoded['education'] = edu_encoder.fit_transform(df_encoded[['education']])
df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['month_num'] / 12)
df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['month_num'] / 12)

df_encoded = df_encoded.drop(columns=['month'])

print(df_encoded.head())