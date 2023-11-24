import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer
from const import MAX_TOKEN_LENGTH, PRETRAINED_MODEL_NAME, TARGET_COL

def get_tokenizer():
    return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

tokenizer = get_tokenizer()

def clean_data(inp, text_feature_cols, numerical_feature_cols):
    # Remove NaNs
    for col in text_feature_cols:
        if col in ['abstract', 'title']:
            # Have a chance to have values in -Text
            inp[col] = inp[col].fillna(inp[col + 'Text'])

        # Replace NaNs with empty string
        inp[col] = inp[col].fillna("")

    # Filter year
    if 'year' in numerical_feature_cols:
        inp = inp[inp['year']!=0]
        inp = inp[~inp['year'].isna()]

    # Filter publicationDate
    if 'publicationDate' in numerical_feature_cols:
        inp = inp[~inp['publicationDate'].isna()]

    return inp.reset_index(drop=True)

def filter_cols(df, all_columns):
    # Filter only desired columns
    # Features + Target
    return df[all_columns]

def encode_and_pad(text):
    # Tokenize the text and truncate if necessary
    tokens = tokenizer.encode(text, truncation=True, max_length=MAX_TOKEN_LENGTH)

    # Pad the sequence if it's shorter than max_length
    if len(tokens) < MAX_TOKEN_LENGTH:
        tokens += [tokenizer.pad_token_id] * (MAX_TOKEN_LENGTH - len(tokens))

    return tokens

def tokenize(df, text_feature_cols):
    all_input_ids = []
    all_att_masks = []
    # Tokenize the text
    for col in text_feature_cols:
        df[col] = df[col].progress_apply(encode_and_pad)

        # Convert lists of tokens into 2D tensors
        input_ids = torch.tensor(df[col].tolist())
        all_input_ids.append(input_ids.long())

        # Attention masks (used to ignore padding tokens)
        attention_masks = input_ids != tokenizer.pad_token_id
        all_att_masks.append(attention_masks.long())


    all_input_ids = np.stack(all_input_ids, axis=2)
    all_att_masks = np.stack(all_att_masks, axis=2)

    print(all_input_ids.shape)

    return all_input_ids, all_att_masks

def preprocess_numerical(df, numerical_feature_cols):
    if 'year' in numerical_feature_cols:
        # Normalize using standard value (Z)
        df['year'] = (df['year'] - df['year'].mean()) / df['year'].std()

    if 'publicationDate' in numerical_feature_cols:
        # Convert to datetime
        df['publicationDate'] = pd.to_datetime(df['publicationDate'])

        # Extract month and day
        df['month'] = df['publicationDate'].dt.month
        df['day'] = df['publicationDate'].dt.day

        # Convert month and day to cyclic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        df = df.drop(['publicationDate', 'month', 'day'], axis=1)

    return df

def all_columns(text_feature_cols, numerical_feature_cols):
    return text_feature_cols + numerical_feature_cols + [TARGET_COL]

def numerical_cols(numerical_feature_cols):
    cols = []
    if 'year' in numerical_feature_cols:
        cols.append('year')

    if 'publicationDate' in numerical_feature_cols:
        cols.extend(['month_sin', 'month_cos', 'day_sin', 'day_cos'])

    return cols

def preprocess(cleaned_df, text_feature_cols, numerical_feature_cols, n, seed):
    # n: If 'all', use the whole data. Else, sample a subset equals to n
    if n == 'all':
        selected_df = cleaned_df
    else:
        selected_df = cleaned_df.sample(n=n, random_state=seed)

    input_ids, att_masks = tokenize(selected_df, text_feature_cols)
    y = selected_df[TARGET_COL].values.astype(np.int64)

    all_cols = all_columns(text_feature_cols, numerical_feature_cols)
    selected_df = filter_cols(selected_df, all_cols)
    selected_df = preprocess_numerical(selected_df, numerical_feature_cols)

    numericals = selected_df[numerical_cols(numerical_feature_cols)].values

    return input_ids, att_masks, numericals, y