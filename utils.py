import torch
import numpy as np

from sklearn.model_selection import train_test_split
from datetime import datetime

def str2list(value):
    return value.split(',')

def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_datetime_format():
    now = datetime.now()
    return f"{now.strftime('%Y%m%d-%H%M%S')}"

def get_filename(config, f1):
    time_str = f"{get_datetime_format()}"
    text_str = str(config['text_feature_cols'])
    num_str = str(config['numerical_feature_cols'])
    epoch_str = str(config['epochs'])
    bs_str = str(config['batch_size'])
    lr_str = f"{config['lr']:.1e}" if type(config['lr']) == float else config['lr']
    comb_h_str = str(config['ff_combined_hidden_dim'])
    h_str = str(config['ff_hidden_dim'])
    f1_str = f"{np.round(f1*1e4):.0f}"

    return f"{time_str}_{text_str}_{num_str}_e{epoch_str}_b{bs_str}_lr{lr_str}_{comb_h_str}_{h_str}_fscore{f1_str}"

def train_val_test_split(input_ids, att_masks, numericals, y, seed, val_size=0.05, test_size=0.15):
    X = {'train': {}, 'val': {}, 'test': {}}
    Y = {}

    indices = list(range(input_ids.shape[0]))

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=seed)

    X['train']['input_ids'], X['val']['input_ids'], X['test']['input_ids'] = input_ids[train_idx], input_ids[val_idx], input_ids[test_idx]
    X['train']['att_masks'], X['val']['att_masks'], X['test']['att_masks'] = att_masks[train_idx], att_masks[val_idx], att_masks[test_idx]
    X['train']['numericals'], X['val']['numericals'], X['test']['numericals'] = numericals[train_idx], numericals[val_idx], numericals[test_idx]
    Y['train'], Y['val'], Y['test'] = y[train_idx], y[val_idx], y[test_idx]

    return X, Y