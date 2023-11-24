import pandas as pd

from tqdm import tqdm
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from preprocess import clean_data, preprocess, numerical_cols
from dataset import get_dataloaders
from citexformer import CiteXFormer
from evaluate import evaluate
from utils import str2list, str_or_int, train_val_test_split, get_device
from const import TARGET_COL

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', type=str, default='filtered_corpusid_input.csv')
parser.add_argument('-w', '--weight_dir', type=str, default='')
parser.add_argument('-S', '--split', choices=['val', 'test'], default='test', help='Specify the split: val or test')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-n', '--num_row_samples', type=str_or_int, default='all')
parser.add_argument('-T', '--text_feature_cols', type=str2list, default=['abstract', 'title'])
parser.add_argument('-N', '--numerical_feature_cols', type=str2list, default=['year', 'publicationDate'])
parser.add_argument('-v', '--val_size', type=float, default=0.05)
parser.add_argument('-t', '--test_size', type=float, default=0.15)
parser.add_argument('-b', '--batch_size', type=int, default=6)
parser.add_argument('-f', '--ff_combined_hidden_dim', type=int, default=256)
parser.add_argument('-F', '--ff_hidden_dim', type=int, default=128)
parser.add_argument('-D', '--dropout', type=float, default=0.1)

args = parser.parse_args()
tqdm.pandas()

CSV_DIR = args.data_dir

device = get_device()

def main():
    # Load model
    try:
        # New version
        citexformer_model = CiteXFormer.from_state_dict_path(args.weight_dir)
        print("Detecting new version of the model... Using the config from training.")

        # Will be used for specifying the result plot image
        config = citexformer_model.config
        
        # Assign variables from the training config
        NUM_ROW_SAMPLES = config['num_row_samples']
        val_size = config['val_size']
        test_size = config['test_size']
        SEED = config['seed']

    except AttributeError:
        # Old version
        print("Detecting old version of the model... Using specified config from command line that used to run this script.")

        # Note that this config doesn't have epochs or learning rate
        # As it is for testing purpose only

        # This config will be used to both initialize the model AND to specify the result plot image.
        # Fields that are marked UNK are the results of old version not retaining any info regarding to those fields.
        config = {
            'text_feature_cols': args.text_feature_cols,
            'numerical_feature_cols': args.numerical_feature_cols,
            'text_col_size': len(args.text_feature_cols),
            'val_size': 'UNK',
            'test_size': 'UNK',
            'seed': 'UNK',
            'num_col_size': len(numerical_cols(args.numerical_feature_cols)),
            'epochs': 'UNK',
            'batch_size': 'UNK',
            'lr': 'UNK',
            'ff_combined_hidden_dim': args.ff_combined_hidden_dim,
            'ff_hidden_dim': args.ff_hidden_dim,
            'dropout': args.dropout,
        }

        citexformer_model = CiteXFormer(config)

        # Then load state dict
        citexformer_model.load_state_dict(torch.load(args.weight_dir))

        # Assign variables from specified config in command line
        NUM_ROW_SAMPLES = args.num_row_samples
        SEED = args.seed
        val_size = args.val_size
        test_size = args.test_size

    text_feature_cols = config['text_feature_cols']
    numerical_feature_cols = config['numerical_feature_cols']

    # Read data
    print("Reading data...")
    df = pd.read_csv(CSV_DIR)

    # Clean data
    print("Cleaning data...")
    cleaned_df = clean_data(df, text_feature_cols, numerical_feature_cols)
    print(cleaned_df.shape)

    # Preprocess
    input_ids, att_masks, numericals, y = preprocess(
        cleaned_df, text_feature_cols, numerical_feature_cols, n=NUM_ROW_SAMPLES, seed=SEED
    )

    X, Y = train_val_test_split(input_ids, att_masks, numericals, y, seed=SEED, val_size=val_size, test_size=test_size)
    print(X['train']['input_ids'].shape)
    print(X['train']['att_masks'].shape)
    print(X['train']['numericals'].shape)
    print(Y['train'].shape)
    print(X['val']['input_ids'].shape)
    print(X['val']['att_masks'].shape)
    print(X['val']['numericals'].shape)
    print(Y['val'].shape)
    print(X['test']['input_ids'].shape)
    print(X['test']['att_masks'].shape)
    print(X['test']['numericals'].shape)
    print(Y['test'].shape)

    # Test
    loss_fn = nn.CrossEntropyLoss()
    dataloader = get_dataloaders(X, Y, args.batch_size)
    citexformer_model = citexformer_model.to(device)
    loss, _, _, f1 = evaluate(citexformer_model, dataloader[args.split], loss_fn, config)

    print(f"Evaluating {Y['test'].shape[0]} rows, {args.split} loss: {loss:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    main()