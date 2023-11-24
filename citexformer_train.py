import pandas as pd
import numpy as np

from tqdm import tqdm
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import argparse

from preprocess import clean_data, numerical_cols, preprocess
from dataset import get_dataloaders
from citexformer import CiteXFormer
from evaluate import evaluate
from utils import get_filename, train_val_test_split, get_device, str_or_int, str2list

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', type=str, default='filtered_corpusid_input.csv')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-n', '--num_row_samples', type=str_or_int, default='all')
parser.add_argument('-T', '--text_feature_cols', type=str2list, default=['abstract', 'title'])
parser.add_argument('-N', '--numerical_feature_cols', type=str2list, default=['year', 'publicationDate'])
parser.add_argument('-v', '--val_size', type=float, default=0.05)
parser.add_argument('-t', '--test_size', type=float, default=0.15)
parser.add_argument('-e', '--epoch', type=int, default=5)
parser.add_argument('-b', '--batch_size', type=int, default=6)
parser.add_argument('-l', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-f', '--ff_combined_hidden_dim', type=int, default=256)
parser.add_argument('-F', '--ff_hidden_dim', type=int, default=128)
parser.add_argument('-D', '--dropout', type=float, default=0.1)

args = parser.parse_args()
tqdm.pandas()

SEED = args.seed
NUM_ROW_SAMPLES = args.num_row_samples
CSV_DIR = args.data_dir

text_feature_cols = args.text_feature_cols
numerical_feature_cols = args.numerical_feature_cols

device = get_device()

def train(model, config, dataloader, loss_fn, optimizer, scheduler=None):
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }

    for epoch in range(config['epochs']):
        # Train
        model.train()
        losses = []
        print(f"Training epoch {epoch+1}...")

        for input_ids, att_masks, numericals, labels in tqdm(dataloader['train']):
            input_ids = input_ids.to(device)
            att_masks = att_masks.to(device)
            numericals = numericals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(input_ids, att_masks, numericals)
            loss = loss_fn(pred, labels)

            loss.backward()
            optimizer.step()
            losses.append(loss)

        if scheduler != None:
            scheduler.step()

        avg_train_loss = torch.stack(losses).mean().item()
        metrics['train_loss'].append(avg_train_loss)

        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

        # Validate
        print(f"Evaluating...")
        val_loss, precision, recall, f1 = evaluate(model, dataloader['val'], loss_fn, config)

        # Save model if best or first
        if epoch == 0 or f1 > np.max(metrics['f1']):
            filename = get_filename(config, f1)
            path = f"weights/{filename}.model"
            model.save_state_dict(path)

        metrics['val_loss'].append(val_loss)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

        print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}")

    return metrics

def main():
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

    X, Y = train_val_test_split(input_ids, att_masks, numericals, y, seed=SEED, val_size=args.val_size, test_size=args.test_size)
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

    # Train
    config = {
        'text_feature_cols': text_feature_cols,
        'numerical_feature_cols': numerical_feature_cols,
        'num_row_samples': NUM_ROW_SAMPLES,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'seed': args.seed,
        'text_col_size': len(text_feature_cols),
        'num_col_size': len(numerical_cols(numerical_feature_cols)),
        'epochs': args.epoch,
        'batch_size': args.batch_size,
        'lr': args.learning_rate,
        'ff_combined_hidden_dim': args.ff_combined_hidden_dim,
        'ff_hidden_dim': args.ff_hidden_dim,
        'dropout': args.dropout,
    }

    loss_fn = nn.CrossEntropyLoss()
    dataloader = get_dataloaders(X, Y, config['batch_size'])

    citexformer_model = CiteXFormer(config).to(device)

    optimizer = torch.optim.AdamW(citexformer_model.parameters(), lr=config['lr'])
    metrics = train(citexformer_model, config, dataloader, loss_fn, optimizer, scheduler=None)

if __name__ == "__main__":

    main()