import torch

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_device, get_filename
from const import NUM_TARGET_CLS

device = get_device()

def evaluate(model, dataloader_to_eval, loss_fn, config):
    model.eval()

    predictions = []
    ground_truths = []
    losses = []

    with torch.no_grad():
        for input_ids, att_masks, numericals, labels in tqdm(dataloader_to_eval):
            ground_truths.extend(labels.tolist())

            input_ids = input_ids.to(device)
            att_masks = att_masks.to(device)
            numericals = numericals.to(device)
            labels = labels.to(device)

            pred = model(input_ids, att_masks, numericals)
            loss = loss_fn(pred, labels)
            losses.append(loss)
            predictions.extend(torch.argmax(pred, dim=-1).tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='micro')
    avg_val_loss = torch.stack(losses).mean().item()

    # Plot Confusion Matrix
    plot_cm(ground_truths, predictions, f1, config)

    return avg_val_loss, precision, recall, f1

def plot_cm(ground_truths, predictions, f1, config):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()  # Flatten the array of axes

    for i, norm in enumerate([None, 'true', 'pred', 'all']):
        cm = confusion_matrix(ground_truths, predictions, normalize=norm)
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='.4f' if norm != None else 'g',
            xticklabels=[*range(NUM_TARGET_CLS)],
            yticklabels=[*range(NUM_TARGET_CLS)],
            ax=axs[i]
        )
        ax.set_title(str(norm))

    plt.tight_layout()
    plt.savefig(f"plots/{get_filename(config, f1)}.png")