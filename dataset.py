import numpy as np

from torch.utils.data import Dataset, DataLoader

class CiteXFormerDataset(Dataset):
    def __init__(self, input_ids, att_masks, numericals, y):
        self.input_ids = input_ids
        self.att_masks = att_masks
        self.numericals = numericals.astype(np.float32)
        self.y = y

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        att_mask = self.att_masks[index]
        numerical = self.numericals[index]
        y = self.y[index]

        return input_id, att_mask, numerical, y

    def __len__(self):
        return self.input_ids.shape[0]

def get_dataloaders(X, Y, batch_size):
    dataloader = {}
    for split in ['train', 'val', 'test']:
        dataloader[split] = DataLoader(
            CiteXFormerDataset(X[split]['input_ids'], X[split]['att_masks'], X[split]['numericals'], Y[split]),
            batch_size=batch_size,
            shuffle=split=='train',
        )

    return dataloader