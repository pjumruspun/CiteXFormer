import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from const import NUM_TARGET_CLS, PRETRAINED_MODEL_NAME, BERT_LAST_LAYER_DIM

class CiteXFormer(nn.Module):
    def __init__(self, config):
        super(CiteXFormer, self).__init__()

        self.config = config

        # Extra nodes for numerical features
        self.num_col_size = config['num_col_size']

        # Load pretrained weights
        self.transformer = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

        # Linears
        self.ff_combined = nn.Linear(BERT_LAST_LAYER_DIM * config['text_col_size'], config['ff_combined_hidden_dim'])
        self.ff = nn.Linear(config['ff_combined_hidden_dim'] + self.num_col_size, config['ff_hidden_dim'])
        self.out = nn.Linear(config['ff_hidden_dim'], NUM_TARGET_CLS)

        # Dropout
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input_ids, att_masks, numericals):
        all_sent_embeddings = []
        for i in range(input_ids.shape[-1]):
            # i = index of text features
            word_embeddings = self.transformer(input_ids[:, :, i], attention_mask=att_masks[:, :, i])[0]
            sent_embeddings = word_embeddings[:, i, :]
            all_sent_embeddings.append(sent_embeddings)

        combined_text_features = self.combine_features(all_sent_embeddings)
        combined_ff_features = self.dropout(F.relu(self.ff_combined(combined_text_features)))

        if self.num_col_size > 0:
            combined_ff_features = self.combine_features([combined_ff_features, numericals])

        ff_features = self.dropout(F.relu(self.ff(combined_ff_features)))
        logits = self.out(ff_features)
        return logits

    def combine_features(self, list_of_tensors):
        return torch.concat(list_of_tensors, dim=-1)
    
    def save_state_dict(self, path):
        custom_state_dict = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }

        torch.save(custom_state_dict, path)

    @staticmethod
    def from_state_dict_path(path):
        loaded_state_dict = torch.load(path)
        if 'config' in loaded_state_dict.keys():
            # New version, print details
            config = loaded_state_dict['config']
            state_dict = loaded_state_dict['state_dict']
            model = CiteXFormer(config)
            model.load_state_dict(state_dict)

            print("Loaded model with config:")
            print(config)

            return model
        
        raise AttributeError(
            "The model file is an old version, need to specify hyperparams. Do not use this method entirely. Instead, use model.load_state_dict(torch.load(PATH)) instead"
        )
