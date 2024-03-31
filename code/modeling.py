import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class StanceClassifier(nn.Module):
    def __init__(self, num_labels):
        super(StanceClassifier, self).__init__()
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')  
        self.bert.pooler = None
        
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels) 

    def forward(self, embedding, input_ids, attention_mask):
        if embedding:
            last_hidden = self.bert(input_ids, attention_mask)
            word_embed = last_hidden[0][:, 1]
            return word_embed

        else:
            last_hidden = self.bert(input_ids, attention_mask)
            cls = last_hidden[0][:, 0]
            query = self.dropout(cls)
            linear = self.relu(self.linear(query))
            out = self.out(linear)                    
            return out
            
