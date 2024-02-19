import torch
import torch.nn as nn
from transformers import AutoModel, BertModel


class stance_classifier(nn.Module):
  def __init__(self,num_labels):
    super(stance_classifier, self).__init__()
    self.dropout = nn.Dropout(0.)
    self.relu = nn.ReLU()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.bert.pooler = None
    self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    self.out = nn.Linear(self.bert.config.hidden_size, num_labels) 
          
      
  def forward(self, embedding, input_ids, token_type_ids, attention_mask):
    if embedding:
      last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      word_embed = last_hidden[0][:,1]      
      return word_embed
      
    else:
      last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
      cls = last_hidden[0][:,0]
      query = self.dropout(cls)
      linear = self.relu(self.linear(query))
      out = self.out(linear)        
      return out
        
