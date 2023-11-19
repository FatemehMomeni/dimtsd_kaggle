# import torch
# import torch.nn as nn
# from transformers import AutoModel, BertModel


# class stance_classifier(nn.Module):
#     def __init__(self,num_labels,model_select):
#         super(stance_classifier, self).__init__()      
#         self.dropout = nn.Dropout(0.)
#         self.relu = nn.ReLU()        
#         if model_select == 'Bertweet':
#             self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
#         elif model_select == 'Bert':
#             self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.bert.pooler = None
#         self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
#         self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
#     def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len):        
#         last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        
#         cls = last_hidden[0][:,0]
#         query = self.dropout(cls)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
        
#         return out


import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, T5ForConditionalGeneration, T5Tokenizer


class stance_classifier(nn.Module):
  def __init__(self,num_labels,model_select):
    super(stance_classifier, self).__init__()        
    # self.dropout = nn.Dropout(0.)
    # self.relu = nn.ReLU()        
    if model_select == 'Bertweet':
      self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
    elif model_select == 'Bert':
      # self.bert = BertModel.from_pretrained("bert-base-uncased")
      self.bert = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # self.bert.pooler = None
    # self.linear = nn.Linear(128,128) # self.bert.config.hidden_size)
    # self.out = nn.Linear(128, num_labels)
    # self.labels = label_vectors
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


  def forward(self, input_ids, attention_mask, labels, train):     
    if train:
      last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      output = last_hidden.loss
    else:
      last_hidden = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False) 
      output = self.tokenizer.batch_decode(last_hidden, skip_special_tokens=True)[0]
    
    # cls = last_hidden[0][:,0]
    # query = self.dropout(cls)
    # linear = self.relu(self.linear(query))
    # out = self.out(linear)    
    # print(self.tokenizer.decode(last_hidden[0]))
    
    return output
