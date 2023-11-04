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
from transformers import AutoModel, BertModel


class stance_classifier(nn.Module):
  def __init__(self,num_labels,model_select, label_vectors):
    super(stance_classifier, self).__init__()        
    # self.dropout = nn.Dropout(0.)
    # self.relu = nn.ReLU()        
    if model_select == 'Bertweet':
      self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
    elif model_select == 'Bert':
      self.bert = BertModel.from_pretrained("bert-base-uncased")
    # self.bert.pooler = None
    # self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    # self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
    # self.mask_cls = nn.Linear(self.bert.config.hidden_size, 512)
    self.labels = label_vectors


def forward(self, x_input_ids, x_seg_ids, x_atten_masks, mask_pos, label):      
    last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)[0]
    predictions = tuple()
    # for i in range(len(x_input_ids)):
    #   predicted_mask_token = last_hidden[i, mask_pos[i]] 
    #   predictions += (torch.dot(predicted_mask_token, self.labels[label]),)
    # predictions_tensor = torch.stack(predictions).cuda()
    
    # return predictions_tensor

    for i in range(len(x_input_ids)):
      predicted_mask_token = last_hidden[i, mask_pos[i]]
      temp = tuple()
      temp += (torch.dot(predicted_mask_token, self.labels[0]),)
      temp += (torch.dot(predicted_mask_token, self.labels[1]),)
      temp += (torch.dot(predicted_mask_token, self.labels[2]),)            
      predictions += (torch.stack(temp).cuda(),)
      # predictions = predictions + (predicted_mask_token,)
    # prediction_tensor = torch.stack(predictions).cuda()
    predictions_tensor = torch.stack(predictions).cuda()
    # pred_reshape = self.mask_cls(prediction_tensor)
    # last_hidden2 = self.bert(pred_reshape.long())
    # cls = last_hidden2[0][:,0]
    # query = self.dropout(predictions_tensor)
    # linear = self.relu(self.linear(query))
    # out = self.out(linear)
    
    return predictions_tensor
