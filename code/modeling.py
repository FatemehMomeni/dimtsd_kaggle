import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertConfig, DistilBertModel


class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select, v_labels, tokenizer):

        super(stance_classifier, self).__init__()        
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()        
        if model_select == 'Bertweet':            
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        elif model_select == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.tokenizer = tokenizer
        self.label_vectors = v_labels
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, mask_indices):              
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)        
        predictions = last_hidden[0]
        prediction_probabilities = list(list())
        for i in range(len(predictions)):          
          _, predicted_label_id = torch.sort(predictions[i, mask_indices[i]],  descending=True)               
          # predicted_label_word = self.tokenizer.convert_ids_to_tokens(predicted_label_id)          
          # encoded_dict = self.tokenizer.encode_plus(predicted_label_word[0], add_special_tokens=True, 
          #                                           max_length=512, padding='max_length',
          #                                           return_attention_mask=True, truncation=True)
          # input_ids_tensor = torch.tensor(encoded_dict['input_ids'], dtype=torch.long).cuda()
          # attention_mask_tensor = torch.tensor(encoded_dict['attention_mask'], dtype=torch.long).cuda()          
          # out0 = self.bert(input_ids=predicted_label_id)
          # out1 = out0[0][:,0]          

          temp = list()
          temp.append(torch.dot(predicted_label_id.float(), self.label_vectors[0]))
          temp.append(torch.dot(predicted_label_id.float(), self.label_vectors[1]))
          temp.append(torch.dot(predicted_label_id.float(), self.label_vectors[2]))            
          prediction_probabilities.append(temp)
        prediction_probabilities_tensor = torch.tensor(prediction_probabilities).cuda()
        
        return prediction_probabilities_tensor
