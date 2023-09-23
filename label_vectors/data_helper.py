import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer


# BERT/BERTweet tokenizer    
def data_helper_bert(model_select, labels):
    print('Loading data')

    if model_select == 'Bertweet':
      tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
      tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    input_ids_l, seg_ids_l, attention_masks_l = [], [], []
    for i in range(len(labels)):
      encoded_dict = tokenizer.encode_plus(labels[i], add_special_tokens=True, max_length=512, padding='max_length',
                                            return_attention_mask=True, truncation=True, )
      input_ids_l.append(encoded_dict['input_ids'])
      seg_ids_l.append(encoded_dict['token_type_ids'])
      attention_masks_l.append(encoded_dict['attention_mask'])
    
    x_labels = [input_ids_l, seg_ids_l, attention_masks_l]
    
    return x_labels


def data_loader(batch_size, model_select, model_name, x_labels, **kwargs):  
  
    l_input_ids = torch.tensor(x_labels[0], dtype=torch.long).cuda()
    l_seg_ids = torch.tensor(x_labels[1], dtype=torch.long).cuda()
    l_atten_masks = torch.tensor(x_labels[2], dtype=torch.long).cuda()          
    l = [l_input_ids, l_seg_ids, l_atten_masks]

    return l
        

def sep_test_set(input_data, dataset_name):
    data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    return data_list
