# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

# def convert_data_to_ids(tokenizer, text):    
#     input_ids, seg_ids, attention_masks, sent_len = [], [], [], []    
#     for txt in text:
#         encoded_dict = tokenizer.encode_plus(txt, add_special_tokens = True,
#                             max_length = 128, padding = 'max_length',
#                             return_attention_mask = True, truncation = True)
    
#         input_ids.append(encoded_dict['input_ids'])
#         seg_ids.append(encoded_dict['token_type_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
#         sent_len.append(sum(encoded_dict['attention_mask']))
    
#     return input_ids, seg_ids, attention_masks, sent_len


# def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select):    
#     print('Loading data')
    
#     x_train, y_train = x_train_all[0],x_train_all[1]
#     x_val, y_val = x_val_all[0],x_val_all[1]
#     x_test, y_test = x_test_all[0],x_test_all[1]
    
#     print("Length of original x_train: %d"%(len(x_train)))
#     print("Length of original x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
#     print("Length of original x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
#     if model_select == 'Bertweet':
#         tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
#     elif model_select == 'Bert':
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

#     x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
#                     convert_data_to_ids(tokenizer, x_train)
#     x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
#                     convert_data_to_ids(tokenizer, x_val)
#     x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
#                     convert_data_to_ids(tokenizer, x_test)
    
#     x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len]
#     x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len]
#     x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len]
    
#     print(len(x_train), sum(y_train))
#     print("Length of final x_train: %d"%(len(x_train)))
    
#     return x_train_all,x_val_all,x_test_all


# def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    
#     x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
#     x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
#     x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
#     y = torch.tensor(x_all[3], dtype=torch.long).cuda()
#     x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()

#     if model_name == 'student' and mode == 'train':
#         y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
#         tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,y2)
#     else:
#         tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len)

#     if mode == 'train':
#         data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
#         data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

#         return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
#     else:
#         data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

#         return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader


# def sep_test_set(input_data,dataset_name):
#     data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
#     return data_list


import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, T5Tokenizer
    

def convert_data_to_ids(tokenizer, text, target, y_str):    
  # input_ids, seg_ids, attention_masks, mask_pos = [], [], [], []    
  for i in range(len(text)):
    text[i] = f"The stance of text '{text[i]}' with respect to target '{target[i]}' from the set 'favor', 'against', 'none' is"
  #   if y_str:
  #     encoded_dict = tokenizer.encode_plus(text[i], y_str[i], max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
  #   else:
  #     encoded_dict = tokenizer.encode_plus(text[i], max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
  #   input_ids.append(encoded_dict.input_ids)
  #   attention_masks.append(encoded_dict.attention_mask)
  #   seg_ids.append(encoded_dict.token_type_ids)
  #   mask_pos.append(input_ids[i].index(tokenizer.mask_token_id))

  encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True).to('cuda:0')
  tokenized_text = [encoding.input_ids, encoding.attention_mask]
  
  label_encoding = tokenizer(y_str, return_tensors='pt', padding='max_length', max_length=128, truncation=True).to('cuda:0')
  labels = label_encoding.input_ids
  labels[labels == tokenizer.pad_token_id] = -100        
  
  return tokenized_text, labels
      
  # return input_ids, seg_ids, attention_masks, mask_pos


def data_helper_bert(x_train_all,x_val_all,x_test_all,model_select):        
    x_train, x_train_tar, y_train, y_train_str = x_train_all[0],x_train_all[1],x_train_all[2],x_train_all[3]
    x_val, x_val_tar, y_val, y_val_str = x_val_all[0],x_val_all[1],x_val_all[2],x_val_all[3]
    x_test, x_test_tar, y_test, y_test_str = x_test_all[0],x_test_all[1],x_test_all[2],x_test_all[3]
    
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')
        # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    train_tokenized_text, train_tokenized_label = convert_data_to_ids(tokenizer, x_train, x_train_tar, y_train_str)
    val_tokenized_text, val_tokenized_labe = convert_data_to_ids(tokenizer, x_val, x_val_tar, y_val_str)
    test_tokenized_text, test_tokenized_label = convert_data_to_ids(tokenizer, x_test, x_test_tar, y_test_str)
    
    # x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_mask]
    # x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_mask]
    # x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_mask]
    x_train_all = [train_tokenized_text, train_tokenized_label, y_train]
    x_val_all = [val_tokenized_text, val_tokenized_labe, y_val]
    x_test_all = [test_tokenized_text, test_tokenized_label, y_test]
        
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    
    x_input_ids_text = x_all[0][0]
    x_attention_mask_text = x_all[0][1]
    x_input_ids_label = x_all[1]
    # x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    # x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[2]).to('cuda:0')
    # x_mask = torch.tensor(x_all[4], dtype=torch.long).cuda()

    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        tensor_loader = TensorDataset(x_input_ids_text,x_attention_mask_text,x_input_ids_label,y,y2)
    else:
        tensor_loader = TensorDataset(x_input_ids_text,x_attention_mask_text,x_input_ids_label,y)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return y, data_loader, data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return y, data_loader


def sep_test_set(input_data):
    data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    return data_list
