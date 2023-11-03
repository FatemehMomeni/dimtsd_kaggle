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
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

def convert_data_to_ids(tokenizer, text):    
  input_ids, attention_masks, seg_ids, mask_pos = [], [], [], []
  for i in range(len(text)):
    encoded_dict = tokenizer(text[i], max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
    input_ids.append(encoded_dict.input_ids)
    attention_masks.append(encoded_dict.attention_mask)
    seg_ids.append(encoded_dict.token_type_ids)
    mask_pos.append(input_ids[i].index(tokenizer.mask_token_id))
  
  return input_ids, seg_ids, attention_masks, mask_pos


def data_helper_bert(x_train_all, x_val_all, x_test_all, model_select):        
    
  x_train, y_train = x_train_all[0], x_train_all[1]
  x_train_a, x_train_n, x_train_f = x_train[0], x_train[1], x_train[2]
  y_train_a, y_train_n, y_train_f = y_train[0], y_train[1], y_train[2]
  x_val, y_val = x_val_all[0], x_val_all[1]
  x_test, y_test = x_test_all[0], x_test_all[1]
  
  if model_select == 'Bertweet':
    tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, mask_token='[MASK]')  
  elif model_select == 'Bert':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')   

  train_input_ids_a, train_seg_ids_a, train_atten_masks_a, train_mask_pos_a = convert_data_to_ids(tokenizer, x_train_a)
  train_input_ids_n, train_seg_ids_n, train_atten_masks_n, train_mask_pos_n = convert_data_to_ids(tokenizer, x_train_n)
  train_input_ids_f, train_seg_ids_f, train_atten_masks_f, train_mask_pos_f = convert_data_to_ids(tokenizer, x_train_f)
  train_input_ids = [train_input_ids_a, train_input_ids_n, train_input_ids_f]
  train_seg_ids = [train_seg_ids_a, train_seg_ids_n, train_seg_ids_f]
  train_atten_masks = [train_atten_masks_a, train_atten_masks_n, train_atten_masks_f]
  train_mask_pos = [train_mask_pos_a, train_mask_pos_n, train_mask_pos_f]

  val_input_ids, val_seg_ids, val_atten_masks, val_mask_pos = convert_data_to_ids(tokenizer, x_val)  
  test_input_ids, test_seg_ids, test_atten_masks, test_mask_pos = convert_data_to_ids(tokenizer, x_test)
  
  x_train_all = [train_input_ids, train_seg_ids, train_atten_masks, train_mask_pos, y_train]
  x_val_all = [val_input_ids, val_seg_ids, val_atten_masks, val_mask_pos, y_val]
  x_test_all = [test_input_ids, test_seg_ids, test_atten_masks, test_mask_pos, y_test]
  
  return x_train_all, x_val_all, x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):    
  if mode == 'train':
    x_input_ids_a = torch.tensor(x_all[0][0], dtype=torch.long).cuda()
    x_input_ids_n = torch.tensor(x_all[0][1], dtype=torch.long).cuda()
    x_input_ids_f = torch.tensor(x_all[0][2], dtype=torch.long).cuda()
    x_seg_ids_a = torch.tensor(x_all[1][0], dtype=torch.long).cuda()
    x_seg_ids_n = torch.tensor(x_all[1][1], dtype=torch.long).cuda()
    x_seg_ids_f = torch.tensor(x_all[1][2], dtype=torch.long).cuda()
    x_atten_masks_a = torch.tensor(x_all[2][0], dtype=torch.long).cuda()
    x_atten_masks_n = torch.tensor(x_all[2][1], dtype=torch.long).cuda()
    x_atten_masks_f = torch.tensor(x_all[2][2], dtype=torch.long).cuda()
    x_mask_a = torch.tensor(x_all[3][0], dtype=torch.long).cuda()
    x_mask_n = torch.tensor(x_all[3][1], dtype=torch.long).cuda()
    x_mask_f = torch.tensor(x_all[3][2], dtype=torch.long).cuda()
    y_a = torch.tensor(x_all[4][0], dtype=torch.long).cuda()
    y_n = torch.tensor(x_all[4][1], dtype=torch.long).cuda()
    y_f = torch.tensor(x_all[4][2], dtype=torch.long).cuda()
  else:
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[4]).cuda()
    x_mask = torch.tensor(x_all[3], dtype=torch.long).cuda()

  if model_name == 'student' and mode == 'train':
    y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
    tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, x_mask, y, y2)
  else:
    if mode == 'train':
      tensor_loader_a = TensorDataset(x_input_ids_a, x_seg_ids_a, x_atten_masks_a, x_mask_a, y_a)
      tensor_loader_n = TensorDataset(x_input_ids_n, x_seg_ids_n, x_atten_masks_n, x_mask_n, y_n)
      tensor_loader_f = TensorDataset(x_input_ids_f, x_seg_ids_f, x_atten_masks_f, x_mask_f, y_f)
    else:
      tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, x_mask, y)

  if mode == 'train':
    data_loader_a = DataLoader(tensor_loader_a, shuffle=True, batch_size=batch_size)
    data_loader_n = DataLoader(tensor_loader_n, shuffle=True, batch_size=batch_size)
    data_loader_f = DataLoader(tensor_loader_f, shuffle=True, batch_size=batch_size)
    data_loader_distill_a = DataLoader(tensor_loader_a, shuffle=False, batch_size=batch_size)
    data_loader_distill_n = DataLoader(tensor_loader_n, shuffle=False, batch_size=batch_size)
    data_loader_distill_f = DataLoader(tensor_loader_f, shuffle=False, batch_size=batch_size)
    return [y_a, y_n, y_f], [data_loader_a, data_loader_n, data_loader_f], [data_loader_distill_a, data_loader_distill_n, data_loader_distill_f]  
  else:
    data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)    
    return y, data_loader


def sep_test_set(input_data):
    data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    return data_list
