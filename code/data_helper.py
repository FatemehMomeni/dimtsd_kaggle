import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    
    
def tokenization(tokenizer, target, text, domain):
  
  for i in range(len(text)):
    text[i] = f"The stance of text '{text[i]}' towards target '{target[i]}' on domain '{domain[i]}' is [MASK] from the set of 'favor', 'against', 'none'."
  
  tokenized_data = tokenizer(text, max_length = 512, padding = 'max_length', return_attention_mask = True, return_token_type_ids = True, truncation = True, return_tensors = 'pt').to('cuda')
  
  return [tokenized_data.input_ids, tokenized_data.token_type_ids, tokenized_data.attention_mask]


def data_helper_bert(x_train_all, x_val_all, x_test_all):    
  print('Loading data')
    
  x_train, x_train_target, x_train_domain, y_train = x_train_all[0], x_train_all[1], x_train_all[2], x_train_all[3]
  x_val, x_val_target, x_val_domain, y_val = x_val_all[0], x_val_all[1], x_val_all[2], x_val_all[3]
  x_test, x_test_target, x_test_domain, y_test = x_test_all[0], x_test_all[1], x_test_all[2], x_test_all[3]
    
  print(f"Length of original x_train: {len(x_train)}")
  print(f"Length of original x_val: {len(x_val)},\t the sum is: {sum(y_val)}")
  print(f"Length of original x_test: {len(x_test)},\t the sum is: {sum(y_test)}")
    
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')

  x_train_all = tokenization(tokenizer, x_train_target, x_train, x_train_domain)
  x_val_all = tokenization(tokenizer, x_val_target, x_val, x_val_domain)
  x_test_all = tokenization(tokenizer, x_test_target, x_test, x_test_domain)
    
  x_train_all.append(y_train)
  x_val_all.append(y_val)
  x_test_all.append(y_test)
  
  return x_train_all, x_val_all, x_test_all


def data_loader(x_all, batch_size, mode, **kwargs):
  input_ids = x_all[0]
  token_type_ids = x_all[1]
  attention_mask = x_all[2]
  y = torch.tensor(x_all[3], dtype=torch.long).to('cuda')

  tensor_loader = TensorDataset(input_ids, token_type_ids, attention_mask, y)

  if mode == 'train':
    data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size) # for training
    data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size) # for evaluation
    return y, data_loader, data_loader_distill
  
  else:
    data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
    return y, data_loader


def sep_test_set(input_data, unique_input):

  target_indices, data_list = [], []
  # find unique targets and their indices
  for unique in set(unique_input):
    target_indices.append(unique_input.index(unique))
  target_indices = sorted(target_indices)
  
  for i in range(len(target_indices)-1):
    data_list.append(input_data[target_indices[i] : target_indices[i+1]])
  data_list.append(input_data[target_indices[-1]:])  
  
  return data_list

