import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
import pandas as pd
    
    
def get_domain(model, tokenizer, x_train_target, x_val_target, x_test_target):
  
  domain_df = pd.read_csv('../dataset/domain.csv')
  domain_names = list()
  prototypes = dict()

  train_uniques = set(x_val_target)
  test_uniques = set(x_test_target)
  train_domain = dict()
  test_domain = dict()

  model.eval()
  with torch.no_grad():
    for i in range(len(domain_df)):
      domain_names.append(domain_df.loc[i, 'domain'])
      target_sample = [[tar] for tar in domain_df.loc[i, 'target'].split(',')]
      encoded = tokenizer(target_sample, is_split_into_words = True, max_length = 128, padding = 'max_length', return_attention_mask = True, return_tensors = 'pt').to('cuda')        
      target_embeds = model(True, encoded.input_ids, encoded.attention_mask, None)
      prototypes[domain_names[i]] = torch.mean(target_embeds, dim=0)      
    prototype_matrix = torch.stack(list(prototypes.values()))  

    for t in train_uniques:
      encoded = tokenizer(t, is_split_into_words = True, max_length = 128, padding = 'max_length', return_attention_mask = True, return_tensors = 'pt').to('cuda')
      train_embeds = model(True, encoded.input_ids, encoded.attention_mask, None)
      scores = torch.cosine_similarity(train_embeds, prototype_matrix)    
      index = (scores == max(scores)).nonzero(as_tuple=True)[0][0] # first [0] as tuple has one element, second [0] first occuerrence
      train_domain[t] = domain_names[index]
  
    for t in test_uniques:
      encoded = tokenizer(t, is_split_into_words = True, max_length = 128, padding = 'max_length', return_attention_mask = True, return_tensors = 'pt').to('cuda')
      test_embeds = model(True, encoded.input_ids, encoded.attention_mask, None)    
      scores = torch.cosine_similarity(test_embeds, prototype_matrix)    
      index = (scores == max(scores)).nonzero(as_tuple=True)[0][0]
      test_domain[t] = domain_names[index]

  for k,v in train_domain.items():
    indices = [i for i, x in enumerate(x_train_target) if x == k]
    for i in indices:
      x_train_target[i] = v
    indices = [i for i, x in enumerate(x_val_target) if x == k]
    for i in indices:
      x_val_target[i] = v
    
  for k,v in test_domain.items():
    indices = [i for i, x in enumerate(x_test_target) if x == k]
    for i in indices:
      x_test_target[i] = v

  return x_train_target, x_val_target, x_test_target


def prompt_template(text, target, domain):
  for i in range(len(text)):
    text[i] = f"The stance of text '{text[i]}' towards target '{target[i]}' on domain '{domain[i]}' is [MASK] from the set of 'favor', 'against', 'none'."
  return text


def tokenization(tokenizer, text, y, mode, batch_size):
    y = torch.tensor(y, dtype=torch.long).to('cuda')
    encoded = tokenizer(text, max_length = 512, padding = 'max_length', return_attention_mask = True, return_token_type_ids = True, truncation = True, return_tensors = 'pt').to('cuda')    
    tensor_loader = TensorDataset(encoded.input_ids, encoded.token_type_ids, encoded.attention_mask, y)

    if mode == 'train':
      data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size) # for training
      data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size) # for train evaluation
      return y, data_loader, data_loader_distill
    
    else:
      data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
      return y, data_loader
    

def data_helper_bert(x_train_all, x_val_all, x_test_all, model, batch_size):    
  print('Loading data')
    
  x_train, x_train_target, y_train = x_train_all[0], x_train_all[1], x_train_all[2]
  x_val, x_val_target, y_val = x_val_all[0], x_val_all[1], x_val_all[2]
  x_test, x_test_target, y_test = x_test_all[0], x_test_all[1], x_test_all[2]  
  
  print(f"Length of original x_train: {len(x_train)}")
  print(f"Length of original x_val: {len(x_val)},\t the sum is: {sum(y_val)}")
  print(f"Length of original x_test: {len(x_test)},\t the sum is: {sum(y_test)}")
    
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')  

  train_domains, val_domains, test_domains = get_domain(model, tokenizer, x_train_target, x_val_target, x_test_target)

  x_train = prompt_template(x_train, x_train_target, train_domains)
  x_val = prompt_template(x_val, x_val_target, val_domains)
  x_test = prompt_template(x_test, x_test_target, test_domains)

  y_train, train_data_loader, train_data_loader_distill = tokenization(tokenizer, x_train, y_train, 'train', batch_size)
  y_val, val_data_loader = tokenization(tokenizer, x_val, y_val, 'val', batch_size)
  y_test, test_data_loader = tokenization(tokenizer, x_test, y_test, 'test', batch_size)

  train_all = [y_train, train_data_loader, train_data_loader_distill]
  val_all = [y_val, val_data_loader]
  test_all = [y_test, test_data_loader]
  
  return train_all, val_all, test_all


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

