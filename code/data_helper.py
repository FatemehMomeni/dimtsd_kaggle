import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def get_domain(model, target: [str]):
    '''
    get domain of each target
    :param model: the model to get embeddings from
    :param target: targets
    :return: list of domains corresponding to each target
    '''
    domain_df = pd.read_csv('/kaggle/working/dimtsd_kaggle/proposed/data/domain.csv')
    domain_names = list()
    prototypes, domain = {}, {}
    uniques = set(target)

    # get embeddings of each domain samples and average them
    model.eval()
    with torch.no_grad():
        for i in range(len(domain_df)):
            domain_names.append(domain_df.loc[i, 'domain'])
            target_sample = [[tar] for tar in domain_df.loc[i, 'target'].split(',')]
#             target_sample = target_sample[:3]
            encoded = model.tokenizer(target_sample, is_split_into_words=True, max_length=128, padding='max_length', return_attention_mask=True, return_tensors='pt').to('cuda')
            target_embeds = model(True, encoded.input_ids, encoded.attention_mask)
            prototypes[domain_names[i]] = torch.mean(target_embeds, dim=0)
        prototype_tensor = torch.stack(list(prototypes.values()))

        # get embeddings of unique targets and find the domain they're most similar to
        for t in uniques:
            encoded = model.tokenizer(t, is_split_into_words=True, max_length=128, padding='max_length', return_attention_mask=True, return_tensors='pt').to('cuda')
            embeds = model(True, encoded.input_ids, encoded.attention_mask)
            scores = torch.cosine_similarity(embeds, prototype_tensor)
            index = (scores == max(scores)).nonzero(as_tuple=True)[0][0]  # first [0] as tuple has one element, second [0] first occurrence
            domain[t] = domain_names[index]

    tar_domains = ['' for _ in range(len(target))]
    # index domain of each target
    for k, v in domain.items():
        indices = [i for i, x in enumerate(target) if x == k]
        for i in indices:
            tar_domains[i] = v

    return tar_domains


def prompt_template(text: [str], target1: [str], domain: [str]):
    '''
    template each tweet by a prompt
    :param text: tweets
    :param target1: targets
    :param domain: domains
    :return: a list of prompted texts
    '''    
    prompt = list()
    for i in range(len(text)):
        text[i] = ' '.join(text[i])
        prompt.append(
            f"The stance of text '{text[i]}' towards target '{target1[i]}' on domain '{domain[i]}' is [MASK] from the set of 'favor', 'against', 'none'."
        )
        
    return prompt


def tokenization(tokenizer, prompt: [str], y: [int], batch_size: int, mode: str):
    '''
    tokenize prompts and convert them and labels to tensors
    :param tokenizer: the tokenizer for tokenizing prompts
    :param prompt: prompts
    :param y: labels of target
    :param batch_size: size of mini-batch
    :param mode: phase mode
    :return: TensorDataset and a tensor of labels
    '''    
    encoded = tokenizer(prompt, max_length=512, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt').to('cuda')    
    y = torch.tensor(y, dtype=torch.long).to('cuda')          
    tensor_loader = TensorDataset(encoded.input_ids, encoded.attention_mask, y)
    data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
    data_loader_train = None
    
    if mode == 'train':
        data_loader_train = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)            

    return y, data_loader, data_loader_train


def data_helper_bert(data: dict, model, batch_size: int, mode: str):
    '''
    callee of prompt and tokenization funcations
    :param data: includ tweets, targets, and labels
    :param model: the model
    :param batch_size: size of mini-batch
    :param mode: phase mode
    :return: TensorDataset and a tensor of labels
    '''
    print('Loading data')
    text, tar, y = data['tweet'], data['target'], data['label']
    domains = get_domain(model, tar)       
    prompt = prompt_template(text, tar, domains)
    y, data_loader, data_loader_train = tokenization(model.tokenizer, prompt, y, batch_size, mode)

    return y, data_loader, data_loader_train


def sep_test_set(input_data, general):
    '''
    split test set for each target
    :param input_data: data to be separated
    :param general: generalization test or not
    :return: list of splited test set
    '''
    if general:
        data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    else:
        data_list = [input_data[:387], input_data[387:1361], input_data[1361:3041], input_data[3041:4217],
                     input_data[4217:4835], input_data[4835:5462], input_data[5462:5725], input_data[5725:5997], 
                     input_data[5997:6217], input_data[6217:6502], input_data[6502:6797], input_data[6797:7077], 
                     input_data[7077:7441], input_data[7441:8228], input_data[8228:8837], input_data[8837:9568], 
                     input_data[9568:10237], input_data[10237:10734], input_data[10734:11231], input_data[11231:11948], 
                     input_data[11948:12550]]
#     target_indices, data_list = [], []
#     # find unique targets and their indices
#     for unique in set(unique_input):
#         target_indices.append(unique_input.index(unique))
#     target_indices = sorted(target_indices)

#     for i in range(len(target_indices) - 1):
#         data_list.append(input_data[target_indices[i]: target_indices[i + 1]])
#     data_list.append(input_data[target_indices[-1]:])

    return data_list
