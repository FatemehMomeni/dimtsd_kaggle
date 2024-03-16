import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def get_domain(model, target: [str]):
    '''
    get domain of each target
    :param model: the model to get embeddings from
    :param target: target
    :return: list of domains corresponding to each target
    '''
    domain_df = pd.read_csv('../data/domain.csv')
    domain_names = list()
    prototypes, domain = {}, {}
    uniques = set(target)

    # get embeddings of each domain samples and average them
    model.eval()
    with torch.no_grad():
        for i in range(len(domain_df)):
            domain_names.append(domain_df.loc[i, 'domain'])
            target_sample = [[tar] for tar in domain_df.loc[i, 'target'].split(',')]
            encoded = model.tokenizer(target_sample, is_split_into_words=True, max_length=128, padding='max_length',
                                      return_attention_mask=True, return_tensors='pt').to('cuda')
            target_embeds = model(True, encoded.input_ids, encoded.attention_mask, None, None, None, None)
            prototypes[domain_names[i]] = torch.mean(target_embeds, dim=0)
        prototype_tensor = torch.stack(list(prototypes.values()))

        # get embeddings of unique targets and find the domain they're most similar to
        for t in uniques:
            encoded = model.tokenizer(t, is_split_into_words=True, max_length=128, padding='max_length',
                                      return_attention_mask=True, return_tensors='pt').to('cuda')
            embeds = model(True, encoded.input_ids, encoded.attention_mask, None, None, None, None)
            scores = torch.cosine_similarity(embeds, prototype_tensor)
            index = (scores == max(scores)).nonzero(as_tuple=True)[0][0]  # first [0] as tuple has one element, second [0] first occuerrence
            domain[t] = domain_names[index]

    # index domain of each target
    for k, v in domain.items():
        indices = [i for i, x in enumerate(target) if x == k]
        for i in indices:
            target[i] = v

    return target


def prompt_template(text: [str], target1: [str], target2: [str], domain: [str]):
    '''
    template each tweet by a prompt
    :param text: tweets
    :param target1: first target
    :param target2: second target
    :param domain: domains
    :return: a list of prompted texts
    '''
    prompt1, prompt2 = [], []
    for i in range(len(text)):
        text[i] = ' '.join(text[i])
        prompt1.append(
            f"The stance of text '{text[i]}' towards target '{target1[i]}' on domain '{domain[i]}' is [MASK] from the set of 'favor', 'against', 'none'."
        )
        if i < len(target2):
            prompt2.append(
                f"The stance of text '{text[i]}' towards target '{target2[i]}' on domain '{domain[i]}' is [MASK] from the set of 'favor', 'against', 'none'."
            )
    return prompt1, prompt2


def tokenization(tokenizer, prompt1: [str], prompt2: [str], y1: [int], y2: [int], batch_size: int, mode: str):
    '''
    tokenize prompts and convert them and labels to tensors
    :param tokenizer: the tokenizer for tokenizing prompts
    :param prompt1: first target prompts
    :param prompt2: second target prompts
    :param y1: labels of first target
    :param y2: labels of second target
    :param batch_size: size of mini-batch
    :param mode: phase mode
    :return: a TensorDataset and a tensor of labels
    '''
    input_ids, attention_masks, mask_position = {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}
    mak_token = tokenizer.mask_token_id
    prompt2_len = len(prompt2)
    prompt2_pad = [-1 for _ in range(512)]

    for i in range(len(prompt1)):
        encoded1 = tokenizer(prompt1[i], max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
        input_ids[1].append(encoded1.input_ids)
        attention_masks[1].append(encoded1.attention_mask)
        mask_position[1].append(input_ids[1][i].index(mak_token))

        if i < prompt2_len:
            encoded2 = tokenizer(prompt2[i], max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
            input_ids[2].append(encoded2.input_ids)
            attention_masks[2].append(encoded2.attention_mask)
            mask_position[2].append(input_ids[2][i].index(mak_token))
        else:
            input_ids[2].append(prompt2_pad)
            attention_masks[2].append(prompt2_pad)
            mask_position[2].append(-1)

    input_ids1 = torch.tensor(input_ids[1], dtype=torch.long).to('cuda')
    attention_masks1 = torch.tensor(attention_masks[1], dtype=torch.long).to('cuda')
    mask_position1 = torch.tensor(mask_position[1], dtype=torch.long).to('cuda')

    input_ids2 = torch.tensor(input_ids[2], dtype=torch.long).to('cuda')
    attention_masks2 = torch.tensor(attention_masks[2], dtype=torch.long).to('cuda')
    mask_position2 = torch.tensor(mask_position[2], dtype=torch.long).to('cuda')
    y_2 = [1 if y1[i] == y2[i] else 0 for i in range(len(y1[:len(y2)]))]
    while len(y_2) < len(y1):
        y_2.append(-1)
    y_2 = torch.tensor(y_2, dtype=torch.long).to('cuda')
    y1 = torch.tensor(y1, dtype=torch.long).to('cuda')
    
    tensor_loader = TensorDataset(input_ids1, attention_masks1, mask_position1, y1,
                                  input_ids2, attention_masks2, mask_position2, y_2)
    data_loader_eval = None
    data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
    if mode == 'train':
        data_loader_eval = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)

    return y1, y2, data_loader, data_loader_eval


def data_helper_bert(data: list, model, batch_size: int, mode: str):
    print('Loading data')
    text, tar1, y1, tar2, y2 = data[0], data[1], data[2], data[3], data[4]
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')
    domains = get_domain(model, tar1)
    prompt1, prompt2 = prompt_template(text, tar1, tar2, domains)
    y1, y2, data_loader, data_loader_eval = tokenization(model.tokenizer, prompt1, prompt2, y1, y2, batch_size, mode)

    return y1, y2, data_loader, data_loader_eval


def sep_test_set(input_data, unique_input):
    target_indices, data_list = [], []
    # find unique targets and their indices
    for unique in set(unique_input):
        target_indices.append(unique_input.index(unique))
    target_indices = sorted(target_indices)

    for i in range(len(target_indices) - 1):
        data_list.append(input_data[target_indices[i]: target_indices[i + 1]])
    data_list.append(input_data[target_indices[-1]:])

    return data_list
