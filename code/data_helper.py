import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

def convert_data_to_ids(tokenizer, text):    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for txt in text:    
        encoded_dict = tokenizer.encode_plus(txt, add_special_tokens = True,
                            max_length = 128, padding = 'max_length',
                            return_attention_mask = True, truncation = True)
    
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))
    
    return input_ids, seg_ids, attention_masks, sent_len


def data_helper_bert(x_train_all, x_val_all, x_test_all, main_task_name, model_select):    
    print('Loading data')
    
    x_train, y_train = x_train_all[0], x_train_all[1]
    x_val, y_val = x_val_all[0], x_val_all[1]
    x_test, y_test = x_test_all[0], x_test_all[1]
    print("Length of original x_train: %d"%(len(x_train)))
    print("Length of original x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
    print("Length of original x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, mask_token='[MASK]')
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')

    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = convert_data_to_ids(tokenizer, x_train)
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = convert_data_to_ids(tokenizer, x_val)
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = convert_data_to_ids(tokenizer, x_test)
    
    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len]
    
    print(len(x_train), sum(y_train))
    print("Length of final x_train: %d"%(len(x_train)))
    
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):    
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()

    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,y2)
    else:
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader


def sep_test_set(input_data, dataset_name):
    data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    return data_list

