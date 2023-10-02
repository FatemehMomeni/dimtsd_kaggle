import torch
from torch.utils.data import TensorDataset, DataLoader


def data_helper_bert(data, main_task_name, model_select, tokenizer):
    print('Loading data')

    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for i in range(len(data[0])):
        encoded_dict = tokenizer.encode_plus(data[0][i], add_special_tokens=True,
                                             max_length=512, padding='max_length',
                                             return_attention_mask=True, truncation=True, )
    
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))

    x_all = [input_ids, seg_ids, attention_masks, data[1], sent_len, data[2]]    
    return x_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()
    masks = torch.tensor(x_all[5], dtype=torch.long).cuda()
    
    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, y2, masks)       
    else:
        tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, masks)
       
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

