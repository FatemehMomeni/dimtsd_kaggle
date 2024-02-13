import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import gc
from transformers import AdamW
import pandas as pd
import data_helper as dh
import modeling, model_eval


def run_classifier():

    random_seeds = [1,2,4,5,9,10]
    target_word_pair = ['all']
    model_select = 'Bert'
    col = 'Stance1'
    train_mode = 'unified'
    model_name = 'teacher'
    dataset_name = 'all'
    file = 'Stance_All_Five_Dataset'
    lr = 2e-5
    batch_size = 32
    total_epoch = 5
    dropout = 0.
    alpha = 0.7
    theta = 0.6
    
    # saved name of teacher predictions
    teacher = {'all':'teacher_output_all_batch',}
    target_num = {'all': 3}
    eval_batch = {'all': True}

    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []
        for seed in random_seeds:    
            print("current random seed: ", seed)

            if train_mode == "unified":
                train = pd.read_csv('/content/dimtsd_kaggle/dataset/train_domain.csv', encoding='ISO-8859-1')
                # train = train.sample(frac=0.6)
                validation = pd.read_csv('/content/dimtsd_kaggle/dataset/val_domain.csv', encoding='ISO-8859-1')
                test = pd.read_csv('/content/dimtsd_kaggle/dataset/test_domain.csv', encoding='ISO-8859-1')

                x_train = train['Tweet'].values.tolist()
                x_train_target = train['Target'].values.tolist()
                y_train = train['Stance'].values.tolist()
                x_train_d = train['domain'].values.tolist()

                x_val = validation['Tweet'].values.tolist()
                x_val_target = validation['Target'].values.tolist()
                y_val = validation['Stance'].values.tolist()
                x_val_d = validation['domain'].values.tolist()

                x_test = test['Tweet'].values.tolist()
                x_test_target = test['Target'].values.tolist()
                y_test = test['Stance'].values.tolist()            
                x_test_d = test['domain'].values.tolist()

            if model_name == 'student':
                y_train2 = torch.load(teacher[dataset_name]+'_seed{}.pt'.format(seed))  # load teacher predictions

            num_labels = 3  # Favor, Against and None
            x_train_all = [x_train,y_train,x_train_target,x_train_d]
            x_val_all = [x_val,y_val,x_val_target,x_val_d]
            x_test_all = [x_test,y_test,x_test_target,x_test_d]
            
            # set up the random seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 

            # prepare for model
            x_train_all,x_val_all,x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,\
                                        target_word_pair[target_index],model_select)
            if model_name == 'teacher':
                y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name)
            else:
                y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name,\
                                                       y_train2=y_train2)
            y_val, valloader =  dh.data_loader(x_val_all, batch_size, model_select, 'val',model_name)                            
            y_test, testloader = dh.data_loader(x_test_all, batch_size, model_select, 'test',model_name)

            model = modeling.stance_classifier(num_labels,model_select).cuda()

            for n,p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
                    
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
                ]
            
            loss_function = nn.CrossEntropyLoss(reduction='sum')
            if model_name == 'student':
                loss_function2 = nn.KLDivLoss(reduction='sum')
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

            sum_loss, sum_loss2 = [], []
            val_f1_average = []
            train_preds_distill,train_cls_distill = [], []
            test_f1_average = [[] for i in range(target_num[dataset_name])]
            
            for epoch in range(0, total_epoch):
                print('Epoch:', epoch)
                train_loss, train_loss2 = [], []
                model.train()
                if model_name == 'teacher':
                    for input_ids,seg_ids,atten_masks,target in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks)
                        loss = loss_function(output1, target)
                        # loss.requires_grad = True
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                else:
                    for input_ids,seg_ids,atten_masks,target,target2 in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks)
                        output2 = output1

                        # 3. proposed AKD
                        output2 = torch.empty(output1.shape).fill_(0.).cuda()
                        for ind in range(len(target2)):
                            soft = max(F.softmax(target2[ind]))
                            if soft <= theta:
                                rrand = random.uniform(2,3)  # parameter b1 and b2 in paper
                            elif soft < theta+0.2 and soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here 
                                rrand = random.uniform(1,2)
                            else:
                                rrand = 1
                            target2[ind] = target2[ind]/rrand
                            output2[ind] = output1[ind]/rrand
                        target2 = F.softmax(target2)
                            
                        loss = (1-alpha)*loss_function(output1, target) + \
                               alpha*loss_function2(F.log_softmax(output2), target2)
                        loss2 = alpha*loss_function2(F.log_softmax(output2), target2)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                        train_loss2.append(loss2.item())
                    sum_loss2.append(sum(train_loss2)/len(x_train))  
                    print(sum_loss2[epoch])
                sum_loss.append(sum(train_loss)/len(x_train))  
                print(sum_loss[epoch])

                if model_name == 'teacher':
                    # train evaluation
                    model.eval()
                    train_preds = []
                    with torch.no_grad():
                        for input_ids,seg_ids,atten_masks,target in trainloader_distill:
                            output1 = model(input_ids, seg_ids, atten_masks)
                            train_preds.append(output1)
                        preds = torch.cat(train_preds, 0)
                        train_preds_distill.append(preds)
                        print("The size of train_preds is: ", preds.size())

                # evaluation on val set 
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for input_ids,seg_ids,atten_masks,target in valloader:
                        pred1 = model(input_ids, seg_ids, atten_masks) # unified
                        val_preds.append(pred1)
                    pred1 = torch.cat(val_preds, 0)
                    acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_val)
                    val_f1_average.append(f1_average)

                # evaluation on test set
                y_test_list = dh.sep_test_set(y_test,dataset_name)
                
                with torch.no_grad():
                    if eval_batch[dataset_name]:
                        test_preds = []
                        for input_ids,seg_ids,atten_masks,target in testloader:
                            pred1 = model(input_ids, seg_ids, atten_masks)
                            test_preds.append(pred1)
                        pred1 = torch.cat(test_preds, 0)
                        pred1_list = dh.sep_test_set(pred1,dataset_name)
                        
                    test_preds = []
                    for ind in range(len(y_test_list)):                        
                        pred1 = pred1_list[ind]
                        test_preds.append(pred1)
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
                        test_f1_average[ind].append(f1_average)

            # model that performs best on the dev set is evaluated on the test set
            best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
            best_result.append([f1[best_epoch] for f1 in test_f1_average])
            
            if model_name == 'teacher':
                best_preds = train_preds_distill[best_epoch]
                torch.save(best_preds, teacher[dataset_name]+'_seed{}.pt'.format(seed))

            print("******************************************")
            print("dev results with seed {} on all epochs".format(seed))
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])
            print("******************************************")
            print("test results with seed {} on all epochs".format(seed))
            print(test_f1_average)
            print("******************************************")
            print(max(best_result))
            print(best_result)

            
if __name__ == "__main__":
    run_classifier()
