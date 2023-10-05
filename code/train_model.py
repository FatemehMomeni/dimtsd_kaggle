import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import data_helper as dh
from transformers import AdamW, BertweetTokenizer, BertTokenizer
import modeling, model_eval


def run_classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="all_teacher")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
    parser.add_argument("--train_mode", type=str, default="unified", help="unified or adhoc")
    parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
    parser.add_argument("--dataset_name", type=str, default="all", help="mt,semeval,am,wtwt,covid or all-dataset")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
    parser.add_argument("--seed", type=int, help="random seed")
    args = parser.parse_args()

    random_seeds = [args.seed]
    target_word_pair = [args.input_target]
    model_select = args.model_select
    col = args.col
    train_mode = args.train_mode
    model_name = args.model_name
    dataset_name = args.dataset_name
    file = args.filename
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    alpha = args.alpha
    theta = args.theta

    # saved name of teacher predictions
    teacher = {'all': 'teacher_output_all_batch'}
    target_num = {'all': 3}
    eval_batch = {'all': True}

    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []
        for seed in random_seeds:
            print("current random seed: ", seed)
            train = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/train_mask_id.csv', encoding='ISO-8859-1')
            validation = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/val_mask_id.csv', encoding='ISO-8859-1')
            test = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test_mask_id.csv', encoding='ISO-8859-1')

            train_txt = train['prompt'].values.tolist()
            y_train = train['stance'].values.tolist()
            train_domain = train['mask'].values.tolist()

            val_txt = validation['prompt'].values.tolist()
            y_val = validation['stance'].values.tolist()
            val_domain = validation['mask'].values.tolist()

            test_txt = test['prompt'].values.tolist()
            y_test = test['stance'].values.tolist()
            test_domain = test['mask'].values.tolist()

            if model_name == 'student':
                y_train2 = torch.load('/kaggle/working/dot_product_linear_seed1.pt')
                # y_train2 = torch.load(teacher[dataset_name] + '_seed{}.pt'.format(seed))  # load teacher predictions

            num_labels = 3  # Favor, Against and None
            train = [train_txt, y_train, train_domain]
            val = [val_txt, y_val, val_domain]
            test = [test_txt, y_test, test_domain]

            # set up the random seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if model_select == 'Bertweet':
              tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
            elif model_select == 'Bert':
              tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

            # prepare for model
            x_train_all = dh.data_helper_bert(train, target_word_pair[target_index], model_select, tokenizer)
            x_val_all = dh.data_helper_bert(val, target_word_pair[target_index], model_select, tokenizer)
            x_test_all = dh.data_helper_bert(test, target_word_pair[target_index], model_select, tokenizer)

            if model_name == 'teacher':
                x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, trainloader, \
                    trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name)                
            else:
                x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, trainloader, \
                    trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name, \
                                                         y_train2=y_train2)
            x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader = \
                dh.data_loader(x_val_all, batch_size, model_select, 'val', model_name)
            x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len, testloader = \
                dh.data_loader(x_test_all, batch_size, model_select, 'test', model_name)
            
            v_labels = torch.load('/kaggle/working/dimtsd_kaggle/lv_hmask_bert.pt')
            model = modeling.stance_classifier(num_labels, model_select, v_labels, tokenizer, batch_size).cuda()
            
            # for n, p in model.named_parameters():
            #     if "bert.embeddings" in n:
            #         p.requires_grad = False
            model.requires_grad = False
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
            ]

            loss_function = nn.CrossEntropyLoss(reduction='sum')
            if model_name == 'student':
                loss_function2 = nn.KLDivLoss(reduction='sum')

            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

            sum_loss, sum_loss2 = [], []
            val_f1_average = []
            train_preds_distill, train_cls_distill = [], []
            if train_mode == "unified":
                test_f1_average = [[] for i in range(target_num[dataset_name])]

            for epoch in range(0, total_epoch):
                print('Epoch:', epoch)
                train_loss, train_loss2 = [], []
                model.train()
                if model_name == 'teacher':
                    for input_ids, seg_ids, atten_masks, target, masks in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks, masks)     
                        loss = loss_function(output1, target)
                        # loss.requires_grad = True
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                else:
                    for input_ids, seg_ids, atten_masks, target, target2, masks in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks, masks)
                        output2 = output1
                        
                        # 3. proposed AKD
                        output2 = torch.empty(output1.shape).fill_(0.).cuda()
                        for ind in range(len(target2)):
                            soft = max(F.softmax(target2[ind]))
                            if soft <= theta:
                                rrand = random.uniform(2, 3)  # parameter b1 and b2 in paper
                            elif soft < theta + 0.2 and soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here
                                rrand = random.uniform(1, 2)
                            else:
                                rrand = 1
                            target2[ind] = target2[ind] / rrand
                            output2[ind] = output1[ind] / rrand
                        target2 = F.softmax(target2)

                        loss = (1 - alpha) * loss_function(output1, target) + \
                               alpha * loss_function2(F.log_softmax(output2), target2)
                        loss2 = alpha * loss_function2(F.log_softmax(output2), target2)
                        loss.requires_grad = True
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                        train_loss2.append(loss2.item())
                    sum_loss2.append(sum(train_loss2) / len(train_txt))
                    print(sum_loss2[epoch])
                sum_loss.append(sum(train_loss) / len(train_txt))
                print(sum_loss[epoch])

                if model_name == 'teacher':
                    # train evaluation
                    model.eval()
                    train_preds = []
                    with torch.no_grad():
                        for input_ids, seg_ids, atten_masks, target, masks in trainloader_distill:
                            output1 = model(input_ids, seg_ids, atten_masks, masks)
                            train_preds.append(output1)
                        preds = torch.cat(train_preds, 0)
                        train_preds_distill.append(preds)
                        print("The size of train_preds is: ", preds.size())

                # evaluation on val set
                model.eval()
                val_preds = []
                with torch.no_grad():
                    if not eval_batch[dataset_name]:
                        pred1 = model(x_val_input_ids, x_val_atten_masks)
                    else:
                        for input_ids, seg_ids, atten_masks, target, masks in valloader:
                            pred1 = model(input_ids, seg_ids, atten_masks, masks)  # unified
                            val_preds.append(pred1)
                        pred1 = torch.cat(val_preds, 0)
                    acc, f1_average, precision, recall = model_eval.compute_f1(pred1, y_val)
                    val_f1_average.append(f1_average)

                # evaluation on test set
                if train_mode == "unified":
                    x_test_len_list = dh.sep_test_set(x_test_len, dataset_name)
                    y_test_list = dh.sep_test_set(y_test, dataset_name)
                    x_test_input_ids_list = dh.sep_test_set(x_test_input_ids, dataset_name)
                    x_test_seg_ids_list = dh.sep_test_set(x_test_seg_ids, dataset_name)
                    x_test_atten_masks_list = dh.sep_test_set(x_test_atten_masks, dataset_name)

                with torch.no_grad():
                    if eval_batch[dataset_name]:
                        test_preds = []
                        for input_ids, seg_ids, atten_masks, target, masks in testloader:
                            pred1 = model(input_ids, seg_ids, atten_masks, masks)
                            test_preds.append(pred1)
                        pred1 = torch.cat(test_preds, 0)
                        if train_mode == "unified":
                            pred1_list = dh.sep_test_set(pred1, dataset_name)

                    test_preds = []
                    for ind in range(len(y_test_list)):
                        if not eval_batch[dataset_name]:
                            pred1 = model(x_test_input_ids_list[ind], x_test_seg_ids_list[ind], \
                                          x_test_atten_masks_list[ind], x_test_len_list[ind])
                        else:
                            pred1 = pred1_list[ind]
                        test_preds.append(pred1)                        
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1, y_test_list[ind])
                        test_f1_average[ind].append(f1_average)
                    
            # model that performs best on the dev set is evaluated on the test set
            best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
            best_result.append([f1[best_epoch] for f1 in test_f1_average])

            if model_name == 'teacher':
                best_preds = train_preds_distill[best_epoch]
                torch.save(best_preds, 'dot_product_linear_seed{}.pt'.format(seed))

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
