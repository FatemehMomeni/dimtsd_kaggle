# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import random
# import numpy as np
# import pandas as pd
# import argparse
# import json
# import gc
# import error_analysis_data as dh
# from transformers import AdamW
# import error_analysis_model, model_eval


# def run_classifier():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_target", type=str, default="all_teacher")
#     parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
#     parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
#     parser.add_argument("--train_mode", type=str, default="unified", help="unified or adhoc")
#     parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
#     parser.add_argument("--dataset_name", type=str, default="all", help="mt,semeval,am,wtwt,covid or all-dataset")
#     parser.add_argument("--filename", type=str)
#     parser.add_argument("--lr", type=float, default=2e-5)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--dropout", type=float, default=0.)
#     parser.add_argument("--alpha", type=float, default=0.7)
#     parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
#     parser.add_argument("--seed", type=int, help="random seed")
#     args = parser.parse_args()

#     random_seeds = [args.seed]
#     target_word_pair = [args.input_target]
#     model_select = args.model_select
#     col = args.col
#     train_mode = args.train_mode
#     model_name = args.model_name
#     dataset_name = args.dataset_name
#     file = args.filename
#     lr = args.lr
#     batch_size = args.batch_size
#     total_epoch = args.epochs
#     dropout = args.dropout
#     alpha = args.alpha
#     theta = args.theta
    
#     teacher = {'all': 'teacher_output_all_batch'}
#     target_num = {'all': 3}
#     eval_batch = {'all': True}

#     for target_index in range(len(target_word_pair)):
#         best_result, best_val = [], []
#         for seed in random_seeds:    
#             print("current random seed: ", seed)
#             train = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/train.csv', encoding='ISO-8859-1')
#             validation = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/validation.csv', encoding='ISO-8859-1')
#             test = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test.csv', encoding='ISO-8859-1')

#             x_train = train['prompt'].values.tolist()
#             y_train = train['stance'].values.tolist()

#             x_val = validation['prompt'].values.tolist()
#             y_val = validation['stance'].values.tolist()

#             x_test = test['prompt'].values.tolist()
#             y_test = test['stance'].values.tolist()
#             x_temp = x_test[:10238]
#             x_test[:1966] = x_test[10238:12204]
#             x_test[1966:12204] = x_temp
#             y_temp = y_test[:10238]
#             y_test[:1966] = y_test[10238:12204]
#             y_test[1966:12204] = y_temp

#             if model_name == 'student':
#                 y_train2 = torch.load('/kaggle/working/pro_extend_no_tars_error_seed1.pt')

#             num_labels = 3  # Favor, Against and None
#             x_train_all = [x_train, y_train]
#             x_val_all = [x_val, y_val]
#             x_test_all = [x_test, y_test]
            
#             random.seed(seed)
#             np.random.seed(seed)
#             torch.manual_seed(seed) 

#             x_train_all, x_val_all, x_test_all, tokenizer = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,model_select)
            
#             if model_name == 'teacher':
#                 y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name)
#             else:
#                 y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name,\
#                                                        y_train2=y_train2)
#             y_val, valloader = dh.data_loader(x_val_all, batch_size, model_select, 'val',model_name)                            
#             y_test, testloader = dh.data_loader(x_test_all, batch_size, model_select, 'test',model_name)

#             labels = list()
#             for l in ['against','none','favor']:
#                 labels.append(torch.load(f"/kaggle/working/dimtsd_kaggle/{l}_lv_bert.pt"))
#             model = error_analysis_model.stance_classifier(num_labels,model_select, labels).cuda()

#             for n,p in model.named_parameters():
#                 if "bert.embeddings" in n:
#                     p.requires_grad = False
                    
#             optimizer_grouped_parameters = [
#                 {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
#                 {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
#                 {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
#                 {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
#                 ]
            
#             loss_function = nn.CrossEntropyLoss(reduction='sum')
#             if model_name == 'student':
#                 loss_function2 = nn.KLDivLoss(reduction='sum')
            
#             optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

#             sum_loss, sum_loss2 = [], []
#             val_f1_average = []
#             train_preds_distill,train_cls_distill = [], []
#             if train_mode == "unified":
#                 test_f1_average = [[] for i in range(target_num[dataset_name])]            
            
#             for epoch in range(0, total_epoch):
#                 print('Epoch:', epoch)
#                 train_loss, train_loss2 = [], []
#                 model.train()                
#                 if model_name == 'teacher':
#                     for input_ids,seg_ids,atten_masks,mask,target in trainloader:                                                       
#                         optimizer.zero_grad()
#                         output1 = model(input_ids, seg_ids, atten_masks,mask)                        
#                         loss = loss_function(output1, target)
#                         loss.backward()
#                         nn.utils.clip_grad_norm_(model.parameters(), 1)
#                         optimizer.step()
#                         train_loss.append(loss.item())
#                 else:
#                     for input_ids,seg_ids,atten_masks,mask,target,target2 in trainloader:
#                         optimizer.zero_grad()
#                         output1 = model(input_ids, seg_ids, atten_masks,mask)
#                         output2 = output1

#                         # 3. proposed AKD
#                         output2 = torch.empty(output1.shape).fill_(0.).cuda()
#                         for ind in range(len(target2)):
#                             soft = max(F.softmax(target2[ind]))
#                             if soft <= theta:
#                                 rrand = random.uniform(2,3)  # parameter b1 and b2 in paper
#                             elif soft < theta+0.2 and soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here 
#                                 rrand = random.uniform(1,2)
#                             else:
#                                 rrand = 1
#                             target2[ind] = target2[ind]/rrand
#                             output2[ind] = output1[ind]/rrand
#                         target2 = F.softmax(target2)
                            
#                         loss = (1-alpha)*loss_function(output1, target) + \
#                                alpha*loss_function2(F.log_softmax(output2), target2)
#                         loss2 = alpha*loss_function2(F.log_softmax(output2), target2)
#                         loss.backward()
#                         nn.utils.clip_grad_norm_(model.parameters(), 1)
#                         optimizer.step()
#                         train_loss.append(loss.item())
#                         train_loss2.append(loss2.item())
#                     sum_loss2.append(sum(train_loss2)/len(x_train))  
#                     print(sum_loss2[epoch])
#                 sum_loss.append(sum(train_loss)/len(x_train))  
#                 print(sum_loss[epoch])

#                 if model_name == 'teacher':
#                     # train evaluation
#                     model.eval()
#                     train_preds = []                    
#                     with torch.no_grad():
#                         for input_ids,seg_ids,atten_masks,mask,target in trainloader_distill:
#                             output1 = model(input_ids, seg_ids, atten_masks,mask)
#                             train_preds.append(output1)                         
#                         preds = torch.cat(train_preds, 0)
#                         train_preds_distill.append(preds)
#                         print("The size of train_preds is: ", preds.size())

#                 # evaluation on val set 
#                 model.eval()
#                 val_preds = []                
#                 with torch.no_grad():                    
#                     for input_ids,seg_ids,atten_masks,mask,target in valloader:
#                         pred1 = model(input_ids, seg_ids, atten_masks,mask) # unified
#                         val_preds.append(pred1)                            
#                     pred1 = torch.cat(val_preds, 0)                        
#                     acc, f1_average, precision, recall, _ = model_eval.compute_f1(pred1,y_val, False)
#                     val_f1_average.append(f1_average)

#                 # evaluation on test set
#                 if train_mode == "unified":
#                     y_test_list = dh.sep_test_set(y_test)
                    
#                 flag = True
#                 with torch.no_grad():
#                     if eval_batch[dataset_name]:
#                         test_preds = []
#                         for input_ids,seg_ids,atten_masks,mask,target in testloader:
#                             pred1 = model(input_ids, seg_ids, atten_masks,mask)
#                             test_preds.append(pred1)
#                             if flag:
#                               error_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#                               error_stance = target.tolist()
#                               error_predict = pred1   
#                               flag = False
#                             else:
#                               break
#                         pred1 = torch.cat(test_preds, 0)
#                         rounded_preds = torch.nn.functional.softmax(error_predict)
#                         _, indices = torch.max(rounded_preds, 1)
#                         indices = indices.tolist()
#                         error_df = pd.DataFrame({'input': error_input, 'stance': error_stance, 'prediction': indices})
#                         error_df.to_csv('pro_mask_domain_error.csv', index=False)                    
#                         pred1_list = dh.sep_test_set(pred1)
                        
#                     test_preds = []
#                     for ind in range(len(y_test_list)):                        
#                         pred1 = pred1_list[ind]
#                         test_preds.append(pred1)
#                         acc, f1_average, precision, recall, confusion = model_eval.compute_f1(pred1,y_test_list[ind], True)
#                         test_f1_average[ind].append(f1_average)
#                         print('-'*20, '\n', confusion)

#             # model that performs best on the dev set is evaluated on the test set
#             best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
#             best_result.append([f1[best_epoch] for f1 in test_f1_average])
            
#             if model_name == 'teacher':
#                 best_preds = train_preds_distill[best_epoch]
#                 torch.save(best_preds, 'pro_extend_no_tars_error_seed{}.pt'.format(seed))

#             print("******************************************")
#             print("dev results with seed {} on all epochs".format(seed))
#             print(val_f1_average)
#             best_val.append(val_f1_average[best_epoch])
#             print("******************************************")
#             print("test results with seed {} on all epochs".format(seed))
#             print(test_f1_average)
#             print("******************************************")
#             print(max(best_result))
#             print(best_result)


# if __name__ == "__main__":
#     run_classifier()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import gc
import error_analysis_data as dh
from transformers import AdamW, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import error_analysis_model, model_eval


def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
    parser.add_argument("--seed", type=int, help="random seed")
    args = parser.parse_args()

    random_seeds = [args.seed]
    model_select = args.model_select
    model_name = args.model_name
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    alpha = args.alpha
    theta = args.theta
    
    teacher = 'teacher_output_all_batch'
    target_num = 3
    eval_batch = True
    labels_map = {0: 'against', 1:'none', 2: 'favor'}

    best_result, best_val = [], []
    for seed in random_seeds:    
      print("current random seed: ", seed)
      train = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/train_domain.csv', encoding='ISO-8859-1')
      validation = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/val_domain.csv', encoding='ISO-8859-1')
      test = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test_domain.csv', encoding='ISO-8859-1')
      
      x_train = train['Tweet'].values.tolist()
      x_train_tar = train['Target'].values.tolist()
      y_train = train['Stance'].values.tolist()           
      y_train_str = list(map(lambda x: labels_map[x], y_train)) 

      x_val = validation['Tweet'].values.tolist()
      x_val_tar = validation['Target'].values.tolist()
      y_val = validation['Stance'].values.tolist()
      y_val_str = list(map(lambda x: labels_map[x], y_val))
      
      x_test = test['Tweet'].values.tolist()
      x_test_tar = test['Target'].values.tolist()
      y_test = test['Stance'].values.tolist()
      y_test_str = list(map(lambda x: labels_map[x], y_test))

      if model_name == 'student':
        y_train2 = torch.load('/kaggle/working/flant5_seed{}.pt'.format(seed))

      num_labels = 3  # Favor, Against and None
      x_train_all = [x_train, x_train_tar, y_train, y_train_str]
      x_val_all = [x_val, x_val_tar, y_val, y_val_str]
      x_test_all = [x_test, x_test_tar, y_test, y_test_str]
      
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed) 

      x_train_all, x_val_all, x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,model_select)
      
      if model_name == 'teacher':
        y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name)
      else:
        y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name, y_train2=y_train2)
      y_val, valloader = dh.data_loader(x_val_all, batch_size, model_select, 'val',model_name)                            
      y_test, testloader = dh.data_loader(x_test_all, batch_size, model_select, 'test',model_name)

      # label_vectors = list()
      # for l in ['against', 'none', 'favor']:
      #   label_vectors.append(torch.load(f"/content/dimtsd_kaggle/{l}_lv_bert.pt"))
      # model = error_analysis_model.stance_classifier(num_labels,model_select).cuda()  
      # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to('cuda:0')
      model = error_analysis_model.stance_classifier(num_labels,model_select).to('cuda:0')

      # for n,p in model.named_parameters():
      #   if "bert.embeddings" in n:
      #     p.requires_grad = False
              
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
      test_f1_average = [[] for i in range(target_num)]           
      
      for epoch in range(0, total_epoch):
        print('Epoch:', epoch)
        train_loss, train_loss2 = [], []
        # model.train()                
        # if model_name == 'teacher':
        #   for input_ids, attention_mask, label, target in trainloader:
        #     optimizer.zero_grad()
        #     loss = model(input_ids, attention_mask, label, True)
        #     # loss = loss_function(output1, target)            
        #     loss.item()
        #     # loss.backward()
        #     nn.utils.clip_grad_norm_(model.parameters(), 1)
        #     optimizer.step()
        #     train_loss.append(loss.item())
        # else:
        #   for input_ids, attention_mask, label, target, target2 in trainloader:
        #     optimizer.zero_grad()
        #     output1 = model(input_ids, attention_mask, label, False)
        #     output2 = output1

        #     # 3. proposed AKD
        #     output2 = torch.empty(output1.shape).fill_(0.).cuda()
        #     for ind in range(len(target2)):
        #       soft = max(F.softmax(target2[ind]))
        #       if soft <= theta:
        #         rrand = random.uniform(2,3)  # parameter b1 and b2 in paper
        #       elif soft < theta+0.2 and soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here 
        #         rrand = random.uniform(1,2)
        #       else:
        #         rrand = 1
        #       target2[ind] = target2[ind]/rrand
        #       output2[ind] = output1[ind]/rrand
        #     target2 = F.softmax(target2)
                
        #     loss = (1-alpha)*loss_function(output1, target) + alpha*loss_function2(F.log_softmax(output2), target2)
        #     loss2 = alpha*loss_function2(F.log_softmax(output2), target2)
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(model.parameters(), 1)
        #     optimizer.step()
        #     train_loss.append(loss.item())
        #     train_loss2.append(loss2.item())
        #   sum_loss2.append(sum(train_loss2) / len(x_train))  
        #   print(sum_loss2[epoch])
        # sum_loss.append(sum(train_loss) / len(x_train))  
        # print(sum_loss[epoch])
        
        # if model_name == 'teacher':
        #   # train evaluation
        #   model.eval()
        #   train_preds = []
        #   with torch.no_grad():
        #     for input_ids, attention_mask, label, target in trainloader_distill:
        #       output1 = model(input_ids, attention_mask, label, False)
        #       if pred1 in labels_map.values():
        #         prediction = list(labels_map.keys())[list(labels_map.values()).index(pred1)]
        #       else:
        #         prediction = 1
        #       train_preds.append(prediction)
        #     preds = torch.cat(train_preds, 0)
        #     train_preds_distill.append(preds)
        #     print("The size of train_preds is: ", preds.size())
                    
        # evaluation on val set 
        # model.eval()
        val_preds = []
        with torch.no_grad():            
          for input_ids, attention_mask, label, target in valloader:
            pred1 = model(input_ids, attention_mask, label, True)
            # if pred1 in labels_map.values():
            #   prediction = list(labels_map.keys())[list(labels_map.values()).index(pred1)]
            # else:
            #   prediction = 1
            pred1.item()
            # val_preds.append(pred1)
          # pred1 = torch.cat(val_preds, 0)
          # acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_val)
          # val_f1_average.append(f1_average)
                
        # evaluation on test set
        y_test_list = dh.sep_test_set(y_test)
        
        with torch.no_grad():
          test_preds = []
          for input_ids, attention_mask, label, target in testloader:
            pred1 = model(input_ids, attention_mask, label, False) #, do_sample=False) # disable sampling to test if batching affects output
            if pred1 in labels_map.values():
              prediction = list(labels_map.keys())[list(labels_map.values()).index(pred1)]
            else:
              prediction = 1
            test_preds.append(prediction)
          pred1 = torch.cat(test_preds, 0)          
          pred1_list_a = dh.sep_test_set(pred1)          
                
          test_preds = []
          for ind in range(len(y_test_list)):              
            pred1 = pred1_list_a[ind]
            test_preds.append(pred1)
            acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
            test_f1_average[ind].append(f1_average)

      # model that performs best on the dev set is evaluated on the test set
      best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
      best_result.append([f1[best_epoch] for f1 in test_f1_average])
      
      if model_name == 'teacher':
        best_preds = train_preds_distill[best_epoch]
        torch.save(best_preds, 'flant5_seed{}.pt'.format(seed))

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
