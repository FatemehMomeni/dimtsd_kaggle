import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
import numpy as np
import json
from transformers import AdamW
import pandas as pd
import preprocessing as pp
import data_helper as dh
import modeling, model_eval


def run_classifier():

  # get inputs
  parser = argparse.ArgumentParser()
  parser.add_argument("--label_column", type=str, default='Stance 1')
  parser.add_argument("--learning_rate", type=float, default=2e-5)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=5)
  args = parser.parse_args()
  
  column_name = args.label_column
  lr = args.learning_rate
  batch_size = args.batch_size
  total_epoch = args.epochs
  random_seeds = [1,2,4,5,9,10]
  
  # create normalization dictionary for preprocessing
  with open("./noslang_data.json", "r") as f:
    data1 = json.load(f)
  data2 = {}
  with open("./emnlp_dict.txt","r") as f:
    lines = f.readlines()
    for line in lines:
      row = line.split('\t')
      data2[row[0]] = row[1].rstrip()
  normalization_dict = {**data1,**data2}


  best_result, best_val = [], []
  for seed in random_seeds:    
    print("current random seed: ", seed)

    train_file = '../datasets/train.csv'
    eval_file = '../datasets/val.csv'
    test_file = '../dataset/test.csv'
    # train = train.sample(frac=0.5)

    # preprocess data
    x_train, x_train_target, y_train = pp.clean_all(train_file, column_name, normalization_dict)
    x_val, x_val_target, y_val = pp.clean_all(eval_file, column_name, normalization_dict)
    x_test, x_test_target, y_test = pp.clean_all(test_file, column_name, normalization_dict)
    y_unique_test = y_test # save list type of y_test for separating test set in sep_test_set function

    # TO DO
    # topic modeling for obtaining domain

    num_labels = len(set(y_val))
    x_train_all = [x_train, x_train_target, x_train_domain, y_train]
    x_val_all = [x_val, x_val_target, x_val_domain, y_val]
    x_test_all = [x_test, x_test_target, x_test_domain, y_test]
      
    # set up the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 

    # prepare for model
    x_train_all, x_val_all, x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all)
    
    y_train, trainloader, trainloader_distill = dh.data_loader(x_train_all, batch_size, 'train')
    y_val, valloader =  dh.data_loader(x_val_all, batch_size, 'val')
    y_test, testloader = dh.data_loader(x_test_all, batch_size, 'test')

    model = modeling.stance_classifier(num_labels).to('cuda')

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    sum_loss, sum_loss2 = [], []
    val_f1_average = []
    train_preds_distill,train_cls_distill = [], []
    test_f1_average = [[] for i in range(len(set(x_test_target)))]
    
    for epoch in range(total_epoch):
      print('Epoch:', epoch)
      train_loss = list()
      model.train()
      for input_ids, token_type_ids, attention_mask, target in trainloader:
        optimizer.zero_grad()
        output = model(input_ids, token_type_ids, attention_mask)
        loss = loss_function(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        train_loss.append(loss.item())
      
      sum_loss.append(sum(train_loss)/len(x_train))  
      print(sum_loss[epoch]) # summation of loss in this epoch

      # train evaluation
      model.eval()
      train_preds = list()
      with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, target in trainloader_distill:
          output = model(input_ids, token_type_ids, attention_mask)
          train_preds.append(output)
        preds = torch.cat(train_preds, 0)
        train_preds_distill.append(preds)
        print("The size of train predictions is: ", preds.size())

      # evaluation on val set           
      val_preds = list()
      with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, target in valloader:
          prediction = model(input_ids, token_type_ids, attention_mask)
          val_preds.append(prediction)
        prediction = torch.cat(val_preds, 0)
        acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y_val)
        val_f1_average.append(f1_average)

      # evaluation on test set
      y_test_list = dh.sep_test_set(y_test, y_unique_test)
      with torch.no_grad():
        test_preds = list()
        for input_ids, token_type_ids, attention_mask, target in testloader:
          prediction = model(input_ids, token_type_ids, attention_mask)
          test_preds.append(prediction)
        prediction = torch.cat(test_preds, 0)
        prediction_list = dh.sep_test_set(prediction, y_unique_test)
            
        test_preds = list()
        for ind in range(len(y_test_list)):                        
          prediction = prediction_list[ind]
          test_preds.append(prediction)
          acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y_test_list[ind])
          test_f1_average[ind].append(f1_average)

    # model that performs best on the eval set is evaluated on the test set
    best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
    best_result.append([f1[best_epoch] for f1 in test_f1_average])
    
    best_preds = train_preds_distill[best_epoch]
    torch.save(best_preds, f'mtsd_seed{seed}.pt')

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
