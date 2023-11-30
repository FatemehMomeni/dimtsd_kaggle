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
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer


def run_classifier():

  labels_map = {0: 'against', 1:'none', 2: 'favor'}
  
  # train = pd.read_csv('/content/dimtsd_kaggle/dataset/train_domain.csv', encoding='ISO-8859-1')
  # validation = pd.read_csv('/content/dimtsd_kaggle/dataset/val_domain.csv', encoding='ISO-8859-1')
  test = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test_domain.csv', encoding='ISO-8859-1')
  
  # x_train = train['Tweet'].values.tolist()
  # x_train_tar = train['Target'].values.tolist()
  # y_train = train['Stance'].values.tolist()           
  # y_train_str = list(map(lambda x: labels_map[x], y_train)) 

  # x_val = validation['Tweet'].values.tolist()
  # x_val_tar = validation['Target'].values.tolist()
  # y_val = validation['Stance'].values.tolist()
  # y_val_str = list(map(lambda x: labels_map[x], y_val))
  
  x_test = test['Tweet'].values.tolist()
  x_test_tar = test['Target'].values.tolist()
  y_test = test['Stance'].values.tolist()
  y_test_str = list(map(lambda x: labels_map[x], y_test))

  model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", load_in_8bit=True)
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1") # Mistral-7B-v0.1

  print('after calling model and tokenizer')

  correct = 0
  wrong = pd.DataFrame(columns=['tweet', 'truth', 'prediction'])
  for i in range(len(x_test)):
    prompt = f"Given the text '{x_test[i]}' and the target '{x_test_tar[i]}', classify the stance of the text towards the target. Stance options are: favor, against, none. The stance is "
    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    # model.to('cuda')
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    prediction = tokenizer.batch_decode(generated_ids)[0]
    if prediction == y_test_str[i]:
      correct += 1
    else:
      wrong = wrong.append({'tweet': x_test[i], 'truth': y_test_str[i], 'prediction': prediction})

  print('number of correct predictions: ', correct)
  print('number of wrong predictions: ', len(wrong))
  pd.to_csv(wrong, index=False)


if __name__ == "__main__":
    run_classifier()
