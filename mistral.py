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
# from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer


# def run_classifier():

#   labels_map = {0: 'against', 1:'none', 2: 'favor'}
  
#   # train = pd.read_csv('/content/dimtsd_kaggle/dataset/train_domain.csv', encoding='ISO-8859-1')
#   # validation = pd.read_csv('/content/dimtsd_kaggle/dataset/val_domain.csv', encoding='ISO-8859-1')
#   test = pd.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test_domain.csv', encoding='ISO-8859-1')
  
#   # x_train = train['Tweet'].values.tolist()
#   # x_train_tar = train['Target'].values.tolist()
#   # y_train = train['Stance'].values.tolist()           
#   # y_train_str = list(map(lambda x: labels_map[x], y_train)) 

#   # x_val = validation['Tweet'].values.tolist()
#   # x_val_tar = validation['Target'].values.tolist()
#   # y_val = validation['Stance'].values.tolist()
#   # y_val_str = list(map(lambda x: labels_map[x], y_val))
  
#   x_test = test['Tweet'].values.tolist()
#   x_test_tar = test['Target'].values.tolist()
#   y_test = test['Stance'].values.tolist()
#   y_test_str = list(map(lambda x: labels_map[x], y_test))

#   model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", load_in_8bit=True)
#   tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1") # Mistral-7B-v0.1

#   print('after calling model and tokenizer')

#   correct = 0
#   wrong = {'tweet': list(), 'truth': list(), 'prediction': list()}
#   for i in range(len(x_test)):
#     prompt = f"Given the text '{x_test[i]}' and the target '{x_test_tar[i]}', classify the stance of the text towards the target. Stance options are: favor, against, none. The stance is "
#     model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
#     # model.to('cuda')
#     generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
#     prediction = tokenizer.batch_decode(generated_ids)[0]
#     if prediction == y_test_str[i]:
#       correct += 1
#     else:
#       wrong['tweet'].append(x_test[i])
#       wrong['truth'].append(y_test_str[i])
#       wrong['prediction'].append(prediction)

#   wrong_df = pd.DataFrame(wrong)
#   print('number of correct predictions: ', correct)
#   print('number of wrong predictions: ', len(wrong_df))
#   pd.to_csv(wrong_df, index=False)


# if __name__ == "__main__":
#     run_classifier()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, platform, warnings, pandas
from datasets import load_dataset, Dataset
from trl import SFTTrainer


#Use a sharded model to fine-tune in the free version of Google Colab.
base_model = "bn22/Mistral-7B-Instruct-v0.1-sharded" # "mistralai/Mistral-7B-v0.1"
new_model = "mistral_7b"

# Loading a Gath_baize dataset
train = pandas.read_csv('/kaggle/working/dimtsd_kaggle/dataset/train_domain.csv', encoding='ISO-8859-1')
val = pandas.read_csv('/kaggle/working/dimtsd_kaggle/dataset/val_domain.csv', encoding='ISO-8859-1')
labels_map = {0: 'against', 1:'none', 2: 'favor'}
y_train = train['Stance'].values.tolist()
y_train_str = list(map(lambda x: labels_map[x], y_train))
y_val = val['Stance'].values.tolist()
y_val_str = list(map(lambda x: labels_map[x], y_val))
for i in range(len(train)):
  train.loc[i, 'Tweet'] = f"<s>[INST] Given the text '{train.loc[i, 'Tweet']}' and the target '{train.loc[i, 'Target']}', classify the stance of the text towards the target. Stance options are: favor, against, none. The stance is [/INST] ```stance\n{y_train_str[i]}```</s>"
for i in range(len(val)):
  val.loc[i, 'Tweet'] = f"<s>[INST] Given the text '{val.loc[i, 'Tweet']}' and the target '{val.loc[i, 'Target']}', classify the stance of the text towards the target. Stance options are: favor, against, none. The stance is [/INST] ```stance\n{y_val_str[i]}```</s>"
train.drop(columns=['Target', 'Stance', 'domain'], axis=1, inplace=True)
train_ds = Dataset.from_pandas(train)
val.drop(columns=['Target', 'Stance', 'domain'], axis=1, inplace=True)
val_ds = Dataset.from_pandas(val)

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(load_in_4bit= True, bnb_4bit_quant_type= "nf4", bnb_4bit_compute_dtype= torch.bfloat16, bnb_4bit_use_double_quant= False,)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"": 0})
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"])
model = get_peft_model(model, peft_config).to('cuda')

# Training Arguments
training_arguments = TrainingArguments(output_dir= "/content/outputs", num_train_epochs= 1, per_device_train_batch_size= 10,
    gradient_accumulation_steps= 2, optim = "paged_adamw_8bit", save_steps= 5000, logging_steps= 30, learning_rate= 2e-5,
    weight_decay= 0.001, fp16= False, bf16= False, max_grad_norm= 0.3, max_steps= -1, warmup_ratio= 0.3, group_by_length= True,
    lr_scheduler_type= "constant",)
# Setting sft parameters
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft_config, max_seq_length= None,
    dataset_text_field="Tweet", tokenizer=tokenizer, args=training_arguments, packing= False,)
trainer.train()
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()


test = pandas.read_csv('/kaggle/working/dimtsd_kaggle/dataset/test_domain.csv', encoding='ISO-8859-1')
y_test = test['Stance'].values.tolist()
y_test_str = list(map(lambda x: labels_map[x], y_test))
correct = 0
wrong = {'prompt': list(), 'truth': list(), 'pred': list()}
for i in range(len(test)):
  prompt = f"<s>[INST] Given the text '{test.loc[i, 'Tweet']}' and the target '{test.loc[i, 'Target']}', classify the stance of the text towards the target. Stance options are: favor, against, none. The stance is [/INST]</s>"
  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
  output = model.generate(**inputs, streamer=streamer, max_new_tokens=200)
  pred = tokenizer.batch_decode(output[0], skip_special_tokens=True)
  if pred == y_test_str[i]:
    correct += 1
  else:
    wrong['prompt'].append(prompt)
    wrong['truth'].append(y_test_str)
    wrong['pred'].append(pred)
print(correct)
wrong_df = pandas.DataFrame(wrong)
wrong_df.to_csv('/content/wrongs.csv', index=False)
