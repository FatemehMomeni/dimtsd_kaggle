import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd


# load all data as DataFrame type
def load_data(filename, column_name):

  dataset = pd.read_csv(filename, encoding='ISO-8859-1')
  dataset[column_name].replace(to_replace=[r'FAVOR|support', r'NONE|comment', r'AGAINST|refute'], value=[2,1,0], regex=True, inplace=True)  
  dataset.rename(columns={'Stance 1':'Stance','Target 1':'Target','Stance 2':'Stance','Target 2':'Target'}, inplace=True)
  dataset = dataset[dataset.Stance != 'unrelated'] # remove 'unrelated' label of WT-WT  
  
  return dataset


# clean data
def data_clean(strings,norm_dict):
    
  p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
  clean_data = p.clean(strings) # using lib to clean URL,hashtags...
  clean_data = re.sub(r"#SemST", "", clean_data)
  clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
  clean_data = [[x.lower()] for x in clean_data]
  
  for i in range(len(clean_data)):
    if clean_data[i][0] in norm_dict.keys():
      clean_data[i] = norm_dict[clean_data[i][0]].split()
      continue
    if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
      clean_data[i] = wordninja.split(clean_data[i][0]) # separate hashtags
  clean_data = [j for i in clean_data for j in i]

  return clean_data


# Clean All Data
def clean_all(filename, column_name, norm_dict):
    
  dataset = load_data(filename, column_name) 
  # convert DataFrame to list of strings ['string','string', ...]
  text = dataset['Tweet'].values.tolist()
  target = dataset['Target'].values.tolist()
  label = dataset['Stance'].values.tolist()
  
  for i in range(len(label)):
    # clean each text to list of list of words [ ['word1','word2'], [...], ... ]
    text[i] = data_clean(text[i], norm_dict)    
    target[i] = target[i].lower()    
  
  return text, target, label
