import preprocessor as p
import re
import wordninja
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(filename: str, column_name: list):
    '''
    load all data as DataFrame type
    :param filename: path of csv file
    :param column_name: a list contains two label column names
    :return: pandas DataFrame
    '''
    dataset = pd.read_csv(filename, encoding='ISO-8859-1')
    dataset[column_name[0]].replace(to_replace=[r'FAVOR|support', r'NONE|comment', r'AGAINST|refute'], value=[2, 1, 0],
                                    regex=True, inplace=True)
    if len(column_name) > 1:
        dataset[column_name[1]].replace(to_replace=[r'FAVOR|support', r'NONE|comment', r'AGAINST|refute'],
                                        value=[2, 1, 0], regex=True, inplace=True)
    # dataset.rename(columns={'Stance 1': 'Stance', 'Target 1': 'Target', 'Stance 2': 'Stance', 'Target 2': 'Target'},
    #                inplace=True)
    dataset = dataset[dataset['Stance 1'] != 'unrelated']  # remove 'unrelated' label of WT-WT

    return dataset


def data_clean(strings: str, norm_dict: dict):
    '''
    clean data
    :param strings: texts to be cleaned
    :param norm_dict: normalization dictionary
    :return: list of cleaned texts
    '''
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])  # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


def clean_all(filename: str, column_name: list, norm_dict: dict):
    '''
    load and clean csv files
    :param filename: path of csv file
    :param column_name: a list contains two label column names
    :param norm_dict: normalization dictionary
    :return: a dictionary of text, targets, and labels
    '''
    dataset = load_data(filename, column_name)
    # convert DataFrame to list of strings ['string','string', ...]
    text = dataset['Tweet'].values.tolist()
    num_targets = len(column_name)
    if column_name[0] == "Stance 1":
        label1 = dataset['Stance 1'].values.tolist()
        target1 = dataset['Target 1'].values.tolist()
        if num_targets > 1:
            label2 = dataset['Stance 2'].values.tolist()
            target2 = dataset['Target 2'].values.tolist()
    else:
        label1 = dataset['Stance 2'].values.tolist()
        target1 = dataset['Target 2'].values.tolist()
        label2 = dataset['Stance 1'].values.tolist()
        target2 = dataset['Target 1'].values.tolist()

    for i in range(len(label1)):
        # clean each text to list of list of words [ ['word1','word2'], [...], ... ]
        text[i] = data_clean(text[i], norm_dict)

        target1[i] = target1[i].lower()
        if num_targets > 1:
            target2[i] = target2[i].lower()
    if num_targets > 1:
        processed = {'tweet': text, 'target1': target1, 'label1': label1, 'target2': target2, 'label2': label2}
    else:
        processed = {'tweet': text, 'target1': target1, 'label1': label1, 'target2': [], 'label2': []}
    return processed


def concatenation(single: dict, general:dict, multi: dict):
    '''
    concatenates single- and multi-target data
    :param single: single target data
    :param general: single target generalization data
    :param multi: multi target data
    :return: five lists of text, targets, and labels
    '''
    text = multi['tweet'] + single['tweet'] + general['tweet']
    target1 = multi['target1'] + single['target1'] + general['target1']
    label1 = multi['label1'] + single['label1'] + general['label1']
    target2 = multi['target2'] + single['target2'] + general['target2']
    label2 = multi['label2'] + single['label2'] + general['label2']

    return text, target1, label1, target2, label2
