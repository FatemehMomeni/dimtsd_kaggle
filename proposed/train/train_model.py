import torch
import torch.nn as nn
import random
import argparse
import numpy as np
import json
import preprocessing as pp
import data_helper as dh
import modeling
import model_eval


def run_classifier():
    # get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    label_column = {'single': ['Stance 1'], 'multi': ['Stance 1', 'Stance 2']}
    lr = args.learning_rate
    batch_size = args.batch_size
    alpha = 0.5
    total_epoch = args.epochs
    random_seeds = [1]# [1, 2, 4, 5, 9, 10]

    # create normalization dictionary for preprocessing
    with open("../noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("../emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    # dataset paths
    train_single_path = '../data/train_single.csv'
    train_multi_path = '../data/train_multi.csv'
    eval_single_path = '../data/val_single.csv'
    eval_multi_path = '../data/val_multi.csv'
    test_single_path = '../data/test_single.csv'
    test_single_general_path = '../data/test_single_generalization.csv'
    test_multi_path = '../data/test_multi.csv'

    # preprocess data
    temp_dict = {'tweet': [], 'target1': [], 'label1': [], 'target2': [], 'label2': []}
    train_single = pp.clean_all(train_single_path, label_column['single'], normalization_dict)
    train_multi = pp.clean_all(train_multi_path, label_column['multi'], normalization_dict)
    text_train, tar1_train, y1_train, tar2_train, y2_train = pp.concatenation(train_single, temp_dict, train_multi)

    val_single = pp.clean_all(eval_single_path, label_column['single'], normalization_dict)
    val_multi = pp.clean_all(eval_multi_path, label_column['multi'], normalization_dict)
    text_val, tar1_val, y1_val, tar2_val, y2_val = pp.concatenation(val_single, temp_dict, val_multi)

    test_single = pp.clean_all(test_single_path, label_column['single'], normalization_dict)
    test_single_general = pp.clean_all(test_single_general_path, label_column['single'], normalization_dict)
    test_multi = pp.clean_all(test_multi_path, label_column['multi'], normalization_dict)
    text_test, tar1_test, y1_test, tar2_test, y2_test = pp.concatenation(test_single, test_single_general, test_multi)

    single_len = len(test_single['label1'])
    multi_len = len(test_multi['label1'])
    # save y_test for separating test set in sep_test_set function
    y_unique_test_single = test_single['label1']
    y_unique_test_single_general = test_single_general['label1']
    y_unique_test_multi = test_multi['label1']

    train_all = [text_train, tar1_train, y1_train, tar2_train, y2_train]
    val_all = [text_val, tar1_val, y1_val, tar2_val, y2_val]
    test_all = [text_test, tar1_test, y1_test, tar2_test, y2_test]

    num_labels = {'single': len(set(val_single['label1'])), 'multi': len(set(val_multi['label1']))}
    classes = list()
    for l in ['against', 'none', 'favor']:
        classes.append(torch.load(f'../model/{l}_label_vector.pt'))

    best_val = []
    best_result = {'single': [], 'single_gen': [], 'multi': []}
    for seed in random_seeds:
        print("current random seed: ", seed)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = modeling.StanceClassifier(num_labels, batch_size, classes).to('cuda')

        # prepare for model
        y1_train, y2_train, train_loader, train_loader_eval = dh.data_helper_bert(train_all, model, batch_size, 'train')
        # y_train_multi, train_loader_multi = dh.data_helper_bert(train_multi, model, batch_size, 'train')
        y1_val, y2_val, val_loader, _ = dh.data_helper_bert(val_all, model, batch_size, 'val')
        # y_val_multi, val_loader_multi = dh.data_helper_bert(val_multi, model, batch_size, 'val')
        y1_test, y2_test, test_loader, _ = dh.data_helper_bert(test_all, model, batch_size, 'test')
        # y_test_single_gen, test_loader_single_gen = dh.data_helper_bert(test_single_general, model, batch_size, 'test')
        # y_test_multi, test_loader_multi = dh.data_helper_bert(test_multi, model, batch_size, 'test')

        # freeze some layers
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': lr},
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
        ]

        loss_function = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        sum_loss, train_predictions_distill, val_f1_average = [], [], []
        test_f1_average = {
            'single': [[] for _ in range(len(set(test_single['target1'])))],
            'single_general': [[] for _ in range(len(set(test_single_general['target1'])))],
            'multi': [[] for _ in range(len(set(test_multi['target1'])))]
        }

        for epoch in range(total_epoch):
            print('Epoch:', epoch)
            # training phase
            train_loss = list()
            model.train()
            for input_ids, attention_mask, mask_position, label1, input_ids2, attention_mask2, mask_position2, label2 \
                    in train_loader:
                optimizer.zero_grad()
                output1, output2 = model(False, input_ids, attention_mask, mask_position,
                                         input_ids2, attention_mask2, mask_position2)
                loss = loss_function(output1, label1)
                if input_ids2:
                    loss2 = loss_function(output2, label2)
                    loss = loss + loss2 * alpha
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss.append(loss.item())
            sum_loss.append(sum(train_loss) / len(text_train))
            print(sum_loss[epoch])  # summation of loss in this epoch

            # evaluation phase, train evaluation
            model.eval()
            train_predictions = list()
            with torch.no_grad():
                for input_ids, attention_mask, mask_position, label1, \
                        input_ids2, attention_mask2, mask_position2, label2 in train_loader_eval:
                    prediction, _ = model(False, input_ids, attention_mask, mask_position,
                                          input_ids2, attention_mask2, mask_position2)
                    train_predictions.append(prediction)
                prediction = torch.cat(train_predictions, 0)
                train_predictions_distill.append(prediction)
                print("The size of train predictions is: ", prediction.size())

            # evaluation on val set
            val_predictions = list()
            with (torch.no_grad()):
                for input_ids, attention_mask, mask_position, label1, \
                        input_ids2, attention_mask2, mask_position2, label2 in val_loader:
                    prediction, _ = model(False, input_ids, attention_mask, mask_position,
                                          input_ids2, attention_mask2, mask_position2)
                    val_predictions.append(prediction)
                prediction = torch.cat(val_predictions, 0)
                acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y1_val)
                val_f1_average.append(f1_average)

            # test phase
            y_test_multi_list = dh.sep_test_set(y1_test[:multi_len], y_unique_test_multi)
            y_test_single_list = dh.sep_test_set(y1_test[multi_len: multi_len+single_len], y_unique_test_single)
            y_test_single_gen_list = dh.sep_test_set(y1_test[multi_len+single_len:], y_unique_test_single_general)
            with torch.no_grad():
                test_predictions = list()
                for input_ids, attention_mask, mask_position, label1, \
                        input_ids2, attention_mask2, mask_position2, label2 in test_loader:
                    prediction, _ = model(False, input_ids, attention_mask, mask_position,
                                          input_ids2, attention_mask2, mask_position2)
                    test_predictions.append(prediction)
                prediction = torch.cat(test_predictions, 0)
                prediction_list_multi = dh.sep_test_set(prediction[:multi_len], y_unique_test_multi)
                prediction_list_single = dh.sep_test_set(prediction[multi_len:single_len], y_unique_test_single)
                prediction_list_single_gen = dh.sep_test_set(prediction[single_len:], y_unique_test_single_general)

                # measuring F1 for multi-target dataset
                test_predictions = list()
                for ind in range(len(y_test_multi_list)):
                    prediction = prediction_list_multi[ind]
                    test_predictions.append(prediction)
                    acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y_test_multi_list[ind])
                    test_f1_average['multi'][ind].append(f1_average)

                # measuring F1 for single-target dataset
                test_predictions = list()
                for ind in range(len(y_test_single_list)):
                    prediction = prediction_list_single[ind]
                    test_predictions.append(prediction)
                    acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y_test_single_list[ind])
                    test_f1_average['single'][ind].append(f1_average)

                # measuring F1 for single-target generalization dataset
                test_predictions = list()
                for ind in range(len(y_test_single_gen_list)):
                    prediction = prediction_list_single_gen[ind]
                    test_predictions.append(prediction)
                    acc, f1_average, precision, recall = model_eval.compute_f1(prediction, y_test_single_gen_list[ind])
                    test_f1_average['single_general'][ind].append(f1_average)

        # model that performs best on the eval set is evaluated on the test set
        best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_result['single'].append([f1[best_epoch] for f1 in test_f1_average['single']])
        best_result['single_gen'].append([f1[best_epoch] for f1 in test_f1_average['single_general']])
        best_result['multi'].append([f1[best_epoch] for f1 in test_f1_average['multi']])

        best_predictions = train_predictions_distill[best_epoch]
        torch.save(best_predictions, f'DI_MtSD_seed{seed}.pt')

        print("*" * 20)
        print(f"dev results with seed {seed} on all epochs")
        print(val_f1_average)
        best_val.append(val_f1_average[best_epoch])

        print("*" * 20)
        print(f"test results with seed {seed} on all epochs")
        print("multi-target")
        print(test_f1_average['multi'])
        print("\nsingle-target")
        print(test_f1_average['single'])
        print("\nsingle-target generalization")
        print(test_f1_average['single_general'])

        print("*" * 20)
        print("multi-target")
        print(max(best_result['multi']))
        print(best_result['multi'])
        print("\nsingle-target")
        print(max(best_result['single']))
        print(best_result['single'])
        print("\nsingle-target generalization")
        print(max(best_result['single_gen']))
        print(best_result['single_gen'])


if __name__ == "__main__":
    run_classifier()
