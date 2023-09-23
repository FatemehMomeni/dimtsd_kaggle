import torch
import argparse
import data_helper as dh
import modeling, model_eval


def run_classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="all")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
    parser.add_argument("--train_mode", type=str, default="unified", help="unified or adhoc")
    parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
    parser.add_argument("--dataset_name", type=str, default="all", help="mt,semeval,am,wtwt,covid or all-dataset")
    parser.add_argument("--filename", default="Stance_All_Five_Datasets", type=str)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
    args = parser.parse_args()

    random_seeds = [1]
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

    for target_index in range(len(target_word_pair)):
      x_labels = dh.data_helper_bert(model_select, ['against', 'none', 'favor'])            
      t_labels = dh.data_loader(batch_size, model_select, model_name, x_labels)
      
      model = modeling.stance_classifier(3, model_select).cuda()         

      for n, p in model.named_parameters():
          if "bert.embeddings" in n:
              p.requires_grad = False

      optimizer_grouped_parameters = [
          {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': lr},
          {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')], 'lr': 1e-3},
          {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
          {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
      ]

      v_labels = model(t_labels[0], t_labels[1], t_labels[2])
      torch.save(v_labels, '/content/label_vectors.pt')


if __name__ == "__main__":
    run_classifier()
