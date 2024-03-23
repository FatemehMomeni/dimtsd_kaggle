import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class StanceClassifier(nn.Module):
    def __init__(self, num_labels, batch, label_vectors):
        super(StanceClassifier, self).__init__()
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, mask_token='[MASK]')
        self.bert.pooler = None
        self.linear = nn.Linear(num_labels['multi'] * 2, batch)
        self.out = nn.Linear(batch, 2)
        self.pad = torch.zeros(batch, 512).to('cuda:0')
        # self.pad = torch.tensor([-1 for _ in range(512)], dtype=torch.long).to('cuda')
        self.lv = torch.cat(label_vectors, 0)
        self.lv = torch.transpose(self.lv, 0, 1)

    def forward(self, embedding, input_ids, attention_mask, mask_position,
                input_ids2, attention_mask2, mask_position2):
        if embedding:
            last_hidden = self.bert(input_ids, attention_mask)
            word_embed = last_hidden[0][:, 1]
            return word_embed

        else:
            last_hidden = self.bert(input_ids, attention_mask)[0]
            last_hidden2 = self.bert(input_ids2, attention_mask2)[0]
            multi_index = list()
            similarity = torch.zeros(1, 3).cuda()
            similarity2 = torch.zeros(1, 3).cuda()
            for i in range(len(input_ids)):
                h_mask = last_hidden[i, mask_position[i]]
                h_mask = self.relu(self.dropout(h_mask))
                h_mask = torch.unsqueeze(h_mask, 0)
                similarity = torch.cat((similarity, torch.tensordot(h_mask, self.lv, dims=1)), 0)  # output size = 1*3

                if not (input_ids2[i] == self.pad).all():
                    multi_index.append(i)
                    h_mask2 = last_hidden2[i, mask_position2[i]]
                    h_mask2 = self.dropout(h_mask2)
                    h_mask2 = torch.unsqueeze(h_mask2, 0)
                    similarity2 = torch.cat((similarity2, torch.tensordot(h_mask2, self.lv, dims=1)), 0)

            similarity = similarity[1:]  # remove first row
            if (similarity2 == torch.zeros(1, 3).to('cuda')).all():
                out2 = None
            else:
                similarity2 = similarity2[1:]
                similarity1 = similarity[multi_index]
                context_vec = torch.cat((similarity1, similarity2), dim=1)
                linear2 = self.relu(self.linear(context_vec))
                out2 = self.out(linear2)
            
            return similarity, out2

            # if input_ids2:
            #     last_hidden2 = self.bert(input_ids2, attention_mask2)[0]
            #     similarity2 = torch.zeros(1, 3).cuda()
            #     for i in range(len(input_ids2)):
            #         h_mask2 = last_hidden2[i, mask_position2[i]]
            #         h_mask2 = self.dropout(h_mask2)
            #         h_mask2 = torch.unsqueeze(h_mask2, 0)
            #         similarity2 = torch.cat((similarity2, torch.tensordot(h_mask2, self.lv, dims=1)), 0)
            #     similarity2 = similarity2[1:]
            #
            #     context_vec = torch.cat((similarity, similarity2), dim=1)
            #     linear2 = self.relu(self.linear2(context_vec))
            #     out2 = self.out2(linear2)
            #
            # return similarity, out2
            # cls = last_hidden[0][:, 0]
            # query = self.dropout(cls)
            # linear = self.relu(self.linear(query))
            # out = self.out(linear)
            # return out
