import torch
import torch.nn as nn
from transformers import BertModel, AdamW

class ProductRanker(nn.Module):
    def __init__(self, l1=8, l2=256):
        super(ProductRanker, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.ranker = nn.Sequential(
            nn.Linear(768, l1),
            nn.ReLU(),
            nn.BatchNorm1d(l1),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.BatchNorm1d(l2),
            nn.Linear(l2,1)
        )

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.ranker(last_hidden_state_cls)
        return logits