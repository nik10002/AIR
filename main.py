import filter, train, eval
from ProductRanker import *
from train import *
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer

def dataPreparation():
    df = pd.read_json("filtered_meta_Electronics.json", lines=True).sample(n = 110)
    #df = pd.read_json("filtered_meta_Electronics.json", lines=True)[:110]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_set = []
    for index, row in df.iterrows():
        tokenized_set.append([tokenizer.encode_plus(row['description'], truncation = True, return_tensors="pt",
                                                    max_length=512, pad_to_max_length=True), row['salesRank']])
    return torch.utils.data.random_split(tokenized_set, [100, 10])

def createDataloader(dataset, test = False):
    if test:
        print("asdf")
    else:
        labeled_set = []
        for i, doc1 in enumerate(dataset):
            for j, doc2 in enumerate(dataset):
                if i != j:
                    label = 0.0
                    if doc1[1] > doc2[1]:
                        label = 1.0
                    if doc1[1] == doc2[1]:
                        label = 0.5

                    labeled_set.append([[doc1[0], doc2[0]], label])
        return torch.utils.data.DataLoader(labeled_set, batch_size=100)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductRanker().to(device)

    train_set, val_set = dataPreparation()
    train_dataloader = createDataloader(train_set)
    val_dataloader = createDataloader(val_set)
    loss_fn = nn.BCELoss()
    opimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    train(epochs=1, dataloader_train=train_dataloader, dataloader_val=val_dataloader, model=model, loss_fn=loss_fn, optimizer=opimizer, device=device)



    #loss_fn = nn.BCELoss()


if __name__ == "__main__":
    main()