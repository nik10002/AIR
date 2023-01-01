import torch
from transformers import BertTokenizer
import pandas as pd
from model import *
from inference import *
import os
from ray import tune

def dataPreparation(data_dir="./data"):
    path = os.path.join(data_dir, "filtered_meta_Electronics.json")
    df = pd.read_json(path, lines=True).sample(n = 120)
    #df = pd.read_json("filtered_meta_Electronics.json", lines=True)[:110]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_set = []
    for index, row in df.iterrows():
        tokenized_set.append([tokenizer.encode_plus(row['description'], truncation = True, return_tensors="pt",
                                                    max_length=512, pad_to_max_length=True), row['salesRank']])
    return torch.utils.data.random_split(tokenized_set, [100, 20])

def createDataloader(dataset, config, shuffle=True):
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
    return torch.utils.data.DataLoader(labeled_set, batch_size=int(config["batch_size"]), num_workers=8, shuffle=shuffle)


def train_tune(config, checkpoint_dir=None, data_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductRanker(config["l1"],config["l2"]).to(device)

    train_set, val_set = dataPreparation(data_dir)
    dataloader_train = createDataloader(train_set, config)
    dataloader_val = createDataloader(val_set, config, shuffle=False)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


    for t in range(4):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(dataloader_train):
            X, y = data
            input_ids_1 = X[0]['input_ids'].squeeze().to(device)
            attention_mask_1 = X[0]['attention_mask'].squeeze().to(device)
            input_ids_2 = X[1]['input_ids'].squeeze().to(device)
            attention_mask_2 = X[1]['attention_mask'].squeeze().to(device)
            out1 = model.forward(input_ids_1, attention_mask_1)
            out2 = model.forward(input_ids_2, attention_mask_2)
            diff = (out1 - out2).squeeze()
            diff = torch.sigmoid(diff).to(device)
            loss = loss_fn(diff, y.float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (t + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0


        val_loss = val(dataloader_val, model, loss_fn, device)

        with tune.checkpoint_dir(t) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss)

def train(epochs, config, data_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductRanker(config["l1"],config["l2"]).to(device)

    train_set, val_set = dataPreparation(data_dir)
    dataloader_train = createDataloader(train_set, config)
    dataloader_val = createDataloader(val_set, config)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    print("Starting training loop...")
    for t in range(2):
        print(f"Epoch {t + 1}\n-------------------------------")
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(dataloader_train):
            X, y = data
            input_ids_1 = X[0]['input_ids'].squeeze().to(device)
            attention_mask_1 = X[0]['attention_mask'].squeeze().to(device)
            input_ids_2 = X[1]['input_ids'].squeeze().to(device)
            attention_mask_2 = X[1]['attention_mask'].squeeze().to(device)
            out1 = model.forward(input_ids_1, attention_mask_1)
            out2 = model.forward(input_ids_2, attention_mask_2)
            diff = (out1 - out2).squeeze()
            diff = torch.sigmoid(diff).to(device)
            loss = loss_fn(diff, y.float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 600 == 599:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.6f" % (t + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        val_loss = val(dataloader_val, model, loss_fn, device)
        print("[%d] val loss: %.6f" % (t+1, val_loss))

        path = 'outputs/model_' + str(t+1) + '.pth'
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        print("Model saved..")

    print("Training finished!")


def val(dataloader, model, loss_fn, device):
        val_loss = 0.0
        val_steps = 0
        for X, y in dataloader:
            with torch.no_grad():
                input_ids_1 = X[0]['input_ids'].squeeze().to(device)
                attention_mask_1 = X[0]['attention_mask'].squeeze().to(device)
                input_ids_2 = X[1]['input_ids'].squeeze().to(device)
                attention_mask_2 = X[1]['attention_mask'].squeeze().to(device)
                out1 = model.forward(input_ids_1, attention_mask_1)
                out2 = model.forward(input_ids_2, attention_mask_2)
                diff = (out1 - out2).squeeze()
                diff = torch.sigmoid(diff)
                loss = loss_fn(diff, y.float().to(device))
                val_loss += loss.cpu().numpy()
                val_steps += 1

        return (val_loss / val_steps)


'''def save_model(t, model, optimizer, loss):
    path = 'outputs/model_' + t + '.pth'
    torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
    print("Model saved..")'''





