import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW

def train(epochs, dataloader_train, dataloader_val, model, loss_fn, optimizer, device):
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_loss = 0
        for X, y in dataloader_train:
            input_ids_1 = X[0]['input_ids'].squeeze().to(device)
            attention_mask_1 = X[0]['attention_mask'].squeeze().to(device)
            input_ids_2 = X[1]['input_ids'].squeeze().to(device)
            attention_mask_2 = X[1]['attention_mask'].squeeze().to(device)
            out1 = model.forward(input_ids_1, attention_mask_1)
            out2 = model.forward(input_ids_2, attention_mask_2)
            diff = (out1 - out2).squeeze()
            diff = torch.sigmoid(diff).to(device)
            loss = loss_fn(diff, y.float().to(device))
            print('batch loss: ',loss)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Total epoch train loss: %f" % epoch_loss)
        val(dataloader_val, model, loss_fn, device)

def val(dataloader, model, loss_fn, device):
        val_loss = 0
        for X, y in dataloader:
            input_ids_1 = X[0]['input_ids'].squeeze().to(device)
            attention_mask_1 = X[0]['attention_mask'].squeeze().to(device)
            input_ids_2 = X[1]['input_ids'].squeeze().to(device)
            attention_mask_2 = X[1]['attention_mask'].squeeze().to(device)
            out1 = model.forward(input_ids_1, attention_mask_1)
            out2 = model.forward(input_ids_2, attention_mask_2)
            diff = (out1 - out2).squeeze()
            diff = torch.sigmoid(diff)
            loss = loss_fn(diff, y.float().to(device))
            print('batch loss: ',loss)
            val_loss += loss

        print("Total epoch val loss: %f" % val_loss)






