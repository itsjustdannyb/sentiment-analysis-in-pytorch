import os
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import TweetTokenizer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from dataset import SentimentDataset
from model import SentimentClassifier
from engine import train



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = Path(r"C:\Users\bamilosin\Documents\dataset\nlp\sentiment data")
train_path = data_path / "train.csv"
test_path = data_path / "test.csv"

# get data.csv encoding

from charset_normalizer import detect

with open(train_path, 'rb') as file:
    result = detect(file.read())


train_data = pd.read_csv(train_path, encoding=result['encoding'])
train_data = train_data[['text', 'sentiment']] # the two columns we need.

# remove null values
train_data = train_data.dropna()
train_data = train_data.reset_index() # reset index because of removed rows


# create dataset and dataloaders

print("creating dataset")
dataset = SentimentDataset(train_data)
indices = torch.randperm(len(dataset)).tolist()

train_size = 0.7
train_dataset = Subset(dataset, indices[:int(train_size*len(dataset))])
test_dataset = Subset(dataset, indices[int(train_size*len(dataset)):])

BATCH_SIZE = 128
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(dataset.vocab)
EMBED_DIM = 128

print("creating model")
model = SentimentClassifier(vocab_size=VOCAB_SIZE,
                            embedding_dim=EMBED_DIM,
                            nhead=8,
                            mlp_size=512,
                            num_layers=4,
                            device=device).to(device)


LR = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss().to(device)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)


train(model, train_dataloader, 20,  optimizer, loss_fn, scheduler)
torch.save(model.state_dict(), "classifier_weights.pth")
