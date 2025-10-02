
import os
import re
import pandas as pd

import nltk
from nltk.tokenize import TweetTokenizer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset



class SentimentDataset(Dataset):
    def __init__(self, df, csv_encoding:str="windows-1250"):
        super(SentimentDataset, self).__init__()

        self.df = df
        self.df.loc[:, 'text'] = self.df.loc[:, 'text'].apply(self.clean_data) # clean data
        self.df.loc[:, 'tokens'] = self.df.loc[:, 'text'].apply(self.tokenize_data) # tokenize data

        self.all_tokens = list(self.df.loc[:, 'tokens'].values) # get all tokens
        self.longest_seq = self.get_longest_seq(self.all_tokens) # get longest token
        self.corpus = sorted(list(set([text for sublist in self.all_tokens for text in sublist]))) # set of words

        self.vocab = self.create_vocab(self.corpus) # create vocab
        self.idx2word = {idx : key for key, idx in self.vocab.items()} # idx to word

        self.df.loc[:, 'tokens'] = self.df.loc[:, 'tokens'].apply(lambda x : self.pad_token(x, self.longest_seq, self.idx2word[0])) # pad tokens

        self.df.loc[:, 'indices'] = self.df.loc[:, 'tokens'].apply(lambda x: self.create_indices(self.vocab, x)) # create indices

        self.label_map = {
            'negative' : 0,
            'positive': 1,
            'neutral' : 2
        }

        self.df.loc[:, 'labels'] = self.df.loc[:, 'sentiment'].apply(lambda x : self.labelizer(x, self.label_map)) # get label id


    def clean_data(self, text):
        text_lower = text.lower()
        clean_text = re.sub('[<>{};@#$%^&*()>]', '', text_lower)
        return clean_text

    def tokenize_data(self, text):

        stopwords = nltk.corpus.stopwords.words('english')
        tokenizer = nltk.TweetTokenizer()
        tokens = tokenizer.tokenize(text)

        # remove stopwords
        tokens = [token for token in tokens if token not in stopwords]
        return tokens

    def get_longest_seq(self, token_lists):
        longest_seq = 0
        for token_list in token_lists:
            if len(token_list) > longest_seq:
                longest_seq = len(token_list)

        return longest_seq

    def pad_token(self, token_list, longest_seq, pad_token:str='<PAD>'):
        if len(token_list) < longest_seq:
            return token_list + [pad_token for _ in range(longest_seq - len(token_list))]
        else:
            return token_list

    def create_vocab(self, corpus):
        vocab = {
            "<PAD>": 0
        }

        for word in corpus:
            vocab[word] = (len(vocab) - 1) +  1


        return vocab

    def create_indices(self, vocab, tokens):
        return [vocab[token] for token in tokens]

    def labelizer(self, label, label_map):
        return label_map[label]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        indices, label = self.df[['indices', 'labels']].loc[idx].values
        indices = torch.tensor(indices, dtype=torch.int)
        label = torch.tensor(label)

        return indices, label    

