
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(y_logits, y_true):
    acc = torch.eq(torch.argmax(y_logits, dim=-1), y_true).sum().item()
    return (acc / len(y_true)) * 100.0

def input_to_indices(dataset, text):
    print(f"original text: {text}")
    # clean and tokenize, pad and get indices
    clean_text = dataset.clean_data(text)
    print(f"cleaned_data: {clean_text}")
    tokens = dataset.tokenize_data(clean_text)
    print(f"text tokens: {tokens}")
    padded_tokens = dataset.pad_token(tokens, dataset.longest_seq)
    print(f"padded_tokens: {padded_tokens}")
    indices = dataset.create_indices(dataset.vocab, padded_tokens)
    print(f"indices: {indices}")

    return indices