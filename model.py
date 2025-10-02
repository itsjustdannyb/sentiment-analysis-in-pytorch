
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, nhead:int, mlp_size:int, num_layers:int, device=device):
        super(SentimentClassifier, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pos_embedding = torch.randn(1, 1, embedding_dim, requires_grad=True, device=device)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                  nhead=nhead,
                                                  dim_feedforward=mlp_size,
                                                  batch_first=True)
        self.encoder_block = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=3)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embedding_layer(x)
        # get positional embeddings for each batch
        pos_embed = self.pos_embedding.expand(batch_size,  1, x.shape[-1])
        # add token embeddings to positional embeddings
        pos_token_embeddings = pos_embed + x
        x = self.encoder_block(pos_token_embeddings)
        return self.classifier(x[:, -1])
