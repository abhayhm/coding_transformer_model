import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Embedding):
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(d_model, vocab_size)

    def forward(self, vocab):
        self.embedding(vocab) * math.sqrt(self.d_model)