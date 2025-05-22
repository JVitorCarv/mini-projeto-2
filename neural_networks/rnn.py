import torch.nn as nn
import torch.nn.functional as F


class RNNClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        x = self.fc(hidden.squeeze(0))
        return self.activation(x)
