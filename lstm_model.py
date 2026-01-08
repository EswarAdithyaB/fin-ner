import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tag_count,
        embedding_dim=100,
        lstm_units=256,
        dense_units=128,
        dropout=0.1
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            embedding_dim,
            lstm_units // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # TimeDistributed Dense
        self.fc = nn.Linear(lstm_units, dense_units)
        self.relu = nn.ReLU()

        # Emission layer
        self.classifier = nn.Linear(dense_units, tag_count)

        # CRF
        self.crf = CRF(tag_count, batch_first=True)

    def forward(self, input_ids, tags=None, mask=None):
        x = self.embedding(input_ids)
        x, _ = self.bilstm(x)
        x = self.relu(self.fc(x))
        emissions = self.classifier(x)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)
