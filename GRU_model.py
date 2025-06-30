from torch import nn
import torch

class GRU(nn.Module):
    # GRU model for time series prediction
    def __init__(self, input_size=21, hidden_size=128, num_layers=1, num_classes=1, device='cpu'):
        super(GRU, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, hn = self.gru(x, h0)
        # out: (batch_size, seq_length, hidden_size)
        # out: (N, 100, 128)

        out = out[:, -1, :]
        # out: (N, 128)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class GRUEmbedding(nn.Module):
    # GRU model with an embedding layer for creator indices
    def __init__(self, num_features=20, hidden_size=128, num_layers=1, num_classes=1, embedding_size=200, device='cpu'):
        super(GRUEmbedding, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.input_size = num_features + embedding_size # input size of the GRU is the concatenation of the features and the embedding
        self.embedding = nn.Embedding(447972, embedding_dim=embedding_size, padding_idx=431231)
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x, x_idx):
        # x: (N, 100, 20)
        # x_idx: (N, 100) # 100 rows of creator indices to match sequence

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # get the embedding for the creator index
        # reshape the embedding to match the shape of x for concatenation
        x_idx = self.embedding(x_idx).view(x.size(0), x.size(1), -1)
        
        # concatenate the features and the embedding
        x = torch.cat((x, x_idx),dim=2)


        out, hn = self.gru(x, h0)
        # out: (batch_size, seq_length, hidden_size)
        # out: (N, 100, F + EMB)

        out = out[:, -1, :]
        # out: (N, F + EMB)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

