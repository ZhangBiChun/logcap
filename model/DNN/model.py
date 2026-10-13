import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, configs):
        super(DNN, self).__init__()
        self.window_size = configs.window_size
        self.n_features = configs.n_features
        self.hidden_size = configs.hidden_size
        
        input_dim = self.window_size * self.n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, self.window_size, self.n_features)
