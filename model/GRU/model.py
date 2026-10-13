import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, configs, device):
        super(GRUModel, self).__init__()
        self.window_size = configs.window_size
        self.n_features = configs.n_features
        self.hidden_size = configs.hidden_size
        self.device = device
        
        self.gru = nn.GRU(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.n_features)

    def forward(self, x):
        # x is (batch_size, window_size, n_features)
        gru_out, _ = self.gru(x)
        # gru_out is (batch_size, window_size, hidden_size)
        out = self.fc(gru_out)
        return out
