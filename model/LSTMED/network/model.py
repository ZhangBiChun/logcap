import torch
import torch.nn as nn

class LSTMEDModule(nn.Module):
    """
    The LSTM-encoder-decoder module. Both the encoder and decoder are LSTMs.

    :meta private:
    """

    def __init__(self, configs, device):
        """
        :param n_features: The input feature dimension
        :param hidden_size: The LSTM hidden size
        :param n_layers: The number of LSTM layers
        :param dropout: The dropout rate
        :param device: CUDA or CPU
        """
        super().__init__()
        self.n_features = configs.n_features
        self.hidden_size = configs.hidden_size
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.device = device

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=True,
            dropout=self.dropout[0],
        )
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=True,
            dropout=self.dropout[1],
        )
        self.output_layer = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x, return_latent=False):
        # Encoder
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x.float(), enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        return (output, enc_hidden[1][-1]) if return_latent else output
