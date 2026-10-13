import torch
import torch.nn as nn

class LaAebModel(nn.Module):
    """
    LaAeb (Log-text analysis Autoencoder based) surrogate model.
    A simple Autoencoder over sequence embeddings as described in the paper.
    """
    def __init__(self, act_vocab_size, time_vocab_size, sem_vocab_size, hidden_size=128, dropout=0.1):
        super().__init__()
        self.embed_act = nn.Embedding(act_vocab_size + 2, hidden_size)
        self.embed_time = nn.Embedding(time_vocab_size + 2, hidden_size)
        self.embed_sem = nn.Embedding(sem_vocab_size + 2, hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
    def forward(self, x):
        # x: (batch_size, window_size, 3)
        e_act = self.embed_act(x[:, :, 0])
        e_time = self.embed_time(x[:, :, 1])
        e_sem = self.embed_sem(x[:, :, 2])
        
        emb = self.fusion(torch.cat([e_act, e_time, e_sem], dim=-1))
        
        # Mean pooling over the window to get a single representation for the sequence
        seq_emb = torch.mean(emb, dim=1) # (batch_size, hidden_size)
        
        encoded = self.encoder(seq_emb)
        decoded = self.decoder(encoded)
        
        return seq_emb, decoded
