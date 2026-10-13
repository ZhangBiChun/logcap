import math
from typing import Optional

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LogBERTModel(nn.Module):
    """
    LogBERT architecture using standard Transformer Encoder for sequence anomaly detection.
    """

    def __init__(
        self,
        act_vocab_size: int,
        time_vocab_size: int,
        sem_vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_users: int = 1,
        user_emb_dim: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.user_emb_dim = user_emb_dim
        self.num_users = max(1, int(num_users))

        self.embed_act = nn.Embedding(act_vocab_size + 2, hidden_size)
        self.embed_time = nn.Embedding(time_vocab_size + 2, hidden_size)
        self.embed_sem = nn.Embedding(sem_vocab_size + 2, hidden_size)
        self.user_emb = nn.Embedding(self.num_users, user_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3 + user_emb_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_act = nn.Linear(hidden_size, act_vocab_size)
        self.fc_time = nn.Linear(hidden_size, time_vocab_size)
        self.fc_sem = nn.Linear(hidden_size, sem_vocab_size)

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_act = x[:, :, 0]
        x_time = x[:, :, 1]
        x_sem = x[:, :, 2]
        b, t, _ = x.shape

        e_act = self.embed_act(x_act)
        e_time = self.embed_time(x_time)
        e_sem = self.embed_sem(x_sem)

        if user_idx is None:
            user_idx = torch.zeros(b, dtype=torch.long, device=x.device)
        ue = self.user_emb(user_idx).unsqueeze(1).expand(b, t, self.user_emb_dim)
        fused = self.fusion(torch.cat([e_act, e_time, e_sem, ue], dim=-1))
        fused = self.pos_encoder(fused)
        
        out = self.transformer_encoder(fused)
        return self.fc_act(out), self.fc_time(out), self.fc_sem(out)
