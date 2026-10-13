from typing import Optional

import torch
import torch.nn as nn


class DeepLogTHNS24(nn.Module):
    """
    DeepLog-style next-action predictor for THNS24 event triples.
    Input shape: [B, T, 3] = [act, hour, sem].
    Optional per-session user conditioning via nn.Embedding (index 0 = unknown).
    """

    def __init__(
        self,
        act_vocab_size: int,
        time_vocab_size: int,
        sem_vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_users: int = 1,
        user_emb_dim: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.user_emb_dim = user_emb_dim
        self.num_users = max(1, int(num_users))

        self.embed_act = nn.Embedding(act_vocab_size + 1, hidden_size)
        self.embed_time = nn.Embedding(time_vocab_size + 1, hidden_size)
        self.embed_sem = nn.Embedding(sem_vocab_size + 1, hidden_size)
        self.user_emb = nn.Embedding(self.num_users, user_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3 + user_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_act = nn.Linear(hidden_size, act_vocab_size + 1)
        self.fc_time = nn.Linear(hidden_size, time_vocab_size + 1)
        self.fc_sem = nn.Linear(hidden_size, sem_vocab_size + 1)

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None):
        x_act = x[:, :, 0]
        x_time = x[:, :, 1]
        x_sem = x[:, :, 2]
        b, t, _ = x.shape
        z = torch.cat(
            [self.embed_act(x_act), self.embed_time(x_time), self.embed_sem(x_sem)],
            dim=-1,
        )
        if user_idx is None:
            user_idx = torch.zeros(b, dtype=torch.long, device=x.device)
        ue = self.user_emb(user_idx)
        ue = ue.unsqueeze(1).expand(b, t, self.user_emb_dim)
        z = torch.cat([z, ue], dim=-1)
        z = self.fusion(z)
        out, _ = self.lstm(z)
        last_out = out[:, -1, :]
        return self.fc_act(last_out), self.fc_time(last_out), self.fc_sem(last_out)
