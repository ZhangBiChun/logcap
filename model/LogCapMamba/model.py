import torch
import torch.nn as nn
from models.LogCapMamba.mamba_pytorch import Mamba

class LogCapMamba(nn.Module):
    def __init__(self, act_vocab_size, time_vocab_size, sem_vocab_size, 
                 hidden_size=128, d_state=16, d_conv=4, expand=2, num_layers=2, dropout=0.1,
                 num_users=1, user_emb_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embed_act = nn.Embedding(act_vocab_size + 1, hidden_size)
        self.embed_time = nn.Embedding(time_vocab_size + 1, hidden_size)
        self.embed_sem = nn.Embedding(sem_vocab_size + 1, hidden_size)
        
        self.user_emb = nn.Embedding(max(1, num_users), user_emb_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3 + user_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(hidden_size)
        
        # Binary classification head: Normal (0) vs Pseudo-Anomaly (1) PER EVENT
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, u: torch.Tensor):
        # x: [B, T, 3]
        h = self.extract_features(x, u)
        logits = self.classifier(h)
        return logits.squeeze(-1)
        
    def extract_features(self, x: torch.Tensor, u: torch.Tensor):
        x_act = x[:, :, 0]
        x_time = x[:, :, 1]
        x_sem = x[:, :, 2]
        b, t = x_act.shape
        
        e_act = self.embed_act(x_act)
        e_time = self.embed_time(x_time)
        e_sem = self.embed_sem(x_sem)
        
        e_u = self.user_emb(u).unsqueeze(1).expand(b, t, -1)
        
        fused = self.fusion(torch.cat([e_act, e_time, e_sem, e_u], dim=-1))
        
        h = fused
        for mamba_layer, norm in zip(self.mamba_layers, self.layer_norms):
            # Pre-norm + Mamba + Residual connection
            h = h + mamba_layer(norm(h))
            
        h = self.norm_f(h)
        
        # REMOVED max pooling to return sequence of hidden states for Token-Level RTD
        return h
