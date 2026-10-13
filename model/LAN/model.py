import torch
import torch.nn as nn
import torch.nn.functional as F

class LANModel(nn.Module):
    """
    LAN (Learning Adaptive Neighbors) surrogate model.
    Uses Sequence embeddings + Adaptive Graph Structure Learning + GCN.
    """
    def __init__(self, act_vocab_size, time_vocab_size, sem_vocab_size, hidden_size=128, num_layers=2, dropout=0.1, num_neighbors=128):
        super().__init__()
        self.embed_act = nn.Embedding(act_vocab_size + 2, hidden_size)
        self.embed_time = nn.Embedding(time_vocab_size + 2, hidden_size)
        self.embed_sem = nn.Embedding(sem_vocab_size + 2, hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # Temporal Dependency (Sequence Model)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Global Adaptive Neighbors (Memory Bank to avoid batch dependency)
        self.num_neighbors = num_neighbors
        self.neighbor_keys = nn.Parameter(torch.randn(num_neighbors, hidden_size))
        self.neighbor_vals = nn.Parameter(torch.randn(num_neighbors, hidden_size))
        
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        # GCN Layer
        self.gcn_weight = nn.Linear(hidden_size, hidden_size)
        
        # Prediction Heads
        self.fc_act = nn.Linear(hidden_size, act_vocab_size)
        self.fc_time = nn.Linear(hidden_size, time_vocab_size)
        self.fc_sem = nn.Linear(hidden_size, sem_vocab_size)
        
    def forward(self, x):
        # x: (batch_size, window_size, 3)
        b, t, _ = x.shape
        e_act = self.embed_act(x[:, :, 0])
        e_time = self.embed_time(x[:, :, 1])
        e_sem = self.embed_sem(x[:, :, 2])
        
        emb = self.fusion(torch.cat([e_act, e_time, e_sem], dim=-1))
        
        # 1. Temporal Dependency
        lstm_out, _ = self.lstm(emb)
        seq_rep = lstm_out[:, -1, :] # (batch_size, hidden_size)
        
        # 2. Adaptive Neighbor Learning (Global Memory Bank)
        Q = self.query_proj(seq_rep)
        K = self.key_proj(self.neighbor_keys)
        
        scores = torch.matmul(Q, K.transpose(0, 1)) / (seq_rep.size(-1) ** 0.5)
        adj = F.softmax(scores, dim=-1) # (batch_size, num_neighbors)
        
        # 3. Graph Convolution
        gcn_out = torch.matmul(adj, self.neighbor_vals)
        gcn_out = F.relu(self.gcn_weight(gcn_out))
        
        # 4. Residual Connection
        final_rep = seq_rep + gcn_out
        
        # 5. Predict next event
        out_act = self.fc_act(final_rep)
        out_time = self.fc_time(final_rep)
        out_sem = self.fc_sem(final_rep)
        
        return out_act, out_time, out_sem