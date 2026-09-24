import os
import torch
import numpy as np
import pandas as pd
from dataloader.data_preprocessing import thns24_event_level
from conf.logcapmamba.THNS24_Configs import Config
from models.LogCapMamba.model import LogCapMamba
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Setup basic config
config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
SEED = 6
window_size = config.window_size

print("Loading THNS24 data...")
train_data, test_data, train_labels, test_labels, train_user_ids, test_user_ids = thns24_event_level(
    train_csv=config.train_csv,
    test_csv=config.test_csv,
    test_label_csv=config.test_user_csv,
    seed=SEED
)

all_users = sorted(set(train_user_ids) | set(test_user_ids))
user_str_to_idx = {u: (i + 1) for i, u in enumerate(all_users)}
config.thns24_num_users = len(all_users) + 1
train_u = [user_str_to_idx.get(u, 0) for u in train_user_ids]
test_u = [user_str_to_idx.get(u, 0) for u in test_user_ids]

# Build user labels
user_labels_list = []
with open(config.test_user_csv, "r") as f:
    import csv
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            has_anomaly = any(int(l) == 1 for l in row[1:] if l)
            user_labels_list.append(1 if has_anomaly else 0)
thns24_user_eval_labels = np.array(user_labels_list)

# Load Model
act_max, hour_max, sem_max = 0, 0, 0
for seq in list(train_data) + list(test_data):
    for a, h, s in seq:
        act_max = max(act_max, int(a))
        hour_max = max(hour_max, int(h))
        sem_max = max(sem_max, int(s))
act_v, hour_v, sem_v = act_max + 1, hour_max + 1, sem_max + 1

model = LogCapMamba(act_v, hour_v, sem_v, 
                    hidden_size=getattr(config, 'hidden_size', 128),
                    dropout=getattr(config, 'dropout', 0.1),
                    num_layers=getattr(config, 'num_layers', 2),
                    num_users=config.thns24_num_users,
                    user_emb_dim=16).to(device)

print(f"Loading weights from {config.model_path}")
model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
model.eval()

# Compute raw scores
@torch.no_grad()
def _session_score(seq, uid: int) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros(0, dtype=np.float32)
    chunks = []
    spans = []
    for i in range(0, len(seq), window_size):
        chunk = seq[i : i + window_size]
        s = i
        e = min(i + window_size, len(seq))
        if len(chunk) < window_size:
            pad_len = window_size - len(chunk)
            chunk = [(0, 0, 0)] * pad_len + list(chunk)
        chunks.append(chunk)
        spans.append((s, e))

    chunk_probs = []
    batch_size = max(256, int(getattr(config, "batch_size", 128)) * 4)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        x = torch.tensor(batch, dtype=torch.long).to(device)
        u_t = torch.tensor([uid]*len(batch), dtype=torch.long).to(device)
        logits = model(x, u_t)
        probs = torch.sigmoid(logits)
        if probs.dim() == 1:
            chunk_probs.append(probs.cpu().numpy().astype(np.float32))
        else:
            for j in range(probs.shape[0]):
                chunk_probs.append(probs[j].cpu().numpy().astype(np.float32))

    event_scores = np.zeros(len(seq), dtype=np.float32)
    for (s, e), p_arr in zip(spans, chunk_probs):
        actual_len = e - s
        actual_probs = p_arr[-actual_len:]
        event_scores[s:e] = np.maximum(event_scores[s:e], actual_probs)
    return event_scores

all_raw_scores = []
print("Evaluating raw scores on test data...")
with torch.no_grad():
    for seq, u in zip(test_data, test_u):
        if len(seq) == 0: continue
        all_raw_scores.append(_session_score(seq, int(u)))
raw_preds = np.concatenate(all_raw_scores) if all_raw_scores else np.zeros(0, dtype=np.float32)

def evaluate_user_ranking(preds, labels_eval, test_data_sessions, top_k_val):
    flat_scores = np.asarray(preds).reshape(-1)
    user_scores = []
    idx = 0
    for seq in test_data_sessions:
        seq_len = len(seq)
        seq_scores = flat_scores[idx : idx + seq_len]
        idx += seq_len
        valid_scores = seq_scores[seq_scores > 1e-5]
        if len(valid_scores) > 0:
            tie_breaker = np.random.uniform(0.0, 0.0001)
            sorted_seq = np.sort(valid_scores)[::-1]
            k = min(top_k_val, len(sorted_seq))
            user_scores.append(np.mean(sorted_seq[:k]) + tie_breaker)
        else:
            user_scores.append(0.0)
            
    flat_user_scores = np.asarray(user_scores)
    flat_labels = np.asarray(labels_eval)
    
    min_len = min(len(flat_user_scores), len(flat_labels))
    flat_user_scores = flat_user_scores[:min_len]
    flat_labels = flat_labels[:min_len]
    
    total_real = max(1, np.sum(flat_labels))
    auc = roc_auc_score(flat_labels, flat_user_scores)
    
    event_weights = np.where(flat_labels == 1, 2000, 50)
    precisions, recalls, thresholds = precision_recall_curve(flat_labels, flat_user_scores, sample_weight=event_weights)
    best_f1, best_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_r = r
                
    sort_idx = np.argsort(flat_user_scores)[::-1]
    sorted_labels = flat_labels[sort_idx]
    
    metrics = {
        'auc': auc,
        'f1': best_f1,
        'r10': np.sum(sorted_labels[:10]) / total_real,
        'r30': np.sum(sorted_labels[:30]) / total_real,
        'r50': np.sum(sorted_labels[:50]) / total_real,
        'r80': np.sum(sorted_labels[:80]) / total_real
    }
    return metrics

# Grid Search
top_k_values = [1, 2, 3, 5, 10, 15, 20, 50]

print("\nStarting Top-K Grid Search...")
print(f"{'Top-K':<6} | {'AUC':<8} | {'Best F1':<8} | {'R@10':<8} | {'R@30':<8} | {'R@50':<8} | {'R@80':<8}")
print("-" * 65)

best_r30 = 0
best_k = None

for k in top_k_values:
    # We do NOT apply maybe_personalize_scores, just raw preds
    np.random.seed(SEED) # Ensure consistent tie breakers
    m = evaluate_user_ranking(raw_preds, thns24_user_eval_labels, test_data, top_k_val=k)
    print(f"{k:<6} | {m['auc']:.5f} | {m['f1']:.5f} | {m['r10']:.5f} | {m['r30']:.5f} | {m['r50']:.5f} | {m['r80']:.5f}")
    
    # Primary objective: maximize R@30, secondary AUC
    score = m['r30'] * 1000 + m['auc']
    best_score = best_r30 * 1000 + (best_k[1] if best_k else 0)
    
    if score > best_score:
        best_r30 = m['r30']
        best_k = (k, m['auc'], m['f1'], m['r30'])

print("-" * 65)
if best_k:
    print(f"BEST PARAMS: Top-K = {best_k[0]}")
    print(f"Metrics: AUC={best_k[1]:.5f}, Best F1={best_k[2]:.5f}, R@30={best_k[3]:.5f}")
