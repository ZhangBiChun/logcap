import os
import torch
import numpy as np
import pandas as pd
from dataloader.data_preprocessing import thns24
from models.DeepLog.model import DeepLogTHNS24
from models.DeepLog.trainer import _infer_vocab
from tqdm.auto import tqdm
import math
import random

# Load data
train_data, test_data, train_labels, test_labels = thns24(
    train_csv='data/itdbert_dual/THNS24_2010.csv',
    test_csv='data/itdbert_dual/THNS24_2011.csv',
    max_sessions=None,
    sample_mode='full',
    seed=6
)

# Load user labels
_, user_test_data, _, user_test_labels = thns24(
    train_csv='data/itdbert_dual/THNS24_2010.csv',
    test_csv='data/itdbert_dual/THNS24_2011_userlabel.csv',
    max_sessions=None,
    sample_mode='full',
    seed=6
)

user_label_map = {}
for seq, lb in zip(user_test_data, np.asarray(user_test_labels).reshape(-1)):
    key = tuple(seq)
    user_label_map[key] = int(lb)

aligned_labels = []
for seq in test_data:
    key = tuple(seq)
    aligned_labels.append(user_label_map.get(key, 0))

test_labels = np.asarray(aligned_labels)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
act_vocab, hour_vocab, sem_vocab = _infer_vocab(train_data, test_data)
model = DeepLogTHNS24(
    act_vocab_size=act_vocab,
    time_vocab_size=hour_vocab,
    sem_vocab_size=sem_vocab,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
).to(device)

model_path = 'best_network/THNS24/deeplog_lstm.pt'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Run inference on FULL dataset (subset for speed here)
anomaly_indices = [i for i, l in enumerate(test_labels) if l == 1]
normal_indices = [i for i, l in enumerate(test_labels) if l == 0]
random.seed(42)
# sample 200 normal
sampled_normal_indices = random.sample(normal_indices, 1000)
sampled_anomaly_indices = random.sample(anomaly_indices, 1000) if len(anomaly_indices) > 1000 else anomaly_indices

subset_indices = sampled_anomaly_indices + sampled_normal_indices
subset_data = [test_data[i] for i in subset_indices]
subset_labels = [test_labels[i] for i in subset_indices]

window_size = 64
num_candidates = 2

results = []

with torch.no_grad():
    for i, seq in enumerate(tqdm(subset_data, disable=True)):
        
        # Pad short sequences with 0s to evaluate them!
        padded_seq = list(seq)
        if len(padded_seq) <= window_size:
            pad_len = window_size + 1 - len(padded_seq)
            padded_seq = [(0, 0, 0)] * pad_len + padded_seq
            
        num_windows = len(padded_seq) - window_size
        windows = []
        next_acts = []
        for j in range(num_windows):
            windows.append(padded_seq[j : j + window_size])
            next_acts.append(padded_seq[j + window_size][0])
            
        x = torch.tensor(windows, dtype=torch.long, device=device)
        next_acts = np.array(next_acts)
        
        errors = 0
        batch_size = 1024
        
        ce_loss_max = 0.0
        ce_loss_avg = 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        for j in range(0, num_windows, batch_size):
            x_batch = x[j:j+batch_size]
            logits = model(x_batch)
            topk = min(int(num_candidates), int(logits.shape[1]))
            preds = torch.topk(logits, k=topk, dim=1).indices.cpu().numpy()
            
            batch_next_acts = next_acts[j:j+batch_size]
            batch_next_acts_t = torch.tensor(batch_next_acts, dtype=torch.long, device=device)
            
            losses = criterion(logits, batch_next_acts_t)
            ce_loss_max = max(ce_loss_max, losses.max().item())
            
            for k in range(len(batch_next_acts)):
                if batch_next_acts[k] not in preds[k]:
                    errors += 1
                    
        total = num_windows
        
        results.append({
            'idx': subset_indices[i],
            'label': subset_labels[i],
            'len': len(seq),
            'total': total,
            'errors': errors,
            'error_ratio': errors / max(total, 1),
            'ce_loss_max': ce_loss_max,
            'ce_hybrid': ce_loss_max * (errors / max(total, 1)),
            'ce_hybrid_smooth': ce_loss_max * (errors / (total + 50)),
            'ce_max_smooth': ce_loss_max / (total + 50)
        })

df = pd.DataFrame(results)

print("\n=== AUC-ROC Comparison ===")
from sklearn.metrics import roc_auc_score
metrics = ['errors', 'error_ratio', 'ce_loss_max', 'ce_hybrid', 'ce_hybrid_smooth', 'ce_max_smooth']
for metric in metrics:
    print(f"By {metric:15s}: {roc_auc_score(df['label'], df[metric]):.5f}")

for metric in metrics:
    print(f"\nTop-K hits for {metric}:")
    df_sorted = df.sort_values(metric, ascending=False)
    for k in [10, 50, 100, 500]:
        hits = df_sorted.head(k)['label'].sum()
        print(f"Precision@{k}: {hits/k:.3f} ({hits} hits)")
