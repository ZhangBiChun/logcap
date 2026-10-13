import os
from typing import List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models import ad_predict
from models.reasonable_metric import reasonable_accumulator
from models.thns24_personalized_scores import maybe_personalize_scores
from .model import LANModel

def _infer_vocab(train_data, test_data):
    act_max, hour_max, sem_max = 0, 0, 0
    for seq in list(train_data) + list(test_data):
        for a, h, s in seq:
            act_max = max(act_max, int(a))
            hour_max = max(hour_max, int(h))
            sem_max = max(sem_max, int(s))
    return act_max + 1, hour_max + 1, sem_max + 1

class LANDataset(torch.utils.data.Dataset):
    def __init__(self, sessions, labels, window_size, session_user_idx=None):
        self.window_size = window_size
        self.flat_seqs = []
        self.starts = []
        per_seq_user = []
        if session_user_idx is None:
            session_user_idx = [0] * len(sessions)

        for seq, lb, u in zip(sessions, labels, session_user_idx):
            if len(seq) <= window_size:
                continue
            seq_arr = np.array(seq, dtype=np.int64)
            seq_idx = len(self.flat_seqs)
            self.flat_seqs.append(seq_arr)
            per_seq_user.append(int(u))
            for i in range(len(seq) - window_size):
                if isinstance(lb, (list, np.ndarray)) and len(lb) > i + window_size and int(lb[i + window_size]) != 0:
                    continue
                elif isinstance(lb, (int, float, str)) and int(lb) != 0:
                    continue
                self.starts.append((seq_idx, i))

        self.session_user_idx = np.asarray(per_seq_user, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        seq_idx, i = self.starts[idx]
        seq = self.flat_seqs[seq_idx]
        x = seq[i : i + self.window_size]
        y = seq[i + self.window_size]
        u = int(self.session_user_idx[seq_idx])
        return (
            torch.from_numpy(x).long(),
            torch.from_numpy(y).long(),
            torch.tensor(u, dtype=torch.long),
        )

@torch.no_grad()
def _session_score(model, seq, window_size, device, user_idx=0):
    padded_seq = list(seq)
    if len(padded_seq) <= window_size:
        pad_len = window_size + 1 - len(padded_seq)
        padded_seq = [(0, 0, 0)] * pad_len + padded_seq

    num_windows = len(padded_seq) - window_size
    windows = []
    next_acts = []
    for i in range(num_windows):
        windows.append(padded_seq[i : i + window_size])
        next_acts.append(padded_seq[i + window_size])
        
    x = torch.tensor(windows, dtype=torch.long, device=device)
    next_acts = np.array(next_acts)
    
    batch_size = 1024
    scores_list = []

    for i in range(0, num_windows, batch_size):
        x_batch = x[i:i+batch_size]
        logits_act, logits_time, logits_sem = model(x_batch)
        
        batch_next = next_acts[i:i+batch_size]
        y_act = torch.tensor(batch_next[:, 0], dtype=torch.long, device=device)
        y_time = torch.tensor(batch_next[:, 1], dtype=torch.long, device=device)
        y_sem = torch.tensor(batch_next[:, 2], dtype=torch.long, device=device)

        loss_act = torch.nn.functional.cross_entropy(logits_act, y_act, reduction='none')
        loss_time = torch.nn.functional.cross_entropy(logits_time, y_time, reduction='none')
        loss_sem = torch.nn.functional.cross_entropy(logits_sem, y_sem, reduction='none')

        total_loss = (loss_act + loss_time + loss_sem) / 3.0
        scores_list.append(total_loss.cpu().numpy())

    scores_arr = np.concatenate(scores_list) if scores_list else np.array([])
    
    if len(seq) <= window_size:
        return np.full(len(seq), scores_arr[0] if len(scores_arr) > 0 else 0.0, dtype=np.float32)
    else:
        pad = np.zeros(window_size, dtype=np.float32)
        return np.concatenate([pad, scores_arr]).astype(np.float32)

def Trainer(train_data, test_data, train_labels, test_labels, config):
    print("LAN training started ....")
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu")

    act_vocab, hour_vocab, sem_vocab = _infer_vocab(train_data, test_data)
    train_u = getattr(config, "thns24_train_user_idx", [0]*len(train_data))
    test_u = getattr(config, "thns24_test_user_idx", [0]*len(test_data))

    ds = LANDataset(train_data, train_labels, config.window_size, session_user_idx=train_u)
    dl = torch.utils.data.DataLoader(ds, batch_size=max(1024, config.batch_size), shuffle=True, drop_last=False)

    model = LANModel(
        act_vocab_size=act_vocab,
        time_vocab_size=hour_vocab,
        sem_vocab_size=sem_vocab,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    stale = 0
    epoch_bar = tqdm(range(config.num_epoch), desc="[LAN] Epoch", leave=True)
    
    if os.path.exists(config.model_path):
        print(f"[LAN] Found saved model at {config.model_path}, skipping training and directly inferring.")
        epoch_bar.close()
    else:
        for epoch in epoch_bar:
            model.train()
            running = 0.0
            batch_bar = tqdm(dl, desc=f"[LAN] Epoch {epoch + 1:03d}", leave=False)
            for x, y, u in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits_act, logits_time, logits_sem = model(x)
                
                loss_act = criterion(logits_act, y[:, 0])
                loss_time = criterion(logits_time, y[:, 1])
                loss_sem = criterion(logits_sem, y[:, 2])
                loss = (loss_act + loss_time + loss_sem) / 3.0
                
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")
                
            avg_loss = running / max(1, len(dl))
            epoch_bar.set_postfix(train_loss=f"{avg_loss:.6f}", best=f"{best_loss:.6f}")
            print(f"[LAN][Epoch {epoch + 1:03d}] train_loss={avg_loss:.6f}")
            if avg_loss + 1e-7 < best_loss:
                best_loss = avg_loss
                stale = 0
                os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
                torch.save(model.state_dict(), config.model_path)
            else:
                stale += 1
                if stale >= config.patience:
                    print(f"[LAN] Early stopping at epoch {epoch + 1}")
                    break

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    model.eval()

    all_test_scores = []
    all_test_labels = []
    for seq, lbl, uid in tqdm(zip(test_data, test_labels, test_u), desc="[LAN] Inferring", total=len(test_data)):
        scores = _session_score(model, seq, config.window_size, device, int(uid))
        all_test_scores.append(scores)
        all_test_labels.append(lbl)

    test_scores = np.concatenate(all_test_scores).astype(np.float32)
    flat_test_labels = np.concatenate(all_test_labels).astype(np.int64)

    def _score_train_session(seq, uid: int) -> np.ndarray:
        return _session_score(model, seq, config.window_size, device, int(uid))

    test_scores = maybe_personalize_scores(
        config, test_scores, test_data, test_labels, test_u,
        train_data, train_labels, train_u,
        session_score_fn=_score_train_session, log_prefix="LAN"
    )

    test_aff, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(
        flat_test_labels, test_scores, config.threshold_determine, config.detect_nu
    )
    score_reasonable = reasonable_accumulator(1, 0)
    print("LAN predicting ended ....")

    return test_scores, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict