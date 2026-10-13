import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models import ad_predict
from models.reasonable_metric import reasonable_accumulator
from models.thns24_personalized_scores import maybe_personalize_scores
from .model import LogBERTModel


def _infer_vocab(
    train_data: Sequence[Sequence[Tuple[int, int, int]]],
    test_data: Sequence[Sequence[Tuple[int, int, int]]],
) -> Tuple[int, int, int]:
    act_max, time_max, sem_max = 0, 0, 0
    for seq in list(train_data) + list(test_data):
        for a, t, s in seq:
            act_max = max(act_max, int(a))
            time_max = max(time_max, int(t))
            sem_max = max(sem_max, int(s))
    return act_max + 1, time_max + 1, sem_max + 1


def _build_train_windows(
    train_data: Sequence[Sequence[Tuple[int, int, int]]],
    train_labels: Sequence,
    window_size: int,
    session_user_idx: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    xs: List[np.ndarray] = []
    us: List[int] = []
    if session_user_idx is None:
        session_user_idx = [0] * len(train_data)
    elif len(session_user_idx) != len(train_data):
        raise ValueError("session_user_idx length must match train_data")

    for seq, lb, u in zip(train_data, train_labels, session_user_idx):
        if len(seq) < window_size:
            continue
        u_int = int(u)
        if not isinstance(lb, (list, tuple, np.ndarray)):
            if int(lb) != 0:
                continue
            for i in range(len(seq) - window_size + 1):
                xs.append(np.asarray(seq[i : i + window_size], dtype=np.int64))
                us.append(u_int)
            continue
        lb_arr = np.asarray(lb, dtype=np.int64).reshape(-1)
        if len(lb_arr) < len(seq):
            lb_arr = np.pad(lb_arr, (0, len(seq) - len(lb_arr)), constant_values=0)
        elif len(lb_arr) > len(seq):
            lb_arr = lb_arr[: len(seq)]
        for i in range(len(seq) - window_size + 1):
            if np.any(lb_arr[i : i + window_size] != 0):
                continue
            xs.append(np.asarray(seq[i : i + window_size], dtype=np.int64))
            us.append(u_int)

    return xs, us


class LogBERTDataset(Dataset):
    def __init__(self, xs, user_idx, act_vocab, time_vocab, sem_vocab):
        self.xs = xs
        self.user_idx = user_idx
        self.MASK_ACT = act_vocab + 1
        self.MASK_TIME = time_vocab + 1
        self.MASK_SEM = sem_vocab + 1

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        context = self.xs[idx].copy()
        u = int(self.user_idx[idx])
        
        seq_len = len(context)
        labels = np.full((seq_len, 3), -100, dtype=np.int64)
        
        # LogBERT standard Masking: randomly mask tokens
        # We mask 20% to differentiate from ITDBERT slightly
        for i in range(seq_len):
            if np.random.rand() < 0.20:
                labels[i] = context[i]  
                context[i][0] = self.MASK_ACT
                context[i][1] = self.MASK_TIME
                context[i][2] = self.MASK_SEM
                
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(u, dtype=torch.long),
        )


@torch.no_grad()
def _session_score(
    model: LogBERTModel,
    seq: Sequence[Tuple[int, int, int]],
    window_size: int,
    topk: int,
    device: torch.device,
    act_vocab: int,
    time_vocab: int,
    sem_vocab: int,
    user_idx: int = 0,
) -> np.ndarray:
    MASK_ACT = act_vocab + 1
    MASK_TIME = time_vocab + 1
    MASK_SEM = sem_vocab + 1

    half = window_size // 2
    
    if len(seq) < window_size:
        return np.zeros(len(seq), dtype=np.float32)

    total = len(seq) - window_size + 1
    xs = []
    next_acts = []
    
    # Evaluate valid windows without OOD zero-padding
    for i in range(total):
        context = np.asarray(seq[i : i + window_size], dtype=np.int64)
        target_acts = context[half].copy()
        context[half][0] = MASK_ACT
        context[half][1] = MASK_TIME
        context[half][2] = MASK_SEM
        xs.append(context)
        next_acts.append(target_acts)
        
    batch_size = 1024
    scores_list = []
    for i in range(0, len(xs), batch_size):
        x_batch = torch.tensor(np.array(xs[i:i+batch_size]), dtype=torch.long).to(device)
        u_batch = torch.full((x_batch.shape[0],), int(user_idx), dtype=torch.long, device=device)
        logits_act, logits_time, logits_sem = model(x_batch, u_batch)  # [B, window_size, vocab_size]
        
        last_logits_act = logits_act[:, half, :]  
        last_logits_time = logits_time[:, half, :]
        last_logits_sem = logits_sem[:, half, :]
        
        batch_next = next_acts[i:i+batch_size]
        batch_next = np.array(batch_next)
        
        y_act = torch.tensor(batch_next[:, 0], dtype=torch.long, device=device)
        y_time = torch.tensor(batch_next[:, 1], dtype=torch.long, device=device)
        y_sem = torch.tensor(batch_next[:, 2], dtype=torch.long, device=device)
        
        loss_act = torch.nn.functional.cross_entropy(last_logits_act, y_act, reduction='none')
        loss_time = torch.nn.functional.cross_entropy(last_logits_time, y_time, reduction='none')
        loss_sem = torch.nn.functional.cross_entropy(last_logits_sem, y_sem, reduction='none')
        
        total_loss = (loss_act + loss_time + loss_sem) / 3.0
        scores_list.append(total_loss.cpu().numpy())

    scores_arr = np.concatenate(scores_list) if scores_list else np.array([])
    
    # Pad the left and right edges with 0.0 since we couldn't evaluate them cleanly
    pad_left = np.zeros(half, dtype=np.float32)
    pad_right_len = len(seq) - (half + total)
    pad_right = np.zeros(pad_right_len, dtype=np.float32)
    
    return np.concatenate([pad_left, scores_arr, pad_right]).astype(np.float32)


def Trainer(train_data, test_data, train_labels, test_labels, config):
    print("LogBERT training started ....")

    device = torch.device("cuda" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu")
    act_vocab, time_vocab, sem_vocab = _infer_vocab(train_data, test_data)

    train_u = getattr(config, "thns24_train_user_idx", None)
    test_u = getattr(config, "thns24_test_user_idx", None)
    num_users = int(getattr(config, "thns24_num_users", 1))
    user_emb_dim = int(getattr(config, "user_emb_dim", 16))
    if train_u is None:
        train_u = [0] * len(train_data)
    if test_u is None:
        test_u = [0] * len(test_data)
    if num_users < 1:
        num_users = 1

    x_train, u_train = _build_train_windows(
        train_data, train_labels, config.window_size, session_user_idx=train_u
    )
    if len(x_train) == 0:
        raise RuntimeError("No valid normal training windows found for THNS24.")

    ds = LogBERTDataset(x_train, u_train, act_vocab, time_vocab, sem_vocab)

    dl = DataLoader(
        ds,
        batch_size=max(1024, config.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    if num_users > 1 and os.path.exists(config.model_path):
        print(
            f"[LogBERT] User embedding enabled (num_users={num_users}); delete stale checkpoint if load fails: "
            f"{config.model_path}"
        )

    model = LogBERTModel(
        act_vocab_size=act_vocab,
        time_vocab_size=time_vocab,
        sem_vocab_size=sem_vocab,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        num_users=num_users,
        user_emb_dim=user_emb_dim,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    stale_epochs = 0
    
    if os.path.exists(config.model_path):
        print(f"[LogBERT] Found existing model checkpoint at {config.model_path}. Loading it directly.")
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
        epoch_bar = []
    else:
        epoch_bar = tqdm(range(config.num_epoch), desc="[LogBERT] Epoch", leave=True)
    for epoch in epoch_bar:
        model.train()
        running = 0.0
        batch_bar = tqdm(dl, desc=f"[LogBERT] Epoch {epoch + 1:03d}", leave=False)
        for x, y, u in batch_bar:
            x = x.to(device)
            y = y.to(device)
            u = u.to(device)

            optimizer.zero_grad()
            logits_act, logits_time, logits_sem = model(x, u)
            
            logits_act = logits_act.view(-1, logits_act.size(-1))
            logits_time = logits_time.view(-1, logits_time.size(-1))
            logits_sem = logits_sem.view(-1, logits_sem.size(-1))
            
            y_act = y[:, :, 0].contiguous().view(-1)
            y_time = y[:, :, 1].contiguous().view(-1)
            y_sem = y[:, :, 2].contiguous().view(-1)
            
            loss_act = criterion(logits_act, y_act)
            loss_time = criterion(logits_time, y_time)
            loss_sem = criterion(logits_sem, y_sem)
            
            loss = (loss_act + loss_time + loss_sem) / 3.0
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")

        avg_loss = running / max(1, len(dl))
        epoch_bar.set_postfix(train_loss=f"{avg_loss:.6f}", best=f"{best_loss:.6f}")
        print(f"[LogBERT][Epoch {epoch + 1:03d}] train_loss={avg_loss:.6f}")
        if avg_loss + 1e-7 < best_loss:
            best_loss = avg_loss
            stale_epochs = 0
            os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
            torch.save(model.state_dict(), config.model_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"[LogBERT] Early stopping at epoch {epoch + 1}")
                break

        if os.path.exists(config.model_path):
            model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
            
    model.eval()

    all_test_scores = []
    all_test_labels = []
    for seq, lbl, uid in tqdm(
        zip(test_data, test_labels, test_u),
        desc="[LogBERT] Inferring",
        total=len(test_data),
    ):
        if len(seq) == 0:
            continue

        scores = _session_score(
            model=model,
            seq=seq,
            window_size=config.window_size,
            topk=config.num_candidates,
            device=device,
            act_vocab=act_vocab,
            time_vocab=time_vocab,
            sem_vocab=sem_vocab,
            user_idx=int(uid),
        )
        all_test_scores.append(scores)
        
        if not isinstance(lbl, (list, tuple, np.ndarray)):
            all_test_labels.append(np.full(len(seq), int(lbl), dtype=np.int64))
        else:
            arr = np.asarray(lbl, dtype=np.int64).reshape(-1)
            if len(arr) < len(seq):
                arr = np.pad(arr, (0, len(seq) - len(arr)), constant_values=0)
            elif len(arr) > len(seq):
                arr = arr[: len(seq)]
            all_test_labels.append(arr)

    test_scores = np.concatenate(all_test_scores).astype(np.float32) if all_test_scores else np.zeros(0, dtype=np.float32)
    flat_test_labels = np.concatenate(all_test_labels).astype(np.int64) if all_test_labels else np.zeros(0, dtype=np.int64)

    def _score_train_session(seq, uid: int) -> np.ndarray:
        return _session_score(
            model=model,
            seq=seq,
            window_size=config.window_size,
            topk=config.num_candidates,
            device=device,
            act_vocab=act_vocab,
            time_vocab=time_vocab,
            sem_vocab=sem_vocab,
            user_idx=int(uid),
        )

    test_scores = maybe_personalize_scores(
        config,
        test_scores,
        test_data,
        test_labels,
        test_u,
        train_data,
        train_labels,
        train_u,
        session_score_fn=_score_train_session,
        log_prefix="LogBERT",
    )

    _dbg = np.asarray(test_scores, dtype=np.float64)
    print(
        f"[LogBERT] Event score stats: n={_dbg.size}, min={_dbg.min():.6g}, max={_dbg.max():.6g}, "
        f"mean={_dbg.mean():.6g}, std={_dbg.std():.6g}"
    )

    test_aff, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(
        flat_test_labels,
        test_scores,
        config.threshold_determine,
        config.detect_nu,
    )
    score_reasonable = reasonable_accumulator(1, 0)

    print("LogBERT predicting ended ....")

    return test_scores, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict
