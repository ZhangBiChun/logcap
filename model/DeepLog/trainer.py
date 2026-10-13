import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from merlion.evaluate.anomaly import accumulate_tsad_score
from merlion.utils import TimeSeries
import pandas as pd
from tqdm.auto import tqdm

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from models.reasonable_metric import reasonable_accumulator
from models.thns24_personalized_scores import maybe_personalize_scores
from .model import DeepLogTHNS24


def _infer_vocab(
    train_data: Sequence[Sequence[Tuple[int, int, int]]],
    test_data: Sequence[Sequence[Tuple[int, int, int]]],
) -> Tuple[int, int, int]:
    act_max, hour_max, sem_max = 0, 0, 0
    for seq in list(train_data) + list(test_data):
        for a, h, s in seq:
            act_max = max(act_max, int(a))
            hour_max = max(hour_max, int(h))
            sem_max = max(sem_max, int(s))
    return act_max + 1, hour_max + 1, sem_max + 1


class DeepLogDataset(torch.utils.data.Dataset):
    def __init__(self, sessions, labels, window_size, session_user_idx=None):
        self.window_size = window_size
        self.flat_seqs = []
        self.starts = []
        per_seq_user: List[int] = []
        if session_user_idx is None:
            session_user_idx = [0] * len(sessions)
        elif len(session_user_idx) != len(sessions):
            raise ValueError("session_user_idx length must match sessions")

        for seq, lb, u in zip(sessions, labels, session_user_idx):
            if len(seq) <= window_size:
                continue
            seq_arr = np.array(seq, dtype=np.int64)
            seq_idx = len(self.flat_seqs)
            self.flat_seqs.append(seq_arr)
            per_seq_user.append(int(u))
            for i in range(len(seq) - window_size):
                # If the target event is an anomaly, we shouldn't train on it to keep the model pure
                # (For training set all labels are usually 0, but just in case)
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
def _session_score(
    model: DeepLogTHNS24,
    seq: Sequence[Tuple[int, int, int]],
    window_size: int,
    num_candidates: int,
    tolerance: int,
    device: torch.device,
    user_idx: int = 0,
) -> np.ndarray:
    # Pad short sequences so they can be evaluated
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
        u_batch = torch.full((x_batch.shape[0],), int(user_idx), dtype=torch.long, device=device)
        logits_act, logits_time, logits_sem = model(x_batch, u_batch)

        batch_next = next_acts[i:i+batch_size]
        y_act = torch.tensor(batch_next[:, 0], dtype=torch.long, device=device)
        y_time = torch.tensor(batch_next[:, 1], dtype=torch.long, device=device)
        y_sem = torch.tensor(batch_next[:, 2], dtype=torch.long, device=device)

        loss_act = torch.nn.functional.cross_entropy(logits_act, y_act, reduction='none')
        loss_time = torch.nn.functional.cross_entropy(logits_time, y_time, reduction='none')
        loss_sem = torch.nn.functional.cross_entropy(logits_sem, y_sem, reduction='none')

        # We sum the losses of all three factors to get the full "surprise" of the event
        total_loss = (loss_act + loss_time + loss_sem) / 3.0
        scores_list.append(total_loss.cpu().numpy())

    scores_arr = np.concatenate(scores_list) if scores_list else np.array([])
    
    # Return an array of length exactly equal to len(seq)
    # The first `window_size` events do not have a prediction, so we assign a score of 0.0
    if len(seq) <= window_size:
        return np.full(len(seq), scores_arr[0] if len(scores_arr) > 0 else 0.0, dtype=np.float32)
    else:
        pad = np.zeros(window_size, dtype=np.float32)
        return np.concatenate([pad, scores_arr]).astype(np.float32)


def _direct_eval(test_labels: np.ndarray, predict: np.ndarray):
    test_labels = np.asarray(test_labels).reshape(-1).astype(int)
    predict = np.asarray(predict).reshape(-1).astype(int)

    if np.count_nonzero(test_labels) != 0:
        events_gt = convert_vector_to_events(test_labels)
        events_pred = convert_vector_to_events(predict)
        trange = (0, len(predict))
        affiliation = pr_from_events(events_pred, events_gt, trange)
    else:
        affiliation = {"precision": 0.0, "recall": 0.0}

    target_ts = TimeSeries.from_pd(pd.DataFrame(test_labels))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)

    return affiliation, score


def Trainer(train_data, test_data, train_labels, test_labels, config):
    print("DeepLog training started ....")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu"
    )
    # Event-level labels are a list of lists, we don't want to reshape them right away,
    # because they are used to skip anomalous windows per sequence.
    # We leave them as lists of lists.
    # train_labels = np.asarray(train_labels).reshape(-1).astype(int)
    # test_labels = np.asarray(test_labels).reshape(-1).astype(int)

    act_vocab, hour_vocab, sem_vocab = _infer_vocab(train_data, test_data)
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

    ds = DeepLogDataset(train_data, train_labels, config.window_size, session_user_idx=train_u)
    if len(ds) == 0:
        raise RuntimeError("No valid normal windows found for DeepLog training.")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1024, config.batch_size), # Override to massive batch_size to survive Event-Level scale
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    if num_users > 1 and os.path.exists(config.model_path):
        print(
            f"[DeepLog] User embedding enabled (num_users={num_users}); delete stale checkpoint if load fails: "
            f"{config.model_path}"
        )

    model = DeepLogTHNS24(
        act_vocab_size=act_vocab,
        time_vocab_size=hour_vocab,
        sem_vocab_size=sem_vocab,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_users=num_users,
        user_emb_dim=user_emb_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    stale = 0
    epoch_bar = tqdm(range(config.num_epoch), desc="[DeepLog] Epoch", leave=True)
    
    # Check if model exists, if yes, skip training. BUT for THNS24, since we regenerated the data, we might want to retrain.
    # We will ignore old model path if we are on a new dataset. But the user has the old model.
    # To prevent loading old incompatible model, we can add a check or just force retrain.
    # Since the user specifically complained about the code running and getting stuck at inference, let's just make sure it loads.
    if os.path.exists(config.model_path):
        print(f"[DeepLog] Found saved model at {config.model_path}, skipping training and directly inferring.")
        epoch_bar.close()
    else:
        for epoch in epoch_bar:
            model.train()
            running = 0.0
            batch_bar = tqdm(dl, desc=f"[DeepLog] Epoch {epoch + 1:03d}", leave=False)
            for x, y, u in batch_bar:
                x = x.to(device)
                y = y.to(device)
                u = u.to(device)
                optimizer.zero_grad()
                logits_act, logits_time, logits_sem = model(x, u)
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
            print(f"[DeepLog][Epoch {epoch + 1:03d}] train_loss={avg_loss:.6f}")
            if avg_loss + 1e-7 < best_loss:
                best_loss = avg_loss
                stale = 0
                os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
                torch.save(model.state_dict(), config.model_path)
            else:
                stale += 1
                if stale >= config.patience:
                    print(f"[DeepLog] Early stopping at epoch {epoch + 1}")
                    break

    # ALWAYS load best model weight before inference
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    model.eval()

    from models import ad_predict

    all_test_scores = []
    all_test_labels = []
    for seq, lbl, uid in tqdm(
        zip(test_data, test_labels, test_u),
        desc="[DeepLog] Inferring",
        total=len(test_data),
    ):
        scores = _session_score(
            model=model,
            seq=seq,
            window_size=config.window_size,
            num_candidates=config.num_candidates,
            tolerance=config.tolerance,
            device=device,
            user_idx=int(uid),
        )
        all_test_scores.append(scores)
        all_test_labels.append(lbl)

    test_scores = np.concatenate(all_test_scores).astype(np.float32)
    flat_test_labels = np.concatenate(all_test_labels).astype(np.int64)

    def _score_train_session(seq, uid: int) -> np.ndarray:
        return _session_score(
            model=model,
            seq=seq,
            window_size=config.window_size,
            num_candidates=config.num_candidates,
            tolerance=config.tolerance,
            device=device,
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
        log_prefix="DeepLog",
    )

    _dbg = np.asarray(test_scores, dtype=np.float64)
    print(
        f"[DeepLog] Event score stats: n={_dbg.size}, min={_dbg.min():.6g}, max={_dbg.max():.6g}, "
        f"mean={_dbg.mean():.6g}, std={_dbg.std():.6g}"
    )
    if os.environ.get("DEEPLOG_SAVE_SCORES", "").strip().lower() in ("1", "true", "yes"):
        dbg_dir = os.path.join("results", "deeplog_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        dbg_path = os.path.join(dbg_dir, "event_scores.npy")
        np.save(dbg_path, test_scores)
        print(f"[DeepLog] Wrote {dbg_path} (set DEEPLOG_SAVE_SCORES=0 to disable)")

    test_aff, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(
        flat_test_labels,
        test_scores,
        config.threshold_determine,
        config.detect_nu,
    )
    score_reasonable = reasonable_accumulator(1, 0)
    print("DeepLog predicting ended ....")

    return test_scores, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict

