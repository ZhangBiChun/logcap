import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from merlion.evaluate.anomaly import accumulate_tsad_score
from merlion.utils import TimeSeries
import pandas as pd

from .model import LogCapMamba
from .data_augmentation import augment_session
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from models.reasonable_metric import reasonable_accumulator

def _infer_vocab(train_data, test_data):
    act_max, hour_max, sem_max = 0, 0, 0
    for seq in list(train_data) + list(test_data):
        for a, h, s in seq:
            act_max = max(act_max, int(a))
            hour_max = max(hour_max, int(h))
            sem_max = max(sem_max, int(s))
    return act_max + 1, hour_max + 1, sem_max + 1


def _is_scalar_label(lb) -> bool:
    return not isinstance(lb, (list, tuple, np.ndarray))


def _session_all_normal(lb) -> bool:
    """Day-level: scalar 0; event-level: every position is 0."""
    if _is_scalar_label(lb):
        return int(lb) == 0
    arr = np.asarray(lb, dtype=np.int64).reshape(-1)
    return bool(arr.size == 0 or np.all(arr == 0))


def _session_has_anomaly(lb) -> bool:
    if _is_scalar_label(lb):
        return int(lb) != 0
    arr = np.asarray(lb, dtype=np.int64).reshape(-1)
    return bool(np.any(arr != 0))


def _valid_sems_per_act_from_train(train_data, train_labels) -> Dict[int, List[int]]:
    """Data-driven (act -> sem) support for semantic swap augmentation."""
    by_act: Dict[int, set] = defaultdict(set)
    for seq, lb in zip(train_data, train_labels):
        if not _session_all_normal(lb):
            continue
        for a, _h, s in seq:
            by_act[int(a)].add(int(s))
    return {a: sorted(sems) for a, sems in by_act.items()}

class SessionDataset(Dataset):
    def __init__(self, sessions, labels, users, window_size, pad_val=0):
        self.samples = []
        for seq, lb, u in zip(sessions, labels, users):
            if len(seq) == 0:
                continue
            
            if isinstance(lb, int) and lb == 0:
                mask = [0] * len(seq)
            else:
                mask = lb

            if len(seq) > window_size:
                seq = seq[-window_size:]
                mask = mask[-window_size:]
            else:
                pad_len = window_size - len(seq)
                seq = [(pad_val, pad_val, pad_val)] * pad_len + list(seq)
                mask = [0] * pad_len + list(mask)

            self.samples.append((seq, mask, int(u)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, u = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32), torch.tensor(u, dtype=torch.long)

def Trainer(train_data, test_data, train_labels, test_labels, config):
    print("LogCapMamba training started ....")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu"
    )
    window_size = getattr(config, 'window_size', 64)
    
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

    act_v, hour_v, sem_v = _infer_vocab(train_data, test_data)
    valid_sems_per_act = _valid_sems_per_act_from_train(train_data, train_labels)

    ablation_tag = str(getattr(config, "ablation_tag", "") or "").strip()
    
    if ablation_tag:
        cache_path = config.model_path.replace(".pt", f"_{ablation_tag}_aug_data.pt")
    else:
        cache_path = config.model_path.replace(".pt", "_aug_data.pt")
        
    if os.path.exists(cache_path):
        print(f"[LogCapMamba] Loading cached augmented data from {cache_path}...")
        cached_data = torch.load(cache_path, weights_only=False)
        aug_train_data = cached_data['data']
        aug_train_labels = cached_data['labels']
        aug_train_u = cached_data['u']
    else:
        normal_chunks: List[List[Tuple[int, int, int]]] = []
        normal_chunks_u: List[int] = []
        print("[LogCapMamba] Chunking long training sequences into smaller windows...")
        for seq, lb, u in zip(train_data, train_labels, train_u):
            if not _session_all_normal(lb):
                continue
            for i in range(0, len(seq), window_size):
                chunk = seq[i : i + window_size]
                if len(chunk) == window_size:
                    normal_chunks.append(list(chunk))
                    normal_chunks_u.append(int(u))

        aug_train_data: List = []
        aug_train_labels: List[int] = []
        aug_train_u: List[int] = []
        vocab_sizes = (act_v, hour_v, sem_v)
        for ci, (chunk, u) in enumerate(zip(normal_chunks, normal_chunks_u)):
            aug_train_data.append(chunk)
            aug_train_labels.append(0)
            aug_train_u.append(u)
            if random.random() < 0.5:
                pseudo_seq, mask = augment_session(
                    chunk,
                    vocab_sizes,
                    valid_sems_per_act=valid_sems_per_act,
                    normal_session_pool=normal_chunks,
                    pool_exclude_idx=ci,
                    max_len=window_size,
                    active_aug_ops=getattr(config, "active_aug_ops", None)
                )
                aug_train_data.append(pseudo_seq)
                aug_train_labels.append(mask)
                aug_train_u.append(u)
                
        print(f"[LogCapMamba] Saving augmented data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'data': aug_train_data,
            'labels': aug_train_labels,
            'u': aug_train_u
        }, cache_path)

    print(f"[LogCapMamba] Created/Loaded {len(aug_train_data)} training chunks.")

    model = LogCapMamba(act_v, hour_v, sem_v, 
                        hidden_size=getattr(config, 'hidden_size', 128),
                        dropout=getattr(config, 'dropout', 0.1),
                        num_layers=getattr(config, 'num_layers', 2),
                        num_users=num_users,
                        user_emb_dim=user_emb_dim).to(device)
    
    train_dataset = SessionDataset(aug_train_data, aug_train_labels, aug_train_u, window_size)
    train_bs = max(1024, int(getattr(config, "batch_size", 128)))
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

    # Optional ablation: separate checkpoint per tag (see conf/logcapmamba/THNS24_Configs.py).
    if ablation_tag:
        base, ext = os.path.splitext(config.model_path)
        config.model_path = f"{base}_{ablation_tag}{ext}"

    use_focal = bool(getattr(config, "use_focal_loss", False))
    smooth_win = int(getattr(config, "inference_score_smooth_window", 0) or 0)
    loss_name = "focal" if use_focal else "bce"
    print(
        f"[LogCapMamba] Ablation: loss={loss_name}, inference_smooth_window={smooth_win}, "
        f"checkpoint={config.model_path}"
    )

    class _FocalLoss(nn.Module):
        def __init__(self, alpha=0.9, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self._bce = nn.BCEWithLogitsLoss(reduction="none")

        def forward(self, inputs, targets):
            bce_loss = self._bce(inputs, targets)
            pt = torch.exp(-bce_loss)
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            return (alpha_t * (1 - pt) ** self.gamma * bce_loss).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(config, 'learning_rate', getattr(config, 'lr', 1e-3)))
    if use_focal:
        criterion = _FocalLoss(
            alpha=float(getattr(config, "focal_alpha", 0.9)),
            gamma=float(getattr(config, "focal_gamma", 2.0)),
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    epochs = getattr(config, 'num_epoch', getattr(config, 'num_epochs', 5))
    
    if os.path.exists(config.model_path):
        print(f"[LogCapMamba] Found saved model at {config.model_path}, skipping training.")
        model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    else:
        model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for x, y, u in train_loader:
                x = x.to(device)
                y = y.to(device)
                u = u.to(device)
                
                optimizer.zero_grad()
                logits = model(x, u).squeeze(-1) # Ensure [B] to match y [B]
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_loader))
            print(f"[LogCapMamba] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            if not np.isnan(avg_loss) and avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
                torch.save(model.state_dict(), config.model_path)
        
        if os.path.exists(config.model_path):
            model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    
    model.eval()

    # Event-level scores: assign each chunk's anomaly prob to all events in that chunk, then concat (aligns with DeepLog/ITDBERT).
    print("[LogCapMamba] Evaluating on test data (event-level scores)...")
    pad_val = 0

    @torch.no_grad()
    def _session_score(seq, uid: int) -> np.ndarray:
        if len(seq) == 0:
            return np.zeros(0, dtype=np.float32)
        chunks: List = []
        spans: List[Tuple[int, int]] = []
        stride = window_size  # Reverted back to non-overlapping
        for i in range(0, len(seq), stride):
            chunk = seq[i : i + window_size]
            s = i
            e = min(i + window_size, len(seq))
            if len(chunk) < window_size:
                pad_len = window_size - len(chunk)
                chunk = [(pad_val, pad_val, pad_val)] * pad_len + list(chunk)
            chunks.append(chunk)
            spans.append((s, e))

        chunk_probs: List[np.ndarray] = []
        chunk_batch_size = max(256, int(getattr(config, "batch_size", 128)) * 4)
        for i in range(0, len(chunks), chunk_batch_size):
            batch_chunks = chunks[i : i + chunk_batch_size]
            x = torch.tensor(batch_chunks, dtype=torch.long).to(device)
            u_t = torch.tensor([uid]*len(batch_chunks), dtype=torch.long).to(device)
            logits = model(x, u_t)
            probs = torch.sigmoid(logits) # [B, window_size]
            
            if probs.dim() == 1:
                # If batch size is 1, probs is [window_size]
                chunk_probs.append(probs.cpu().numpy().astype(np.float32))
            else:
                # probs is [B, window_size]
                for j in range(probs.shape[0]):
                    chunk_probs.append(probs[j].cpu().numpy().astype(np.float32))

        event_scores = np.zeros(len(seq), dtype=np.float32)
        for (s, e), p_arr in zip(spans, chunk_probs):
            actual_len = e - s
            # Because we left-padded with pad_val, the real events are the last actual_len elements
            actual_probs = p_arr[-actual_len:]
            event_scores[s:e] = np.maximum(event_scores[s:e], actual_probs)
        if smooth_win >= 3 and len(event_scores) >= smooth_win:
            kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
            pad_scores = np.pad(event_scores, (smooth_win // 2, smooth_win // 2), mode="edge")
            event_scores = np.convolve(pad_scores, kernel, mode="valid").astype(np.float32)
        return event_scores

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    
    with torch.no_grad():
        for seq, lb, u in zip(test_data, test_labels, test_u):
            if len(seq) == 0:
                continue

            event_scores = _session_score(seq, int(u))
            all_scores.append(event_scores)

            if _is_scalar_label(lb):
                all_labels.append(np.full(len(seq), int(lb), dtype=np.int64))
            else:
                arr = np.asarray(lb, dtype=np.int64).reshape(-1)
                if len(arr) < len(seq):
                    arr = np.pad(arr, (0, len(seq) - len(arr)), constant_values=0)
                elif len(arr) > len(seq):
                    arr = arr[: len(seq)]
                all_labels.append(arr)

    preds = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32)
    test_labels_arr = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)

    from models.thns24_personalized_scores import maybe_personalize_scores
    preds = maybe_personalize_scores(
        config,
        preds,
        test_data,
        test_labels,
        test_u,
        train_data,
        train_labels,
        train_u,
        session_score_fn=_session_score,
        log_prefix="LogCapMamba",
    )

    _dbg = np.asarray(preds, dtype=np.float64)
    if _dbg.size:
        print(
            f"[LogCapMamba] Event score stats: n={_dbg.size}, min={_dbg.min():.6g}, max={_dbg.max():.6g}, "
            f"mean={_dbg.mean():.6g}, std={_dbg.std():.6g}"
        )

    from models import ad_predict
    test_aff, test_rpa_score, test_pa_score, test_pw_score, predict_labels = ad_predict(
        test_labels_arr,
        preds,
        getattr(config, 'threshold_determine', 'best_f1'),
        getattr(config, 'detect_nu', 0.05),
    )
    score_reasonable = reasonable_accumulator(1, 0)
    
    print("[LogCapMamba] Generating t-SNE visualization...")
    try:
        _generate_tsne_visualization(
            model,
            train_data,
            train_labels,
            test_data,
            test_labels,
            act_v,
            hour_v,
            sem_v,
            window_size,
            device,
            config,
        )
    except Exception as e:
        print(f"[LogCapMamba] Failed to generate t-SNE visualization: {e}")
    
    return preds, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict_labels

def _generate_tsne_visualization(
    model,
    train_data,
    train_labels,
    test_data,
    test_labels,
    act_v,
    hour_v,
    sem_v,
    window_size,
    device,
    config,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import random
    import torch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    print("[LogCapMamba] Extracting Latent Features for LDA Type Analysis...")
    model.eval()

    test_u = getattr(config, "thns24_test_user_idx", None)
    if test_u is None:
        test_u = [0] * len(test_data)

    normals_seq, normals_u = [], []
    real_seq, real_u = [], []
    
    for seq, lb, u in zip(test_data, test_labels, test_u):
        if _session_has_anomaly(lb):
            real_seq.append(seq)
            real_u.append(int(u))
        else:
            normals_seq.append(seq)
            normals_u.append(int(u))
            
    # Sample normals to form background
    sample_idx = random.sample(range(len(normals_seq)), min(2000, len(normals_seq)))
    bg_seq = [normals_seq[i] for i in sample_idx]
    bg_u = [normals_u[i] for i in sample_idx]
    
    valid_sems_per_act = _valid_sems_per_act_from_train(train_data, train_labels)
    
    # Generate 4 pure types of pseudo anomalies
    pseudo_pool = random.sample(range(len(normals_seq)), min(500, len(normals_seq)))
    
    p1_s, p1_u = [], []
    p2_s, p2_u = [], []
    p3_s, p3_u = [], []
    p4_s, p4_u = [], []
    
    for i in pseudo_pool:
        seq = list(normals_seq[i])
        u = normals_u[i]
        n = len(seq)
        if n < 5: continue
        
        # T1: Semantic Shift
        s1 = seq[:]
        num_swaps = random.randint(1, 3)
        start_idx = random.randint(0, max(0, n - num_swaps))
        for j in range(start_idx, min(n, start_idx + num_swaps)):
            a, h, s = s1[j]
            s1[j] = (a, h, random.randint(0, sem_v - 1))
        p1_s.append(s1)
        p1_u.append(u)
        
        # T2: Burst
        s2 = seq[:]
        num_dups = random.randint(8, 15)
        idx = random.randint(0, n - 1)
        ev = s2[idx]
        for _ in range(num_dups): s2.insert(idx, ev)
        p2_s.append(s2)
        p2_u.append(u)
        
        # T3: Off-hours
        s3 = seq[:]
        w = random.randint(3, min(8, n))
        start = random.randint(0, n - w)
        chosen_hour = random.choice([0, 1, 2, 3, 4, 22, 23])
        for j in range(start, start + w):
            a, h, s = s3[j]
            s3[j] = (a, chosen_hour, s)
        p3_s.append(s3)
        p3_u.append(u)
        
        # T4: Alien Fragment
        s4 = seq[:]
        alien_src = random.choice(normals_seq)
        if len(alien_src) >= 5:
            frag_len = random.randint(5, min(10, len(alien_src)))
            src_start = random.randint(0, len(alien_src) - frag_len)
            frag = alien_src[src_start:src_start+frag_len]
            dst = random.randint(0, len(s4))
            for k in reversed(range(frag_len)):
                s4.insert(dst, frag[k])
            p4_s.append(s4)
            p4_u.append(u)

    def get_mean_feats(seqs, users):
        if not seqs: return np.array([])
        padded = []
        for seq in seqs:
            if len(seq) > window_size:
                padded.append(seq[-window_size:])
            else:
                pad_len = window_size - len(seq)
                padded.append([(0,0,0)]*pad_len + list(seq))
                
        feats_list = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(padded), batch_size):
                batch = torch.tensor(padded[i:i+batch_size], dtype=torch.long).to(device)
                u_t = torch.tensor(users[i:i+batch_size], dtype=torch.long).to(device)
                
                h = model.extract_features(batch, u_t)
                feats = torch.mean(h, dim=1) # Average over sequence to get holistic representation
                feats_list.append(feats.cpu().numpy())
        return np.concatenate(feats_list, axis=0)

    f_norm = get_mean_feats(bg_seq, bg_u)
    f_t1 = get_mean_feats(p1_s, p1_u)
    f_t2 = get_mean_feats(p2_s, p2_u)
    f_t3 = get_mean_feats(p3_s, p3_u)
    f_t4 = get_mean_feats(p4_s, p4_u)
    f_real = get_mean_feats(real_seq, real_u)
    
    if len(f_norm) == 0: return
    
    # --------------------------------------------------------------------------------
    # SUPERVISED DIMENSIONALITY REDUCTION (LDA)
    # PCA fails because the biggest variance in the dataset is 'User ID' or 'Time of day', 
    # so anomalies get buried. LDA specifically finds the axes that separate the ANOMALY TYPES!
    # --------------------------------------------------------------------------------
    
    X_train = np.vstack([f_norm, f_t1, f_t2, f_t3, f_t4])
    y_train = np.concatenate([
        np.zeros(len(f_norm)),
        np.ones(len(f_t1)),
        np.full(len(f_t2), 2),
        np.full(len(f_t3), 3),
        np.full(len(f_t4), 4)
    ])
    
    lda = LDA(n_components=2)
    lda.fit(X_train, y_train)
    
    proj_norm = lda.transform(f_norm)
    proj_t1 = lda.transform(f_t1)
    proj_t2 = lda.transform(f_t2)
    proj_t3 = lda.transform(f_t3)
    proj_t4 = lda.transform(f_t4)
    proj_real = lda.transform(f_real) if len(f_real) > 0 else []

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot Normal
    ax.scatter(proj_norm[:, 0], proj_norm[:, 1], c='#DDDDDD', label='Normal (Base State)', alpha=0.4, s=40, zorder=1)
    norm_center = np.mean(proj_norm, axis=0)
    ax.scatter(norm_center[0], norm_center[1], c='black', marker='P', s=400, label='Normal Centroid', zorder=5)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    labels = ['Pseudo 1: Semantic Shift', 'Pseudo 2: Event Burst', 'Pseudo 3: Off-hours', 'Pseudo 4: Alien Fragment']
    
    for proj, c, l in zip([proj_t1, proj_t2, proj_t3, proj_t4], colors, labels):
        if len(proj) > 0:
            ax.scatter(proj[:, 0], proj[:, 1], c=c, label=l, alpha=0.6, s=50, zorder=2)
            
            pseudo_center = np.mean(proj, axis=0)
            ax.annotate('', xy=pseudo_center, xytext=norm_center,
                        arrowprops=dict(arrowstyle="->", color=c, lw=3, alpha=0.9), zorder=3)
            
            # Confidence ellipse
            cov = np.cov(proj.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            from matplotlib.patches import Ellipse
            ell = Ellipse(xy=pseudo_center, width=np.sqrt(eigenvalues[0])*4, height=np.sqrt(eigenvalues[1])*4,
                          angle=angle, edgecolor=c, fc='none', lw=2, linestyle='--', alpha=0.7, zorder=4)
            ax.add_patch(ell)
            
    # Plot Real Anomalies
    if len(proj_real) > 0:
        ax.scatter(proj_real[:, 0], proj_real[:, 1], c='#d62728', label='Real CERT Anomalies', 
                   alpha=1.0, s=400, marker='*', edgecolor='black', linewidths=2.0, zorder=10)
        
    ax.set_title("LDA: Discriminative Deviation Directions and Real Anomaly Alignment", fontsize=18)
    ax.set_xlabel("Linear Discriminant 1 (Max Anomaly Separation)", fontsize=14)
    ax.set_ylabel("Linear Discriminant 2", fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Latent Space Type Alignment (Supervised Projection)', fontsize=24, fontweight='bold', y=0.98)
    
    save_dir = os.path.join("results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "LogCapMamba_Type_Alignment.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[LogCapMamba] Type alignment visualization saved to {save_path}")
