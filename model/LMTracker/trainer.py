import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from gensim.models import Word2Vec
import networkx as nx

from .model import LMTrackerAutoEncoder
from models import ad_predict
from models.reasonable_metric import reasonable_accumulator
from models.thns24_personalized_scores import maybe_personalize_scores

def build_graph_and_embed(train_data, test_data, train_u, test_u, embed_dim=64):
    print("[LMTracker] Building heterogeneous graph...")
    # We build a graph of Users and Event-Types
    # Event-Type node: "E_{act}_{hour}_{sem}"
    # User node: "U_{uid}"
    
    G = nx.DiGraph()
    
    def add_seq_to_graph(seq, uid):
        u_node = f"U_{uid}"
        prev_e_node = None
        for (a, h, s) in seq:
            e_node = f"E_{a}_{h}_{s}"
            # Edge: User -> Event
            if G.has_edge(u_node, e_node):
                G[u_node][e_node]['weight'] += 1
            else:
                G.add_edge(u_node, e_node, weight=1)
                
            # Edge: Event -> Event (sequential transition)
            if prev_e_node is not None:
                if G.has_edge(prev_e_node, e_node):
                    G[prev_e_node][e_node]['weight'] += 1
                else:
                    G.add_edge(prev_e_node, e_node, weight=1)
            prev_e_node = e_node

    for seq, u in zip(train_data, train_u):
        add_seq_to_graph(seq, u)
    for seq, u in zip(test_data, test_u):
        add_seq_to_graph(seq, u)
        
    print(f"[LMTracker] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("[LMTracker] Generating random walks for Graph Embedding...")
    walks = []
    nodes = list(G.nodes())
    for _ in range(10): # num_walks
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            curr = node
            for _ in range(20): # walk_length
                neighbors = list(G.successors(curr))
                if not neighbors:
                    break
                weights = [G[curr][n]['weight'] for n in neighbors]
                total_w = sum(weights)
                probs = [w / total_w for w in weights]
                curr = np.random.choice(neighbors, p=probs)
                walk.append(curr)
            walks.append(walk)
            
    print("[LMTracker] Training Word2Vec (DeepWalk/Node2Vec equivalent)...")
    w2v = Word2Vec(sentences=walks, vector_size=embed_dim, window=5, min_count=1, sg=1, workers=4)
    return w2v.wv

def get_path_vectors(seq, u, w2v, window_size, embed_dim):
    u_node = f"U_{u}"
    u_emb = w2v[u_node] if u_node in w2v else np.zeros(embed_dim)
    
    padded_seq = list(seq)
    if len(padded_seq) <= window_size:
        pad_len = window_size + 1 - len(padded_seq)
        padded_seq = [(0, 0, 0)] * pad_len + padded_seq
        
    num_windows = len(padded_seq) - window_size
    path_vectors = []
    
    for i in range(num_windows):
        window = padded_seq[i:i+window_size]
        embs = []
        for (a, h, s) in window:
            e_node = f"E_{a}_{h}_{s}"
            if e_node in w2v:
                embs.append(w2v[e_node])
        if embs:
            path_emb = np.mean(embs, axis=0)
        else:
            path_emb = np.zeros(embed_dim)
            
        combined = np.concatenate([u_emb, path_emb])
        path_vectors.append(combined)
        
    return np.array(path_vectors, dtype=np.float32), num_windows, len(seq)

def Trainer(train_data, test_data, train_labels, test_labels, config):
    print("LMTracker training started ....")
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "cpu")
    
    train_u = getattr(config, "thns24_train_user_idx", [0]*len(train_data))
    test_u = getattr(config, "thns24_test_user_idx", [0]*len(test_data))
    
    embed_dim = 64
    w2v = build_graph_and_embed(train_data, test_data, train_u, test_u, embed_dim=embed_dim)
    
    train_vectors = []
    for seq, u in tqdm(zip(train_data, train_u), total=len(train_data), desc="[LMTracker] Encoding Train Paths"):
        p_vecs, _, _ = get_path_vectors(seq, u, w2v, config.window_size, embed_dim)
        if len(p_vecs) > 0:
            train_vectors.append(p_vecs)
            
    train_vectors = np.concatenate(train_vectors)
    train_tensor = torch.from_numpy(train_vectors).to(device)
    
    model = LMTrackerAutoEncoder(input_dim=embed_dim*2, hidden_dim=config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    batch_size = max(1024, config.batch_size)
    dataset = torch.utils.data.TensorDataset(train_tensor)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(config.num_epoch):
        running_loss = 0.0
        for (x,) in dl:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[LMTracker][Epoch {epoch+1:03d}] Loss: {running_loss/len(dl):.6f}")
        
    model.eval()
    all_test_scores = []
    all_test_labels = []
    
    with torch.no_grad():
        for seq, lbl, u in tqdm(zip(test_data, test_labels, test_u), total=len(test_data), desc="[LMTracker] Inferring"):
            p_vecs, num_windows, seq_len = get_path_vectors(seq, u, w2v, config.window_size, embed_dim)
            if len(p_vecs) == 0:
                scores = np.zeros(seq_len, dtype=np.float32)
            else:
                x = torch.from_numpy(p_vecs).to(device)
                out = model(x)
                mse = torch.mean((out - x)**2, dim=1).cpu().numpy()
                
                if seq_len <= config.window_size:
                    scores = np.full(seq_len, mse[0] if len(mse)>0 else 0.0, dtype=np.float32)
                else:
                    pad = np.zeros(config.window_size, dtype=np.float32)
                    scores = np.concatenate([pad, mse]).astype(np.float32)
            
            all_test_scores.append(scores)
            all_test_labels.append(lbl)
            
    test_scores = np.concatenate(all_test_scores).astype(np.float32)
    flat_test_labels = np.concatenate(all_test_labels).astype(np.int64)
    
    def _score_train_session(seq, uid: int) -> np.ndarray:
        p_vecs, num_windows, seq_len = get_path_vectors(seq, uid, w2v, config.window_size, embed_dim)
        if len(p_vecs) == 0:
            return np.zeros(seq_len, dtype=np.float32)
        x = torch.from_numpy(p_vecs).to(device)
        with torch.no_grad():
            out = model(x)
            mse = torch.mean((out - x)**2, dim=1).cpu().numpy()
        if seq_len <= config.window_size:
            return np.full(seq_len, mse[0] if len(mse)>0 else 0.0, dtype=np.float32)
        else:
            pad = np.zeros(config.window_size, dtype=np.float32)
            return np.concatenate([pad, mse]).astype(np.float32)

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
        log_prefix="LMTracker",
    )

    test_aff, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(
        flat_test_labels, test_scores, config.threshold_determine, config.detect_nu
    )
    score_reasonable = reasonable_accumulator(1, 0)
    print("LMTracker predicting ended ....")

    return test_scores, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict