import os
import random
import torch
import numpy as np
import csv

# We load the normal dataset
def parse_thns24_event_csv(filepath):
    sessions = []
    user_strs = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            u_str = row[0]
            events = row[1:]
            parsed_seq = []
            for e in events:
                parts = e.split('_')
                if len(parts) == 3:
                    parsed_seq.append((int(parts[0]), int(parts[1]), int(parts[2])))
            sessions.append(parsed_seq)
            user_strs.append(u_str)
    return sessions, user_strs

# =====================================================================
# LLM Agent Playbooks (Extracted from CERT r4.2 Red Team Scenarios)
# =====================================================================

def playbook_1_leak(window_size):
    """
    Scenario 1: Off-hours, Wikileaks, USB exfiltration
    Inner threat logs on at 2 AM, connects a USB, copies files, uploads to wikileaks, and disconnects.
    """
    h = random.randint(0, 5) # off hours
    num_events = random.randint(10, 20)
    seq = []
    seq.append((6, h, 0)) # connect USB
    for _ in range(num_events - 3):
        if random.random() < 0.5:
            seq.append((5, h, random.choice([12, 13]))) # file doc/zip
        else:
            seq.append((3, h, random.choice([3, 4]))) # http leak/cloud
    seq.append((7, h, 0)) # disconnect USB
    
    # Pad to exact window size
    seq = seq[-window_size:]
    pad_len = window_size - len(seq)
    if pad_len > 0:
        seq = [(0,0,0)] * pad_len + seq
    return seq

def playbook_2_job_exfil(window_size):
    """
    Scenario 2: Job sites and Mass Data Exfiltration
    User browses job sites, then performs a massive burst of file copying and sending external emails.
    """
    h = random.randint(8, 18)
    num_events = random.randint(40, window_size)
    seq = []
    seq.append((3, h, 2)) # http job
    for _ in range(num_events - 2):
        if random.random() < 0.8:
            seq.append((5, h, random.choice([11, 12, 13]))) # mass file copy
        else:
            seq.append((4, h, 22)) # mass email ext
    seq.append((4, h, 22))
    
    seq = seq[-window_size:]
    pad_len = window_size - len(seq)
    if pad_len > 0:
        seq = [(0,0,0)] * pad_len + seq
    return seq

def playbook_3_hack_lateral(window_size, pool):
    """
    Scenario 3: Keylogger / Lateral Movement
    User visits hacker site, runs exe, and suddenly their behavior matches an IT Admin (Alien Fragment).
    """
    h = random.randint(0, 23)
    seq = [(3, h, 5), (5, h, 14)] # Http Hack -> File Exe
    if pool:
        alien = random.choice(pool)
        frag_len = random.randint(20, window_size - 2)
        start = random.randint(0, max(0, len(alien) - frag_len))
        seq += list(alien[start:start+frag_len])
        
    seq = seq[-window_size:]
    pad_len = window_size - len(seq)
    if pad_len > 0:
        seq = [(0,0,0)] * pad_len + seq
    return seq

def playbook_4_off_hours_burst(window_size, normal_chunk):
    """
    LLM Extracted Behavior: Time Shift + Burst
    User takes their normal afternoon tasks, shifts them to 3 AM, and does them 3x faster (bursts).
    """
    h_new = random.randint(0, 5)
    seq = []
    for (a, h, s) in normal_chunk:
        seq.append((a, h_new, s))
        if random.random() < 0.3: # Burst injection
            seq.append((a, h_new, s))
            seq.append((a, h_new, s))
    
    seq = seq[-window_size:]
    pad_len = window_size - len(seq)
    if pad_len > 0:
        seq = [(0,0,0)] * pad_len + seq
    return seq

def playbook_5_alien_injection(window_size, normal_chunk, pool):
    """
    LLM Extracted Behavior: Credential Compromise / Account Takeover
    Normal context is suddenly interrupted by 20-40 completely alien events from another user's session.
    """
    seq = list(normal_chunk)
    if pool:
        alien = random.choice(pool)
        frag_len = random.randint(20, 40)
        start = random.randint(0, max(0, len(alien) - frag_len))
        frag = alien[start:start+frag_len]
        insert_pos = random.randint(0, len(seq))
        seq = seq[:insert_pos] + list(frag) + seq[insert_pos:]
        
    seq = seq[-window_size:]
    pad_len = window_size - len(seq)
    if pad_len > 0:
        seq = [(0,0,0)] * pad_len + seq
    return seq

def generate_agent_aug_dataset():
    window_size = 64
    print("[LLM Agent] Loading THNS24_2010.csv normal data...")
    filepath = "data/itdbert_event/THNS24_2010.csv"
    sessions, user_strs = parse_thns24_event_csv(filepath)
    
    # Map users exactly as baselines.py does
    unique_users = sorted(list(set(user_strs)))
    user2idx = {u: i+1 for i, u in enumerate(unique_users)} # 1-indexed
    
    normal_chunks = []
    normal_chunks_u = []
    
    print("[LLM Agent] Chunking data into 64-event windows...")
    for seq, u_str in zip(sessions, user_strs):
        u_idx = user2idx[u_str]
        for i in range(0, len(seq), window_size):
            chunk = seq[i:i+window_size]
            if len(chunk) == window_size:
                normal_chunks.append(chunk)
                normal_chunks_u.append(u_idx)
                
    aug_train_data = []
    aug_train_labels = []
    aug_train_u = []
    
    print("[LLM Agent] Synthesizing Threat Scenarios via AI Playbooks...")
    for ci, (chunk, u) in enumerate(zip(normal_chunks, normal_chunks_u)):
        # Keep the normal chunk (Label 0)
        aug_train_data.append(chunk)
        aug_train_labels.append(0)
        aug_train_u.append(u)
        
        # 50% chance to generate a synthetic anomaly for this normal context (Label 1)
        if random.random() < 0.5:
            pb_type = random.randint(1, 5)
            if pb_type == 1:
                pseudo = playbook_1_leak(window_size)
            elif pb_type == 2:
                pseudo = playbook_2_job_exfil(window_size)
            elif pb_type == 3:
                pseudo = playbook_3_hack_lateral(window_size, normal_chunks)
            elif pb_type == 4:
                pseudo = playbook_4_off_hours_burst(window_size, chunk)
            else:
                pseudo = playbook_5_alien_injection(window_size, chunk, normal_chunks)
                
            aug_train_data.append(pseudo)
            aug_train_labels.append(1)
            aug_train_u.append(u)
            
    # Save the exactly formatted cache for LogCapMamba
    cache_path = "best_network/THNS24/logcapmamba_aug_data.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"[LLM Agent] Synthesized {len(aug_train_data)} total chunks (Normals + Anomalies).")
    print(f"[LLM Agent] Saving ready-to-train tensor cache to {cache_path}...")
    torch.save({
        'data': aug_train_data,
        'labels': aug_train_labels,
        'u': aug_train_u
    }, cache_path)
    print("[LLM Agent] Data Augmentation Complete! Ready for Model Training.")

if __name__ == "__main__":
    generate_agent_aug_dataset()