import os
import re

file_path = r'c:\Users\Zhang\Desktop\baseline\1\tsad\models\LogCapMamba\trainer.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find('def _generate_tsne_visualization(')

new_func = """def _generate_tsne_visualization(
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
"""

content = content[:start_idx] + new_func
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Trainer patched successfully.")