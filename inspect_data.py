import csv
from collections import Counter
import numpy as np

def inspect_custom_csv(csv_path):
    print(f"--- Inspecting {csv_path} ---")
    seq_lengths = []
    labels = []
    
    actions = set()
    hours = set()
    sems = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # First element is label
                label = int(row[0])
                labels.append(label)
                
                # The rest are events 'a_h_s', stopping at first empty string
                length = 0
                for item in row[1:]:
                    if not item: 
                        break
                    length += 1
                    try:
                        a, h, s = item.split('_')
                        actions.add(int(a))
                        hours.add(int(h))
                        sems.add(int(s))
                    except:
                        pass
                seq_lengths.append(length)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    print(f"Total sessions (rows): {len(seq_lengths)}")
    
    seq_lengths = np.array(seq_lengths)
    print(f"Sequence length - Mean: {seq_lengths.mean():.2f}, Median: {np.median(seq_lengths):.2f}, Max: {seq_lengths.max()}, Min: {seq_lengths.min()}")
    print(f"Sequence length - 90th percentile: {np.percentile(seq_lengths, 90):.2f}")
    print(f"Sequence length - 95th percentile: {np.percentile(seq_lengths, 95):.2f}")
    print(f"Sequence length - 99th percentile: {np.percentile(seq_lengths, 99):.2f}")
    
    print(f"Action vocab: {len(actions)} (Max ID: {max(actions) if actions else 0})")
    print(f"Hour vocab: {len(hours)} (Max ID: {max(hours) if hours else 0})")
    print(f"Semantic vocab: {len(sems)} (Max ID: {max(sems) if sems else 0})")
    
    anomalies = sum(labels)
    print(f"Total anomalous sessions: {anomalies} ({(anomalies/len(labels)*100):.4f}%)\n")


inspect_custom_csv('data/itdbert_dual/THNS24_2010.csv')
inspect_custom_csv('data/itdbert_dual/THNS24_2011.csv')
