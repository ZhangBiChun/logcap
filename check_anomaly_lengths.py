import numpy as np
import pandas as pd
from dataloader.data_preprocessing import thns24

train_data, test_data, train_labels, test_labels = thns24(
    train_csv='data/itdbert_dual/THNS24_2010.csv',
    test_csv='data/itdbert_dual/THNS24_2011_userlabel.csv',
    max_sessions=None,
    sample_mode='full',
    seed=6
)

labels = np.asarray(test_labels).reshape(-1)
anomaly_indices = np.where(labels == 1)[0]
normal_indices = np.where(labels == 0)[0]

print(f"Total anomalies: {len(anomaly_indices)}")
print(f"Total normals: {len(normal_indices)}")

anomaly_lengths = [len(test_data[i]) for i in anomaly_indices]
normal_lengths = [len(test_data[i]) for i in normal_indices]

print("\nAnomaly lengths statistics:")
print(pd.Series(anomaly_lengths).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\nNormal lengths statistics:")
print(pd.Series(normal_lengths).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

short_anomalies = sum(1 for l in anomaly_lengths if l <= 64)
print(f"\nAnomalies with length <= 64: {short_anomalies} out of {len(anomaly_lengths)}")
