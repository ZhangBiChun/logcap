import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def anomaly_split(anomalies):
    """
    Splits anomalies into contiguous segments.

    Parameters:
        anomalies (list or np.array): Indices of anomalies.

    Returns:
        list: List of tuples representing start and end indices of anomaly segments.
    """
    if len(anomalies) == 0:
        return []

    anomaly_segments = []
    start = anomalies[0]

    for i in range(1, len(anomalies)):
        if anomalies[i] != anomalies[i - 1] + 1:  
            anomaly_segments.append((start, anomalies[i - 1]))
            start = anomalies[i]
    anomaly_segments.append((start, anomalies[-1]))

    return anomaly_segments

def anomaly_vis(data, feature_cols, anomaly_segments, vis_type="point", context_size=100):
    """
    Visualizes anomalies in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing features and labels.
        feature_cols (list): List of feature column names to visualize.
        anomaly_segments (list): List of tuples representing start and end indices of anomaly segments.
        vis_type (str): Visualization type - "point" for anomaly points, "segment" for anomaly segments.
        context_size (int): Number of data points to include as context around anomalies.
    """
    for idx, (start, end) in enumerate(anomaly_segments):
        anomalyP_count = 0
        anomaly_length = end - start + 1
        anomalyP_count += anomaly_length
        context_size = max(100, anomaly_length)
        left = max(0, start - context_size)
        right = min(len(data) - 1, end + context_size)

        plt.figure(figsize=(12, 6))
        for feature in feature_cols:
            plt.plot(data.index[left:right+1], data[feature][left:right+1], label=feature, alpha=0.7)

        if vis_type == "point" and anomaly_length == 1:
            plt.axvline(x=start, color='red', linestyle='--', label="Anomaly Point")
            plt.title(f"Anomalous Point {idx + 1}: {start}")
        elif vis_type == "segment":
            plt.axvspan(start, end, color='red', alpha=0.3, label="Anomaly Segment")
            plt.title(f"Anomalous Segment {idx + 1}: {start} to {end}")

        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Feature Values")
        plt.show()
    
    return anomalyP_count

# file_path = "./data/wadi/WADI_test.csv"  
file_path = "./data/swat/SWaT_test.csv"  
# file_path = "./data/iops_competition/phase2_test.csv"

df = pd.read_csv(file_path)
df.index = range(len(df))
print("Columns in the dataset:", df.columns)
print("Shape of the dataset:", df.shape)

if 'label' not in df.columns:
    raise ValueError(" 'label' is not exist in the dataset")

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    df['timestamp'] = range(len(df))

    # KPI or SWaT,WADI
if 'KPI ID' in df.columns:
    # 按 KPI ID 分组
    grouped = df.groupby('KPI ID')

    # 遍历每个 KPI 组
    for kpi_id, group in grouped:
        print(f"Analyzing KPI ID: {kpi_id}")
        
        # 设置索引
        group.index = range(len(group))
        feature_col = group.columns[1]
        print("Group: ", group.columns)
        print(f"Feature shape for KPI {kpi_id}: ", group.shape)

        # 找到异常点
        anomalies = group[group['label'] == 1].index

        if anomalies.size == 0:
            print(f"No anomalies found for KPI {kpi_id}")
            continue
        else:
            anomaly_segments = []
            anomaly_count = 0
            anomalyP_count = 0

            anomaly_segments = anomaly_split(anomalies)
            anomalyP_count = anomaly_vis(group, [ feature_col ], anomaly_segments, "segment")

            print(f"Total anomaly points for KPI {kpi_id}: ", anomaly_count)
            print(f"Total anomalyp for KPI {kpi_id}: ", anomalyP_count)

else:

    feature_cols = df.columns[1:-1]  
    print("Feature shape: ", df.shape)

    anomalies = df[df['label'] == 1].index

    if anomalies.size == 0:
        print("No anomalies found in the dataset")
    else:
        anomaly_segments = []
        anomaly_count = 0
        anomalyP_count = 0
        start = anomalies[0]

        anomaly_segments = anomaly_split(anomalies)
        anomalyP_count = anomaly_vis(df, feature_cols, anomaly_segments, "segment")
        
        print("Total anomaly points: ", anomaly_count)
        print("Total anomalyp ", anomalyP_count)
