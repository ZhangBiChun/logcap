import csv
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize


def norm(train, test):
    scaler = StandardScaler()
    # scaler.fit(np.concatenate((train, test), axis=0))
    scaler.fit(train)
    train_data = scaler.transform(train)
    test_data = scaler.transform(test)
    return train_data, test_data

def swat_preprocessing():
    # the following code is adapted from the source code in [Zhihan Li et al. KDD21]
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    dataset_folder = os.path.join('../data/', 'swat')

    test_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx'), header=1)

    test_df = test_df.set_index(' Timestamp')
    test_df['label'] = np.where(test_df['Normal/Attack'] == 'Attack', 1, 0)
    # test_df.apply(lambda x: 1 if test_df['Normal/Attack'] == 'Attack' else 0)
    test_df = test_df.drop('Normal/Attack', axis=1)
    assert test_df.shape == (449919, 52)

    train_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v0.xlsx'), header=1)
    # train_df = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 52'])
    train_df = train_df.set_index(' Timestamp')
    train_df['label'] = np.where(train_df['Normal/Attack'] == 'Attack', 1, 0)
    train_df = train_df.drop('Normal/Attack', axis=1)

    # following [Zhihan Li et al. KDD21] & [Dan Li. ICANN. 2019]
    # fow SWaT data, due to the cold start of the system, starting point is 21600
    train_df = train_df.iloc[21600:]
    assert train_df.shape == (475200, 52)

    output_dir = '../data/swat/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'SWaT_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'SWaT_test.csv'))


def wadi_preprocessing():
    # preprocess for WADI. WADI.A2_19Nov2019
    dataset_folder = os.path.join('../data/', 'wadi')

    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'), index_col=0, header=0)
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'), index_col=0, header=1)

    train_df = train_df.iloc[:, 2:]
    test_df = test_df.iloc[:, 2:]

    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # trim column names
    train_df = train_df.rename(columns=lambda x: x.strip())
    test_df = test_df.rename(columns=lambda x: x.strip())

    train_df['label'] = np.zeros(len(train_df))
    test_df['label'] = np.where(test_df['Attack LABLE (1:No Attack, -1:Attack)'] == -1, 1, 0)
    test_df = test_df.drop(columns=['Attack LABLE (1:No Attack, -1:Attack)'])

    output_dir = '../data/wadi/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'WADI_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'WADI_test.csv'))

def swat():
    dataset_folder = os.path.join('data/', 'swat')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_train.csv'))
    train_df = np.array(train_df.set_index(' Timestamp'))
    train_data = train_df[:, :51]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_test.csv'))
    test_df = np.array(test_df.set_index(' Timestamp'))
    test_data = test_df[:, :51]

    # scaler = StandardScaler()
    # scaler.fit(np.concatenate((train_data, test_data), axis=0))
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)
    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 51]
    test_labels = test_df[:, 51]

    return train_data, test_data, train_labels, test_labels


def wadi():
    dataset_folder = os.path.join('data/', 'wadi')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_train.csv'))
    train_df = np.array(train_df.set_index('Row'))
    train_data = train_df[:, :127]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_test.csv'))
    # print(test_df.columns)
    test_df = np.array(test_df.set_index('Row '))
    test_data = test_df[:, :127]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 127]
    test_labels = test_df[:, 127]

    return train_data, test_data, train_labels, test_labels

def esa():
    dataset_folder = os.path.join('data/', 'esa')
    train_data = pd.read_csv(os.path.join(dataset_folder, 'esa_train.csv'))
    train_data = np.array(train_data.set_index('datetime'))
    print(train_data.shape)
    test_data = pd.read_csv(os.path.join(dataset_folder, 'esa_test.csv'))
    test_data = np.array(test_data.set_index('datetime'))
    print(test_data.shape)
    train_labels = pd.read_csv(os.path.join(dataset_folder, 'esa_train_labels.csv'))
    train_labels = np.array(train_labels.set_index('datetime'))

    test_labels = pd.read_csv(os.path.join(dataset_folder, 'esa_test_labels.csv'))  
    test_labels = np.array(test_labels.set_index('datetime'))

    print(test_data.shape)
    train_data, test_data = norm(train_data, test_data)

    return train_data, test_data, train_labels, test_labels


# Extract datasets processed by salesforce-merlion such as IOpsCompetition, UCR, NAB, SWaT, WADI, SMAP, etc.
def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    # salesforce-merlion==1.1.1
    # bias, scale = mvn.bias, mvn.scale

    # salesforce-merlion==1.3.0
    # bias, scale = mvn.bias[0], mvn.scale[0]
    

    # salesforce-merlion==2.0.0
    bias, scale = mvn.bias, mvn.scale
    bias, scale = list(bias.values())[0], list(scale.values())[0]

    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_data = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_data = (test_time_series - bias) / scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def other_datasets_no_Normalize(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    train_data = train_time_series_ts.to_pd().to_numpy()
    test_data = test_time_series_ts.to_pd().to_numpy()

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def _parse_thns24_event_csv(csv_path, label_csv_path=None, max_rows=None):
    sequences = []
    labels = []
    user_ids = []
    limit = None
    if max_rows is not None:
        try: limit = int(max_rows)
        except Exception: limit = None
        
    user_labels = {}
    if label_csv_path and os.path.exists(label_csv_path):
        with open(label_csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2: continue
                user = row[0]
                user_labels[user] = [int(l) for l in row[1:] if l]

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: continue
            user = row[0]
            seq = []
            for token in row[1:]:
                token = token.strip()
                if not token: continue
                try:
                    parts = list(map(int, token.split("_")))
                    if len(parts) >= 3:
                        act, hour, sem = parts[0], parts[1], parts[2]
                        seq.append((act, hour, sem))
                except Exception:
                    continue
            if not seq: continue
            
            sequences.append(seq)
            user_ids.append(user)
            if user in user_labels and len(user_labels[user]) == len(seq):
                labels.append(user_labels[user])
            else:
                # If no labels (like in train), we just pad with 0s. 
                # (Train set uses pure normal sequences anyway, so all labels are 0).
                labels.append([0] * len(seq))
                
            if limit is not None and limit > 0 and len(sequences) >= limit:
                break
                
    return sequences, labels, user_ids


def thns24_event_level(
    train_csv="data/itdbert_event/THNS24_2010.csv",
    test_csv="data/itdbert_event/THNS24_2011.csv",
    test_label_csv="data/itdbert_event/THNS24_2011_event_labels.csv",
    max_sessions=None,
    sample_mode="full",
    seed=42,
):
    read_cap = max_sessions if str(sample_mode).lower() == "head" else None
    train_data, train_labels, train_user_ids = _parse_thns24_event_csv(train_csv, max_rows=read_cap)
    test_data, test_labels, test_user_ids = _parse_thns24_event_csv(
        test_csv, label_csv_path=test_label_csv, max_rows=read_cap
    )

    if str(sample_mode).lower() == "head" and max_sessions is not None:
        n = int(max_sessions)
        if n > 0:
            train_data, train_labels, train_user_ids = train_data[:n], train_labels[:n], train_user_ids[:n]
            test_data, test_labels, test_user_ids = test_data[:n], test_labels[:n], test_user_ids[:n]

    return train_data, test_data, train_labels, test_labels, train_user_ids, test_user_ids


def _stratified_sample_sessions(sequences, labels, max_sessions, seed=42):
    if max_sessions is None:
        return sequences, labels
    try:
        n = int(max_sessions)
    except Exception:
        return sequences, labels
    if n <= 0 or len(sequences) <= n:
        return sequences, labels

    labels = np.asarray(labels).reshape(-1)
    idx_all = np.arange(len(sequences))
    idx_0 = idx_all[labels == 0]
    idx_1 = idx_all[labels == 1]

    rng = np.random.default_rng(seed)

    # Keep label distribution as much as possible while guaranteeing at least 1 positive if available.
    p1 = len(idx_1) / max(1, len(sequences))
    n1 = int(round(n * p1))
    if len(idx_1) > 0:
        n1 = max(1, n1)
    n1 = min(n1, len(idx_1))
    n0 = min(n - n1, len(idx_0))

    # Fill any remaining quota from whichever class has remaining samples.
    remain = n - (n0 + n1)
    if remain > 0:
        room0 = len(idx_0) - n0
        add0 = min(remain, max(0, room0))
        n0 += add0
        remain -= add0
    if remain > 0:
        room1 = len(idx_1) - n1
        add1 = min(remain, max(0, room1))
        n1 += add1

    choose0 = rng.choice(idx_0, size=n0, replace=False) if n0 > 0 else np.asarray([], dtype=np.int64)
    choose1 = rng.choice(idx_1, size=n1, replace=False) if n1 > 0 else np.asarray([], dtype=np.int64)
    picked = np.concatenate([choose0, choose1], axis=0)
    rng.shuffle(picked)

    sampled_sequences = [sequences[i] for i in picked.tolist()]
    sampled_labels = labels[picked]
    return sampled_sequences, sampled_labels


def thns24(
    train_csv="data/itdbert/THNS24_2010.csv",
    test_csv="data/itdbert/THNS24_2011.csv",
    max_sessions=None,
    sample_mode="stratified",
    seed=42,
):
    # For head mode, stop early while reading for speed.
    read_cap = max_sessions if str(sample_mode).lower() == "head" else None
    train_data, train_labels = _parse_thns24_csv(train_csv, max_rows=read_cap)
    test_data, test_labels = _parse_thns24_csv(test_csv, max_rows=read_cap)

    mode = str(sample_mode).lower()
    if mode == "stratified":
        train_data, train_labels = _stratified_sample_sessions(train_data, train_labels, max_sessions, seed=seed)
        test_data, test_labels = _stratified_sample_sessions(test_data, test_labels, max_sessions, seed=seed + 1)
    elif mode == "head" and max_sessions is not None:
        n = int(max_sessions)
        if n > 0:
            train_data, train_labels = train_data[:n], train_labels[:n]
            test_data, test_labels = test_data[:n], test_labels[:n]
    # mode == "full": keep all

    return train_data, test_data, train_labels, test_labels


def thns24_adapt_for_model(
    train_sessions,
    test_sessions,
    train_session_labels,
    test_session_labels,
    univariate=False,
):
    """
    Keep THNS24 raw session data unchanged in storage, and adapt at model-input stage.

    Returns:
        train_data: np.ndarray [N_train_events, D]
        test_data:  np.ndarray [N_test_events, D]
        train_labels: np.ndarray [N_train_events]
        test_labels:  np.ndarray [N_test_events]
    """

    def _session_to_features(session):
        feats = []
        n = len(session)
        for idx, (a, h, s) in enumerate(session):
            ah = float(a)
            hh = float(h)
            ss = float(s)

            # Preserve session boundary/position information for window-based models.
            is_start = 1.0 if idx == 0 else 0.0
            is_end = 1.0 if idx == (n - 1) else 0.0
            rel_pos = float(idx) / float(max(1, n - 1))

            # Cyclic hour encoding keeps temporal periodicity without dropping raw hour.
            hour_sin = np.sin(2.0 * np.pi * hh / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hh / 24.0)

            # Keep raw act/hour/sem as first 3 channels (no information dropped).
            feats.append([ah, hh, ss, is_start, is_end, rel_pos, hour_sin, hour_cos])
        if len(feats) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def _flatten(sessions, labels):
        x_list, y_list = [], []
        for seq, lb in zip(sessions, labels):
            f = _session_to_features(seq)
            if f.shape[0] == 0:
                continue
            x_list.append(f)
            if isinstance(lb, (list, tuple, np.ndarray)):
                lb_arr = np.asarray(lb, dtype=np.int64)
                if len(lb_arr) == f.shape[0]:
                    y_list.append(lb_arr)
                else:
                    y_list.append(np.full((f.shape[0],), int(lb_arr[0] if len(lb_arr) > 0 else 0), dtype=np.int64))
            else:
                y_list.append(np.full((f.shape[0],), int(lb), dtype=np.int64))
        if not x_list:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    train_x, train_y = _flatten(train_sessions, train_session_labels)
    test_x, test_y = _flatten(test_sessions, test_session_labels)

    if train_x.shape[0] > 0:
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True) + 1e-8
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

    if univariate:
        # Lossless packing for single-channel models (SR/DAMP):
        # packed_id = act*10000 + hour*100 + sem
        # ranges in THNS24 ensure reversibility and no collisions.
        train_x = (train_x[:, 0] * 10000.0) + (train_x[:, 1] * 100.0) + train_x[:, 2]
        test_x = (test_x[:, 0] * 10000.0) + (test_x[:, 1] * 100.0) + test_x[:, 2]

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    # swat_preprocessing()
    wadi_preprocessing()
