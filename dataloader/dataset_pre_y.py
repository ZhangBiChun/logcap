import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from dataloader.augmentations import DataTransform
from dataloader.generate_negative import *
from sklearn.model_selection import train_test_split
from utils import subsequences, calculate_fid, calculate_mahalanobis
from merlion.transform.normalize import MeanVarNormalize, MinMaxNormalize
from merlion.utils import TimeSeries
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


class Load_Dataset(Dataset):
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X = dataset["samples"]              
        y_value = dataset["pre_y_value"]   
        y_label = dataset["pre_y_labels"]   

        if len(X.shape) < 3:
            X = X.unsqueeze(2)

        self.x_data = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.float()
        self.y_value = torch.from_numpy(y_value).float() if isinstance(y_value, np.ndarray) else y_value.float()
        self.y_label = torch.from_numpy(y_label).long() if isinstance(y_label, np.ndarray) else y_label.long()

        self.len = self.x_data.shape[0]

        
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_value[index], self.y_label[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_value[index], self.y_label[index]

    def __len__(self):
        return self.len

# according to window size and time step to create subsequence
def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        print('Array dimension error')
        os.exit()
    # print(sq.strides)
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq
def generate_target_y(data, window_size, horize, time_step):
    N = (data.shape[0] - window_size - horize + 1) // time_step
    D = data.shape[1]
    y = np.zeros((N, horize, D))  # horize 维度
    for i in range(N):
        start_idx = i * time_step + window_size
        end_idx = start_idx + horize
        if end_idx <= len(data):
            y[i, :, :] = data[start_idx:end_idx]
    return y

def generate_target_label(labels, window_size, horize, time_step):
    N = (labels.shape[0] - window_size - horize + 1) // time_step
    y_label = np.zeros((N, horize))  # horize 维度
    for i in range(N):
        start_idx = i * time_step + window_size
        end_idx = start_idx + horize
        if end_idx <= len(labels):
            y_label[i, :] = labels[start_idx:end_idx].reshape(-1)
    return y_label

# def generate_target_y(data, window_size, horize, time_step):
#     N = (data.shape[0] - window_size - horize + 1) // time_step
#     D = data.shape[1]
#     y = np.zeros((N, 1, D))
#     for i in range(N):
#         idx = i * time_step + window_size + horize - 1
#         if idx < len(data):
#             y[i, 0, :] = data[idx]
#     return y

# def generate_target_label(labels, window_size, horize, time_step):
#     N = (labels.shape[0] - window_size - horize + 1) // time_step
#     y_label = np.zeros((N,))
#     for i in range(N):
#         idx = i * time_step + window_size + horize - 1
#         if idx < len(labels):
#             y_label[i] = int(labels[idx])
#     return y_label

def data_generator_pre(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)

    train_y_value = generate_target_y(train_data, configs.window_size, configs.horize, configs.time_step)
    test_y_value = generate_target_y(test_data, configs.window_size, configs.horize, configs.time_step)

    train_y_label = generate_target_label(train_labels, configs.window_size, configs.horize, configs.time_step)
    test_y_label = generate_target_label(test_labels, configs.window_size, configs.horize, configs.time_step)

    
    min_len = min(len(train_x), len(train_y_value), len(train_y_label))
    train_x = train_x[:min_len].transpose(0, 2, 1)
    train_y_value = train_y_value[:min_len]
    train_y_label = train_y_label[:min_len]

    min_len = min(len(test_x), len(test_y_value), len(test_y_label))
    test_x = test_x[:min_len].transpose(0, 2, 1)
    test_y_value = test_y_value[:min_len]
    test_y_label = test_y_label[:min_len]

    
    _, val_x, _, val_y_value, _, val_y_label = train_test_split(
        test_x, test_y_value, test_y_label,
        test_size=0.2, shuffle=True, random_state=seed,
    )

    
    train_dict = {"samples": train_x, "pre_y_value": train_y_value, "pre_y_labels": train_y_label}
    val_dict = {"samples": val_x, "pre_y_value": val_y_value, "pre_y_labels": val_y_label}
    test_dict = {"samples": test_x, "pre_y_value": test_y_value, "pre_y_labels": test_y_label}

    
    train_dataset = Load_Dataset(train_dict, configs)
    val_dataset = Load_Dataset(val_dict, configs)
    test_dataset = Load_Dataset(test_dict, configs)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size,
                              shuffle=True, drop_last=configs.drop_last, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size,
                            shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader, test_loader, test_anomaly_window_num
