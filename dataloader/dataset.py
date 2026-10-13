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
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# Gives label values (test_y_window) by time window.
def data_generator1(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


# Gives label values (test_y) by point-wise way.
def data_generator2(train_data, test_data, train_labels, test_labels, seed, configs):

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    _, val_x, _, val_y = train_test_split(test_x, test_y, test_size=0.2, shuffle=True, random_state=seed)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


#
def data_generator3(train_data, test_data, train_labels, test_labels, configs):
    train_time_series_ts = train_data
    test_time_series_ts = test_data

    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)

    # salesforce-merlion==1.1.1
    bias, scale = mvn.bias, mvn.scale

    # salesforce-merlion==1.3.0
    # bias, scale = mvn.bias[0], mvn.scale[0]

    # salesforce-merlion==2.0.0
    # bias, scale = mvn.bias, mvn.scale
    # bias, scale = list(bias.values())[0], list(scale.values())[0]

    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_time_series = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_time_series = (test_time_series - bias) / scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_time_series, configs.window_size, configs.time_step)
    test_x = subsequences(test_time_series, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.2, shuffle=False)

    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier(train_origin, configs.trend_rate)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier(train_origin, configs.trend_rate)
        train_aug_x_2 = cut_add_paste_outlier(train_origin, configs.trend_rate)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    train_aug_x = sample
    train_aug_y = np.zeros(sample_num) + 1

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y_window

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


def data_generator4(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    train_origin = train_x.copy()

    if configs.rate1 != 0:
        train_aug_x1 = point_outliers(train_origin, configs)
        sample_num_1 = int(configs.rate1 * len(train_origin))
        sample_list_1 = [i for i in range(sample_num_1)]
        sample_list_1 = random.sample(sample_list_1, sample_num_1)
        sample_1 = train_aug_x1[sample_list_1, :, :]
    else:
        sample_num_1 = 0
        sample_1 = train_x[[], :, :]

    if configs.rate2 != 0:
        train_aug_x2 = collective_trend_outliers(train_origin, configs)
        sample_num_2 = int(configs.rate2 * len(train_origin))
        sample_list_2 = [i for i in range(sample_num_2)]
        sample_list_2 = random.sample(sample_list_2, sample_num_2)
        sample_2 = train_aug_x2[sample_list_2, :, :]
    else:
        sample_num_2 = 0
        sample_2 = train_x[[], :, :]

    if configs.rate3 != 0:
        train_aug_x3 = cut_outliers(train_origin)
        sample_num_3 = int(configs.rate3 * len(train_origin))
        sample_list_3 = [i for i in range(sample_num_3)]
        sample_list_3 = random.sample(sample_list_3, sample_num_3)
        sample_3 = train_aug_x3[sample_list_3, :, :]
    else:
        sample_num_3 = 0
        sample_3 = train_x[[], :, :]

    if configs.rate4 != 0:
        train_aug_x4 = cut_paste_outliers_same(train_origin)
        sample_num_4 = int(configs.rate4 * len(train_origin))
        sample_list_4 = [i for i in range(sample_num_4)]
        sample_list_4 = random.sample(sample_list_4, sample_num_4)
        sample_4 = train_aug_x4[sample_list_4, :, :]
    else:
        sample_num_4 = 0
        sample_4 = train_x[[], :, :]

    if configs.rate5 != 0:
        train_aug_x5 = cut_paste_outliers_other(train_origin)
        sample_num_5 = int(configs.rate5 * len(train_origin))
        sample_list_5 = [i for i in range(sample_num_5)]
        sample_list_5 = random.sample(sample_list_5, sample_num_5)
        sample_5 = train_aug_x5[sample_list_5, :, :]
    else:
        sample_num_5 = 0
        sample_5 = train_x[[], :, :]

    if configs.rate6 != 0:
        train_aug_x6 = cut_paste_outliers_same_trend(train_origin, configs)
        sample_num_6 = int(configs.rate6 * len(train_origin))
        sample_list_6 = [i for i in range(sample_num_6)]
        sample_list_6 = random.sample(sample_list_6, sample_num_6)
        sample_6 = train_aug_x6[sample_list_6, :, :]
    else:
        sample_num_6 = 0
        sample_6 = train_x[[], :, :]

    train_aug_x = np.concatenate((sample_1, sample_2, sample_3, sample_4, sample_5, sample_6), axis=0)
    train_aug_y = np.zeros(sample_num_1 + sample_num_2 + sample_num_3 + sample_num_4 + sample_num_5 + sample_num_6) + 1

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


#简单mixup和用皮尔逊系数smooth label
def data_generator5(train_data, test_data, train_labels, test_labels, seed, configs):

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.1, shuffle=False)
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    # 改了这个部分
    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]
    #改成在这mixup 悟空跺脚
    train_aug_x = sample
    train_aug_y = np.zeros(sample_num) + 1

#用皮尔逊系数SL
    # train_origin = train_x.copy()
    # y_origin = train_y.copy()

    # if ((configs.rate != 0) and (configs.rate <= 1)):
    #     train_aug_x, train_aug_y = cut_add_paste_outlier_with_sl(train_origin, y_origin, configs)
    #     sample_num = int(configs.rate * len(train_origin))
    #     sample_list = [i for i in range(sample_num)]
    #     sample_list = random.sample(sample_list, sample_num)
    #     sample = train_aug_x[sample_list, :, :]
    # elif configs.rate > 1:
    #     train_aug_x_1, train_aug_y_1 = cut_add_paste_outlier_with_sl(train_origin, y_origin, configs)
    #     train_aug_x_2, train_aug_y_2 = cut_add_paste_outlier_with_sl(train_origin, y_origin, configs)
    #     train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
    #     train_aug_y = np.concatenate((train_aug_y_1, train_aug_y_2), axis=0)
    #     sample_num = int(configs.rate * len(train_origin))
    #     sample_list = [i for i in range(sample_num)]
    #     sample_list = random.sample(sample_list, sample_num)
    #     sample = train_aug_x[sample_list, :, :]
    # else:
    #     sample_num = 0
    #     sample = train_x[[], :, :]
    #     train_aug_y = np.zeros(sample_num) + 1
    # train_aug_x = sample
    # if configs.dataset == 'IOpsCompetition':
    train_normal_index = np.where(train_y == 0)
    print(train_y.shape)
    print(train_x.shape)
    train_normal = train_x[train_normal_index]
    print(train_normal.shape)
    alpha = configs.alpha
    if alpha > 0:
        mix_num = int(alpha * len(train_aug_x))
        mix_list = random.sample(list(train_aug_x), mix_num)
        print(mix_num)
        to_mix_index_list = np.random.choice(range(len(train_normal)), size=mix_num, replace=True)
        # to_mix_index_list = random.sample(range(len(train_normal)), mix_num)
        to_mix_x = train_x[to_mix_index_list]
        to_mix_y = train_y[to_mix_index_list]

        lams = np.random.beta(alpha, alpha, mix_num)
        mixed_y = lams * ((np.zeros(mix_num) + 1)) + (1-lams) * to_mix_y
        lams = lams.reshape(mix_num,1,1)
        mixed_x = lams * np.array(mix_list) + (1-lams) * np.array(to_mix_x)

        train_x = np.concatenate((train_x, mixed_x), axis=0)
        train_y = np.concatenate((train_y, mixed_y), axis=0)

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    #解决样本数除以batch size之后不足1的问题
    num_samples = len(train_x)
    batches = int(num_samples / configs.batch_size)

    if num_samples % configs.batch_size <= 1 :
        train_x = train_x[0:batches * configs.batch_size - 1]  # 去掉最后一个样本
        train_y = train_y[0:batches * configs.batch_size - 1 ] 
    
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


#用马氏距离smooth label
def data_generator_Mah(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.1, shuffle=False)
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
# 原来的版本
#     train_origin = train_x.copy()

#     if ((configs.rate != 0) and (configs.rate <= 1)):
#         train_aug_x = cut_add_paste_outlier(train_origin, configs)
#         sample_num = int(configs.rate * len(train_origin))
#         sample_list = [i for i in range(sample_num)]
#         sample_list = random.sample(sample_list, sample_num)
#         sample = train_aug_x[sample_list, :, :]
#     elif configs.rate > 1:
#         train_aug_x_1 = cut_add_paste_outlier(train_origin, configs)
#         train_aug_x_2 = cut_add_paste_outlier(train_origin, configs)
#         train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
#         sample_num = int(configs.rate * len(train_origin))
#         sample_list = [i for i in range(sample_num)]
#         sample_list = random.sample(sample_list, sample_num)
#         sample = train_aug_x[sample_list, :, :]
#     else:
#         sample_num = 0
#         sample = train_x[[], :, :]
# #改成在这mixup 悟空跺脚
#     train_aug_x = sample


#用马氏距离三角形算软标签
    train_origin = train_x.copy()
    y_origin = train_y.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x, train_aug_y = cut_add_paste_outlier_with_mah(train_origin, y_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1, train_aug_y_1 = cut_add_paste_outlier_with_mah(train_origin, y_origin, configs)
        train_aug_x_2, train_aug_y_2 = cut_add_paste_outlier_with_mah(train_origin, y_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        train_aug_y = np.concatenate((train_aug_y_1, train_aug_y_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]
        train_aug_y = np.zeros(sample_num) + 1

    train_aug_x = sample
    num_lt_1 = np.sum(train_aug_y < 1)
    num_gt_1 = np.sum(train_aug_y > 1)
    train_aug_y[train_aug_y > 1] = 1


    #算距离，fid
    fid_train_this_turn = calculate_fid(train_data, train_aug_x.reshape(-1,train_aug_x.shape[2]), "train")
    fid_test_this_turn = calculate_fid(test_data,   train_aug_x.reshape(-1,train_aug_x.shape[2]), "test")
    test_data_clean = test_data[test_labels != 1]
    fid_test_clean_this_turn = calculate_fid(test_data_clean,  train_aug_x.reshape(-1,train_aug_x.shape[2]), "test clean")
    fid_train_test = calculate_fid(train_data.reshape(-1,), test_data.reshape(-1,), "test and train")

    #算马氏距离及与fid开方比值，获取软标签
    # distances = np.array([calculate_mahalanobis(aux[:,0], train_x[:,:,0]) for  aux in train_aug_x])
    # sq_fid = fid_train_this_turn ** 0.5
    # train_aug_y = distances/sq_fid
    # train_aug_y[distances > 1] = 1
    # num_lt_1 = np.sum(distances < 1)
    # num_gt_1 = np.sum(distances > 1)


    alpha = configs.alpha
    if alpha > 0:
        mix_num = int(alpha * len(train_aug_x))
        mix_list = random.sample(list(train_aug_x), mix_num)
        to_mix_index_list = random.sample(range(len(train_x)), mix_num)
        to_mix_x = train_x[to_mix_index_list]
        to_mix_y = train_y[to_mix_index_list]

        lams = np.random.beta(alpha, alpha, mix_num)
        mixed_y = lams * ((np.zeros(mix_num) + 1)) + (1-lams) * to_mix_y
        lams = lams.reshape(mix_num,1,1)
        mixed_x = lams * np.array(mix_list) + (1-lams) * np.array(to_mix_x)

        train_x = np.concatenate((train_x, mixed_x), axis=0)
        train_y = np.concatenate((train_y, mixed_y), axis=0)

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

#解决样本数除以batch size之后不足1的问题
    num_samples = len(train_x)
    batches = int(num_samples / configs.batch_size)

    if num_samples % configs.batch_size <= 1 :
        train_x = train_x[0:batches * configs.batch_size - 1]  # 去掉最后一个样本
        train_y = train_y[0:batches * configs.batch_size - 1 ] 

#重新归一化
    # scaler = StandardScaler()
    # B,T,D = train_x.shape

    # train_x = train_x.reshape(-1, D)
    # test_x = test_x.reshape(-1, D)
    # val_x = val_x.reshape(-1, D)
    # # scaler.fit(np.concatenate((train, test), axis=0))
    # scaler.fit(train_x)
    # train_x = scaler.transform(train_x)
    # test_x = scaler.transform(test_x)
    # val_x = scaler.transform(val_x)


    # train_x = train_x.reshape(-1,T,D)
    # test_x = test_x.reshape(-1,T,D)
    # val_x = val_x.reshape(-1,T,D)



    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num, fid_train_this_turn, fid_test_this_turn, fid_test_clean_this_turn, num_gt_1, num_lt_1


# 可视化t-sne演示用的
def data_generator6(train_data, test_data, train_labels, test_labels, seed, configs):

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.1, shuffle=False)
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)

    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    train_aug_x = sample
    train_aug_y = np.zeros(sample_num) + 2

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))
    train_aug_x = train_aug_x.transpose((0, 2, 1))

    return train_aug_x, test_x, train_aug_y, test_y


# 原始的cut add paste的方法
def data_generator7(train_data, test_data, train_labels, test_labels, seed, configs):

    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)

    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    train_aug_x = sample
    train_aug_y = np.zeros(sample_num) + 1

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


# 根据DTW对异常增强样本重新给标签
def data_generator8(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    # 改了这个部分
    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier_plus(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier_plus(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier_plus(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    # 改成在这mixup 悟空跺脚
    train_aug_x = sample
    # train_aug_y = np.zeros(sample_num) + 1

    train_normal_index = np.where(train_y == 0)
    # print(train_y.shape)
    # print(train_x.shape)
    train_normal_x = train_x[train_normal_index]
    train_normal_y = np.zeros(len(train_normal_x))
    # print(train_normal.shape)

    center_normal = np.sum(train_normal_x, axis=0) / train_normal_x.shape[0]
    all_distance = []
    for i, x_i in enumerate(sample):
        distance, path = fastdtw.fastdtw(center_normal, x_i)
        all_distance.append(distance)
    all_distance = np.array(all_distance)
    print("all_distance:", all_distance.shape)
    distance_mean = np.mean(all_distance)
    distance_std = np.std(all_distance)
    print("distance_mean:", distance_mean)
    print("distance_std:", distance_std)
    print("distance_max:", np.max(all_distance))
    print("distance_min:", np.min(all_distance))
    gamma = configs.gamma
    condition = all_distance < (distance_mean - gamma * distance_std)
    revise_aug_nor_index = np.where(condition)
    revise_aug_ano_index = np.where(~condition)
    print("revise_aug_nor_index:", len(revise_aug_nor_index))
    print("revise_aug_ano_index:", len(revise_aug_ano_index))

    revise_aug_nor_x = train_aug_x[revise_aug_nor_index]
    revise_aug_ano_x = train_aug_x[revise_aug_ano_index]
    print("revise_aug_nor_x:", revise_aug_nor_x.shape)
    print("revise_aug_ano_x:", revise_aug_ano_x.shape)

    revise_aug_nor_y = np.zeros(len(revise_aug_nor_x))
    revise_aug_ano_y = np.zeros(len(revise_aug_ano_x)) + 1

    alpha = configs.alpha
    if alpha > 0:
        mix_num = int(alpha * len(revise_aug_ano_x))
        mix_list = random.sample(list(revise_aug_ano_x), mix_num)
        to_mix_index_list = np.random.choice(range(len(train_normal_x)), size=mix_num, replace=True)
        # to_mix_index_list = random.sample(range(len(train_normal)), mix_num)
        to_mix_x = train_normal_x[to_mix_index_list]
        to_mix_y = train_normal_y[to_mix_index_list]

        lams = np.random.beta(alpha, alpha, mix_num)
        mixed_y = lams * ((np.zeros(mix_num) + 1)) + (1 - lams) * to_mix_y
        lams = lams.reshape(mix_num, 1, 1)
        mixed_x = lams * np.array(mix_list) + (1 - lams) * np.array(to_mix_x)

        train_x = np.concatenate((train_x, mixed_x), axis=0)
        train_y = np.concatenate((train_y, mixed_y), axis=0)

    train_x = np.concatenate((train_x, revise_aug_nor_x, revise_aug_ano_x), axis=0)
    train_y = np.concatenate((train_y, revise_aug_nor_y, revise_aug_ano_y), axis=0)

    # 解决样本数除以batch size之后不足1的问题
    num_samples = len(train_x)
    batches = int(num_samples / configs.batch_size)

    if num_samples % configs.batch_size <= 1:
        train_x = train_x[0:batches * configs.batch_size - 1]  # 去掉最后一个样本
        train_y = train_y[0:batches * configs.batch_size - 1]

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=True, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


# 根据DTW对异常增强样本重新给标签，并且应用Hidden Mixup
def data_generator9(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    # 改了这个部分
    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier_plus(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier_plus(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier_plus(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    # 改成在这mixup 悟空跺脚
    train_aug_x = sample
    # train_aug_y = np.zeros(sample_num) + 1

    train_normal_index = np.where(train_y == 0)
    # print(train_y.shape)
    # print(train_x.shape)
    train_normal_x = train_x[train_normal_index]
    train_normal_y = np.zeros(len(train_normal_x))
    # print(train_normal.shape)

    center_normal = np.sum(train_normal_x, axis=0) / train_normal_x.shape[0]
    all_distance = []
    for i, x_i in enumerate(sample):
        distance, path = fastdtw.fastdtw(center_normal, x_i)
        all_distance.append(distance)
    all_distance = np.array(all_distance)
    print("all_distance:", all_distance.shape)
    distance_mean = np.mean(all_distance)
    distance_std = np.std(all_distance)
    print("distance_mean:", distance_mean)
    print("distance_std:", distance_std)
    print("distance_max:", np.max(all_distance))
    print("distance_min:", np.min(all_distance))
    gamma = configs.gamma
    condition = all_distance < (distance_mean - gamma * distance_std)
    revise_aug_nor_index = np.where(condition)
    revise_aug_ano_index = np.where(~condition)
    print("revise_aug_nor_index:", len(revise_aug_nor_index))
    print("revise_aug_ano_index:", len(revise_aug_ano_index))

    revise_aug_nor_x = train_aug_x[revise_aug_nor_index]
    revise_aug_ano_x = train_aug_x[revise_aug_ano_index]
    print("revise_aug_nor_x:", revise_aug_nor_x.shape)
    print("revise_aug_ano_x:", revise_aug_ano_x.shape)

    # revise_aug_nor_y = np.zeros(len(revise_aug_nor_x))
    revise_aug_nor_y = np.zeros(len(revise_aug_nor_x)) + 1 / gamma
    revise_aug_ano_y = np.zeros(len(revise_aug_ano_x)) + 1

    alpha = configs.alpha
    if alpha > 0 and configs.layer_mix == 0:
        mix_num = int(alpha * len(revise_aug_ano_x))
        mix_list = random.sample(list(revise_aug_ano_x), mix_num)
        to_mix_index_list = np.random.choice(range(len(train_normal_x)), size=mix_num, replace=True)
        # to_mix_index_list = random.sample(range(len(train_normal)), mix_num)
        to_mix_x = train_normal_x[to_mix_index_list]
        to_mix_y = train_normal_y[to_mix_index_list]

        lams = np.random.beta(alpha, alpha, mix_num)
        mixed_y = lams * ((np.zeros(mix_num) + 1)) + (1 - lams) * to_mix_y
        lams = lams.reshape(mix_num, 1, 1)
        mixed_x = lams * np.array(mix_list) + (1 - lams) * np.array(to_mix_x)

        train_x = np.concatenate((train_x, mixed_x), axis=0)
        train_y = np.concatenate((train_y, mixed_y), axis=0)

    train_x = np.concatenate((train_x, revise_aug_nor_x, revise_aug_ano_x), axis=0)
    train_y = np.concatenate((train_y, revise_aug_nor_y, revise_aug_ano_y), axis=0)

    # 解决样本数除以batch size之后不足1的问题
    num_samples = len(train_x)
    batches = int(num_samples / configs.batch_size)

    if num_samples % configs.batch_size <= 1:
        train_x = train_x[0:batches * configs.batch_size - 1]  # 去掉最后一个样本
        train_y = train_y[0:batches * configs.batch_size - 1]

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=True, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num




