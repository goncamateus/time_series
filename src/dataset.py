import os

import mat73
import numpy as np
import scipy.io
import torch
import torch.utils.data
from src.preprocess import prepare_data


class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                 window,
                 horizon,
                 data_name='wecar',
                 set_type='train',    # 'train'/'validation'/'test'
                 data_dir='./data'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(
            data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))

        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num-1))
        Y = torch.zeros((self.sample_num, self.horizon))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :-1])
            Y[i, :] = torch.from_numpy(data[end:end+self.horizon, -1])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx]]

        return sample


class GoncaDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                 window,
                 horizons,
                 data_name='wecar',
                 data_dir='./data/inputs',
                 decomp=True,
                 decomp_method='fft',
                 type='train'):
        self.window = window
        self.horizons = horizons
        self.data_dir = data_dir
        self.type = type
        self.decomp_method = decomp_method

        file_path = os.path.join(data_dir, '{}.mat'.format(data_name))
        matlab73 = False
        try:
            matfile = scipy.io.loadmat(file_path)
        except NotImplementedError:
            matlab73 = True
            matfile = mat73.loadmat(file_path)
        if not matlab73:
            try:
                serie_nan = np.array(matfile[data_name], dtype=np.float32)
            except KeyError:
                serie_nan = np.array(matfile['P'], dtype=np.float32)
        else:
            serie_nan = matfile['SCADA']['PotTotal']['bruta']['DADOS']['avg'].T
        rawdata = serie_nan[~np.isnan(serie_nan)]
        X_train, y_train, X_test, y_test,\
            train_scaler, test_scaler = prepare_data(rawdata, decomp,
                                                     data_name, window,
                                                     horizons, decomp_method)

        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.train_scaler = train_scaler
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        self.test_scaler = test_scaler

        if self.type == 'train':
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def set_type(self, type='train'):
        self.type = type
        if self.type == 'train':
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx]]

        return sample


class UTSFDataset(torch.utils.data.Dataset):
    """Univariate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                 window,
                 horizon,
                 data_name='wecar',
                 set_type='train',    # 'train'/'validation'/'test'
                 data_dir='./data'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(
            data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))

        rawdata = np.loadtxt(open(file_path), delimiter=',').reshape(-1, 1)
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, 1))
        Y = torch.zeros((self.sample_num, self.horizon))

        for i in range(self.sample_num):
            start = i
            end = i + self.window

            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :] = torch.from_numpy(data[end:end+self.horizon, :]).squeeze()

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx]]

        return sample
