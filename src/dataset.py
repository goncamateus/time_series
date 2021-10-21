import os

import mat73
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.utils.data

from src.preprocess import prepare_data, prepare_data_bench, prepare_data_relatorio


class Dataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self, window, horizon, data_name="wecar", data_dir="./data"):
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir

        file_path = os.path.join(data_dir, data_name, "{}.txt".format(data_name))

        rawdata = np.loadtxt(open(file_path), delimiter=",")
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num - 1))
        Y = torch.zeros((self.sample_num, self.horizon))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :-1])
            Y[i, :] = torch.from_numpy(data[end : end + self.horizon, -1])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx]]

        return sample


class CERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        window,
        horizons,
        complexo_eolico="Icarai",
        central_eolica="Icarai_I",
        time_step="30_min",
        mes="dec",
        data_dir="./data/cer/",
        decomp=True,
        decomp_method="fft",
        type="train",
    ):

        self.window = window
        self.horizons = horizons
        self.horizon_i = 0
        self.data_dir = data_dir
        self.type = type
        self.decomp_method = decomp_method

        central_dir = central_eolica + "_SCADA_" + time_step
        file_path = os.path.join(data_dir, complexo_eolico, central_dir)
        data_name = central_dir + "_" + "OP" + mes.title()
        train_name = data_name + "_Cal.csv"
        test_name = data_name + "_Op.csv"

        train_name = os.path.join(file_path, train_name)
        test_name = os.path.join(file_path, test_name)

        train_data = pd.read_csv(train_name)
        test_data = pd.read_csv(test_name)

        self.df_train = train_data.dropna()
        self.df_test = test_data.dropna()

        if time_step == "30_min":
            self.df_train = self.df_train.tail(2928)

        train_data = self.df_train["PotTotal"].values
        test_data = self.df_test["PotTotal"].values

        (
            X_train,
            y_train,
            X_test,
            y_test,
            train_scaler,
            test_scaler,
        ) = prepare_data_relatorio(
            train_data,
            test_data,
            decomp,
            data_name,
            window,
            horizons,
            decomp_method,
            time_step=1/50
        )

        self.train_scaler = train_scaler
        self.test_scaler = test_scaler

        self.X_train_all = X_train
        self.y_train_all = y_train
        self.X_test_all = X_test
        self.y_test_all = y_test

        self.X_train = torch.Tensor(np.asarray(self.X_train_all[self.horizon_i]))
        self.y_train = torch.Tensor(np.asarray(self.y_train_all[self.horizon_i]))
        self.X_test = torch.Tensor(np.asarray(self.X_test_all[self.horizon_i]))
        self.y_test = torch.Tensor(np.asarray(self.y_test_all[self.horizon_i]))

        self.train_name = train_name
        self.test_name = test_name

        if self.type == "train":
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def set_type(self, type="train"):
        self.type = type
        if self.type == "train":
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def set_horizon(self, horizon):
        self.horizon_i = horizon
        self.X_train = torch.Tensor(np.asarray(self.X_train_all[self.horizon_i]))
        self.y_train = torch.Tensor(np.asarray(self.y_train_all[self.horizon_i]))
        self.X_test = torch.Tensor(np.asarray(self.X_test_all[self.horizon_i]))
        self.y_test = torch.Tensor(np.asarray(self.y_test_all[self.horizon_i]))
        if self.type == "train":
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


class GoncaDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(
        self,
        window,
        horizons,
        data_name="wecar",
        data_dir="./data/inputs",
        decomp=True,
        decomp_method="fft",
        type="train",
    ):
        self.window = window
        self.horizons = horizons
        self.data_dir = data_dir
        self.type = type
        self.decomp_method = decomp_method

        file_path = os.path.join(data_dir, "{}.mat".format(data_name))
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
                serie_nan = np.array(matfile["P"], dtype=np.float32)
        else:
            serie_nan = matfile["SCADA"]["PotTotal"]["bruta"]["DADOS"]["avg"].T
        rawdata = serie_nan[~np.isnan(serie_nan)]
        X_train, y_train, X_test, y_test, train_scaler, test_scaler = prepare_data(
            rawdata, decomp, data_name, window, horizons, decomp_method
        )

        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.train_scaler = train_scaler
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        self.test_scaler = test_scaler

        if self.type == "train":
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def set_type(self, type="train"):
        self.type = type
        if self.type == "train":
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


class BenchDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(
        self,
        window,
        horizons,
        data_name="wecar",
        data_dir="./data/inputs",
        decomp=True,
        decomp_method="fft",
        type="train",
    ):
        self.window = window
        self.horizons = horizons
        self.data_dir = data_dir
        self.type = type
        self.decomp_method = decomp_method

        file_path = os.path.join(data_dir, "{}_treino.csv".format(data_name))
        train_data = pd.read_csv(file_path)[data_name].values

        file_path = os.path.join(data_dir, "{}_teste.csv".format(data_name))
        test_data = pd.read_csv(file_path)[data_name].values

        (
            X_train,
            y_train,
            X_test,
            y_test,
            train_scaler,
            test_scaler,
        ) = prepare_data_bench(
            train_data, test_data, decomp, data_name, window, horizons, decomp_method
        )

        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.train_scaler = train_scaler
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        self.test_scaler = test_scaler

        if self.type == "train":
            self.samples = self.X_train
            self.labels = self.y_train
            self.sample_num = len(self.X_train)
        else:
            self.samples = self.X_test
            self.labels = self.y_test
            self.sample_num = len(self.X_test)

    def set_type(self, type="train"):
        self.type = type
        if self.type == "train":
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

    def __init__(
        self,
        window,
        horizon,
        data_name="wecar",
        set_type="train",  # 'train'/'validation'/'test'
        data_dir="./data",
    ):
        assert type(set_type) == type("str")
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(
            data_dir, data_name, "{}_{}.txt".format(data_name, set_type)
        )

        rawdata = np.loadtxt(open(file_path), delimiter=",").reshape(-1, 1)
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
            Y[i, :] = torch.from_numpy(data[end : end + self.horizon, :]).squeeze()

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx]]

        return sample
