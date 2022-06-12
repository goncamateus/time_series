import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.decomposition import decomp, get_periods


def get_comps(serie: np.ndarray, sub=False, sampling=30 / 525600):
    periods = get_periods(serie, sampling=sampling)
    plus_str = "sub comp" if sub else "serie"
    print(f"Decomp {plus_str} in {periods.size} components")
    components = decomp(serie, periods)

    return components, periods


def get_sub_comps(main_components: np.ndarray, sampling=30 / 525600):
    sub_comps = []
    for comp in main_components.transpose():
        comps, sub_periods = get_comps(comp, sub=True, sampling=sampling)
        sub_comps.append((comps, sub_periods))
    return sub_comps


def prepare_train(train_data, sampling=30 / 525600):
    main_components, periods = get_comps(train_data, sampling=sampling)
    sub_comps = get_sub_comps(main_components, sampling=sampling)
    sub_periods = [sub[1] for sub in sub_comps]
    sub_comps = [sub[0] for sub in sub_comps]
    decomposed = list()
    for comps in sub_comps:
        decomposed += list(comps.T)
    decomposed = np.array(decomposed).T

    return decomposed, periods, sub_periods


def create_test_series(train, test, test_shift=2999):
    test_series = []
    before_series = train[-test_shift:]
    for i in range(test.shape[0]):
        series = np.concatenate((before_series, [test[i]]))
        test_series.append(series)
        before_series = np.concatenate((before_series, [test[i]]))
        before_series = before_series[1:]
    return test_series


def get_comps_test(serie: np.ndarray, periods: list, sub_periods: list):
    main_components = decomp(serie, periods)
    sub_comps = list()
    for i, comp in enumerate(main_components.transpose()):
        sub = decomp(comp, sub_periods[i])
        sub_comps.append(sub)
    return sub_comps


def do_job(serie, periods, sub_periods):
    return get_comps_test(serie, periods, sub_periods)


def prepare_test(train, test, periods, sub_periods, test_shift):
    print('Preparing test data')
    to_decomp = create_test_series(train, test, test_shift)
    test_list_all = Parallel(n_jobs=-1)(
        delayed(do_job)(serie, periods, sub_periods) for serie in tqdm(to_decomp)
    )
    test_list = []
    for decomposed in test_list_all:
        squeezed_comps = []
        for comps in decomposed:
            squeezed_comps += list(comps.T)
        squeezed_comps = np.array(squeezed_comps).T
        test_list.append(squeezed_comps)
    test_list = np.array(test_list)
    decomposed = test_list[:, -1]
    return decomposed


def set_data(original, decomposed, reg_vars, horizons):
    X = []
    y = list()
    for i in range(horizons):
        obs_y = original[reg_vars + i :]
        y.append(obs_y)

    for hi in range(horizons):
        X.append([])
        for i in range(reg_vars, decomposed.shape[0] - hi):
            X[hi].append(decomposed[i - reg_vars : i])

    return np.array(X, dtype=np.ndarray), np.array(y, dtype=np.ndarray)


def prepare_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    reg_vars: int,
    horizons: int,
    sampling=30 / 525600,
) -> tuple:

    train_scaler = MinMaxScaler()
    test_scaler = MinMaxScaler()
    train_data = train_scaler.fit_transform(train_data.reshape(-1, 1)).squeeze()
    test_data = test_scaler.fit_transform(test_data.reshape(-1, 1)).squeeze()


    train_decomposed, periods, sub_periods = prepare_train(train_data, sampling)
    test_decomposed = prepare_test(
        train_data, test_data, periods, sub_periods, test_data.shape[0] - 1
    )

    X_train, y_train = set_data(
        train_data, train_decomposed, reg_vars=reg_vars, horizons=horizons
    )
    X_test, y_test = set_data(
        test_data, test_decomposed, reg_vars=reg_vars, horizons=horizons
    )

    return X_train, y_train, X_test, y_test, train_scaler, test_scaler
