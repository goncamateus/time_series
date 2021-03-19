import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.decomposition import decomp, get_periods


def get_comps(serie: np.ndarray, sub=False):
    periods = get_periods(serie)
    plus_str = 'sub comp' if sub else 'serie'
    print(f'Decomp {plus_str} in {periods.size} components')
    components = decomp(serie, periods)

    return components, periods


def get_sub_comps(main_components: np.ndarray):
    sub_comps = []
    for comp in main_components.transpose():
        comps, sub_periods = get_comps(comp, sub=True)
        sub_comps.append((comps, sub_periods))
    return sub_comps


def get_comps_test(serie: np.ndarray, periods: list, sub_periods: list):
    print('Getting test components')
    main_components = decomp(serie, periods)
    # sub_comps = list()
    # for i, comp in enumerate(main_components.transpose()):
    #     sub = decomp(comp, sub_periods[i])
    #     sub_comps.append(sub)
    return main_components


def save_comps(train_comps: np.ndarray,
               test_comps: np.ndarray,
               central: str):
    path = 'data/components/{}/'.format(central)
    with open(path+'train_comps.pkl', 'wb') as pf:
        pickle.dump(train_comps, pf)
    with open(path+'test_comps.pkl', 'wb') as pf:
        pickle.dump(test_comps, pf)


def load_sub_comps(central: str):
    path = 'data/components/{}/'.format(central)
    with open(path+'train_comps.pkl', 'rb') as pf:
        train_comps = pickle.load(pf)
    with open(path+'test_comps.pkl', 'rb') as pf:
        test_comps = pickle.load(pf)
    return train_comps, test_comps


def set_data(serie: np.ndarray, sub_comps: list,
             reg_vars=60, horizons=12):

    data = sub_comps
    # for comps in sub_comps:
    #     data += list(comps.T)
    # data = np.array(data).T

    X = []
    y = list()
    for i in range(reg_vars, data.shape[0]-horizons+1):
        obs_y = [serie[i+j] for j in range(horizons)]
        y.append(obs_y)

    for i in range(reg_vars, data.shape[0]-horizons+1):
        X.append(data[i-reg_vars:i])

    return np.array(X), np.array(y)


def nodecomp_prepare(serie: np.ndarray, reg_vars: int, horizons: int):
    X = []
    y = list()
    for i in range(reg_vars, serie.shape[0]-horizons+1):
        obs_y = [serie[i+j] for j in range(horizons)]
        y.append(obs_y)

    for i in range(reg_vars, serie.shape[0]-horizons+1):
        X.append(serie[i-reg_vars:i])

    return np.array(X), np.array(y)


def prepare_data(serie: np.ndarray, dec: bool,
                 central: str, reg_vars: int,
                 horizons: int, decomp_method: str) -> tuple:
    """
    Preprocess data and separates it in train and test.
    All the data is normalized.

    Parameters
    ----------
    serie : np.ndarray
        Time Serie data

    dec : bool
        Rather you wanna decompose or not your serie

    central : str
        The dataset name

    reg_vars : int
        How many regvars you want for your model(s)

    horizons : int
        Number of horizons you want to predict

    decomp_method : str
        Lucas method or CWT method

    Returns
    -------
    tuple
        X_train, y_train, X_test, y_test, normalizer scaler
    """
    train_data = serie[:int(len(serie)*2/3)]
    test_data = serie[int(len(serie)*2/3):]
    train_scaler = MinMaxScaler()
    test_scaler = MinMaxScaler()
    train_data = train_scaler.fit_transform(
        train_data.reshape(-1, 1)).squeeze()
    test_data = test_scaler.fit_transform(test_data.reshape(-1, 1)).squeeze()
    if decomp_method == 'fft':
        if dec:
            main_components, periods = get_comps(train_data)
            # sub_comps = get_sub_comps(main_components)
            # sub_periods = [sub[1] for sub in sub_comps]
            # sub_comps = [sub[0] for sub in sub_comps]
            test_components = get_comps_test(
                test_data, periods, periods)
            save_comps(main_components, test_components, central)
        else:
            main_components, test_components = load_sub_comps(central)
        X_train, y_train = set_data(train_data, main_components,
                                    reg_vars=reg_vars, horizons=horizons)
        X_test, y_test = set_data(test_data, test_components,
                                  reg_vars=reg_vars,
                                  horizons=horizons)
    else:
        X_train, y_train = nodecomp_prepare(train_data, reg_vars, horizons)
        X_test, y_test = nodecomp_prepare(test_data, reg_vars, horizons)
        X_train = np.expand_dims(X_train, 2)
        X_test = np.expand_dims(X_test, 2)

    return X_train, y_train, X_test, y_test, train_scaler, test_scaler
