# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn
import skill_metrics as sm
import torch
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_lightning import Trainer
from torch.nn.functional import mse_loss as MSE
from torch.utils.data import DataLoader

from src.dataset import GoncaDataset
from src.model import MLP, AutoEncoder, FinalModule

seaborn.set()
torch.manual_seed(32)
np.random.seed(32)

ds = ['GUNNING1', 'KIATAWF1', 'MERCER01', 'MUSSELR1',
      'NBHWF1', 'STARHLWF', 'WATERLWF', 'YAMBUKWF']
MLP_MODEL = {DATASET: {key: list() for key in ['mse', 'rmse', 'mae']} for DATASET in ds}
TAETS = {DATASET: {key: list() for key in ['mse', 'rmse', 'mae']} for DATASET in ds}
MLP_PURE = {DATASET: {key: list() for key in ['mse', 'rmse', 'mae']} for DATASET in ds}
TAETS_PURE = {DATASET: {key: list() for key in ['mse', 'rmse', 'mae']} for DATASET in ds}
HORIZONS = 12
WINDOW = 3
FORWARD_EXPANSION = 1
N_LAYERS = 1
DROPOUT = 0.0
DECOMP_METHOD = 'fft'
DEVICE = torch.device('cuda')


# %%
for DATASET in ds:
    print(f'-------> DATASET {DATASET}')
    if not os.path.exists(f'data/components/{DATASET}'):
        os.system(f'mkdir data/components/{DATASET}')
    if not os.path.exists(f'data/out/{DATASET}'):
        os.system(f'mkdir data/out/{DATASET}')

    # %%
    dataset = GoncaDataset(window=WINDOW,
                           horizons=HORIZONS,
                           data_name=DATASET,
                           decomp_method=DECOMP_METHOD,
                           decomp=True)

    train_loader = DataLoader(dataset, batch_size=128,
                              shuffle=True, num_workers=8)

    input_example = next(iter(train_loader))[0]
    input_size = input_example.shape[1]*input_example.shape[2]

    auto_encoder = AutoEncoder(input_size=input_size,
                               horizons=HORIZONS, device=DEVICE,
                               forward_expansion=FORWARD_EXPANSION,
                               num_layers=N_LAYERS,
                               dropout=DROPOUT)

    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(auto_encoder, train_dataloader=train_loader)

    final_model = FinalModule(input_size=input_size,
                              horizons=HORIZONS, device=DEVICE,
                              forward_expansion=FORWARD_EXPANSION,
                              num_layers=N_LAYERS,
                              dropout=DROPOUT)
    final_model.load_encoder(auto_encoder.encoder)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(final_model, train_dataloader=train_loader)

    mlp = MLP(input_size=input_size, horizons=HORIZONS)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(mlp, train_dataloader=train_loader)

    dataset.set_type('test')
    mlp = mlp.cpu()
    final_model = final_model.cpu()
    X_test = dataset.samples
    y = dataset.labels/dataset.test_scaler.scale_
    y_final = final_model(X_test).detach()/dataset.test_scaler.scale_
    y_mlp = mlp(X_test).detach()/dataset.test_scaler.scale_
    # preds = {}
    # preds['Transformer'] = {i: y_final[:, i] for i in range(HORIZONS)}
    # preds['MLP'] = {i: y_mlp[:, i] for i in range(HORIZONS)}
    # refs = {key: {i: y[:, i] for i in range(HORIZONS)} for key in ['T', 'M']}
    for i in range(HORIZONS):
        mse = MSE(y_final[:, i], y[:, i]).item() 
        TAETS[DATASET]['mse'].append(mse)
        rmse = RMSE().loss(y_final[:, i], y[:, i]).mean().item()
        TAETS[DATASET]['rmse'].append(rmse)
        mae = MAE().loss(y_final[:, i], y[:, i]).mean().item()
        TAETS[DATASET]['mae'].append(mae)

    for i in range(HORIZONS):
        mse = MSE(y_mlp[:, i], y[:, i]).item() 
        MLP_MODEL[DATASET]['mse'].append(mse)
        rmse = RMSE().loss(y_mlp[:, i], y[:, i]).mean().item()
        MLP_MODEL[DATASET]['rmse'].append(rmse)
        mae = MAE().loss(y_mlp[:, i], y[:, i]).mean().item()
        MLP_MODEL[DATASET]['mae'].append(mae)

    y_final = y_final.numpy()
    y_mlp = y_mlp.numpy()
    y = y.numpy()
    preds = {'Nosso': y_final, 'MLP': y_mlp}
    for method, prediction in preds.items():
        sio.savemat(f'data/out/{DATASET}/{DATASET}_{method}_decomp.mat',
                    {DATASET: prediction})

    # %%
    # No decomposition

    dataset_pure = GoncaDataset(window=WINDOW,
                                horizons=HORIZONS,
                                data_name=DATASET,
                                decomp_method='nodecomp',
                                decomp=True)

    train_loader_pure = DataLoader(dataset_pure, batch_size=128,
                                   shuffle=True, num_workers=8)

    input_example = next(iter(train_loader_pure))[0]
    input_size = input_example.shape[1]*input_example.shape[2]

    auto_encoder_pure = AutoEncoder(input_size=input_size,
                                    horizons=HORIZONS, device=DEVICE,
                                    forward_expansion=FORWARD_EXPANSION,
                                    num_layers=N_LAYERS,
                                    dropout=DROPOUT)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(auto_encoder_pure, train_dataloader=train_loader_pure)

    final_model_pure = FinalModule(input_size=input_size,
                                   horizons=HORIZONS, device=DEVICE,
                                   forward_expansion=FORWARD_EXPANSION,
                                   num_layers=N_LAYERS,
                                   dropout=DROPOUT)
    final_model_pure.load_encoder(auto_encoder_pure.encoder)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(final_model_pure, train_dataloader=train_loader_pure)

    mlp_pure = MLP(input_size=input_size, horizons=HORIZONS)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(mlp_pure, train_dataloader=train_loader_pure)

    # # %%
    dataset_pure.set_type('test')
    mlp_pure = mlp_pure.cpu()
    final_model_pure = final_model_pure.cpu()
    X_test_pure = dataset_pure.samples
    y_pure = dataset_pure.labels/dataset_pure.test_scaler.scale_
    y_final_pure = final_model_pure(X_test_pure).detach()/dataset_pure.test_scaler.scale_
    y_mlp_pure = mlp_pure(X_test_pure).detach()/dataset_pure.test_scaler.scale_

    for i in range(HORIZONS):
        mse = MSE(y_final_pure[:, i], y_pure[:, i]).item() 
        TAETS_PURE[DATASET]['mse'].append(mse)
        rmse = RMSE().loss(y_final_pure[:, i], y_pure[:, i]).mean().item()
        TAETS_PURE[DATASET]['rmse'].append(rmse)
        mae = MAE().loss(y_final_pure[:, i], y_pure[:, i]).mean().item()
        TAETS_PURE[DATASET]['mae'].append(mae)

    for i in range(HORIZONS):
        mse = MSE(y_mlp_pure[:, i], y_pure[:, i]).item() 
        MLP_PURE[DATASET]['mse'].append(mse)
        rmse = RMSE().loss(y_mlp_pure[:, i], y_pure[:, i]).mean().item()
        MLP_PURE[DATASET]['rmse'].append(rmse)
        mae = MAE().loss(y_mlp_pure[:, i], y_pure[:, i]).mean().item()
        MLP_PURE[DATASET]['mae'].append(mae)

    y_pure = y_pure.numpy()
    y_final_pure = y_final_pure.numpy()
    y_mlp_pure = y_mlp_pure.numpy()
    preds = {'Nosso': y_final_pure, 'MLP': y_mlp_pure}
    for method, prediction in preds.items():
        sio.savemat(f'data/out/{DATASET}/{DATASET}_{method}_pure.mat',
                    {DATASET: prediction})

    sio.savemat(f'data/out/{DATASET}/{DATASET}_ref.mat', {DATASET: y_pure})

tables = list()
for METRIC in ['mae', 'mse', 'rmse']:
    print('-----------------------------------------------------------------')
    print()
    table = ''
    for DATASET in ds:
        a = f"\multicolumn{'{1}'}{'{|c|}'}{'{'}{'t'}extbf{'{'}{DATASET}{'}'}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_PURE[DATASET][METRIC][0], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS_PURE[DATASET][METRIC][0], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_MODEL[DATASET][METRIC][0], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS[DATASET][METRIC][0], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_PURE[DATASET][METRIC][5], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS_PURE[DATASET][METRIC][5], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_MODEL[DATASET][METRIC][5], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS[DATASET][METRIC][5], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_PURE[DATASET][METRIC][11], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS_PURE[DATASET][METRIC][11], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(MLP_MODEL[DATASET][METRIC][11], 2)}{'}'} & \multicolumn{'{1}'}{'{|c|}'}{'{'}{round(TAETS[DATASET][METRIC][11], 2)}{'}'} \\\ \hline"
        table += a + '\n'
    with open(f'{METRIC}.txt', 'w') as table_file:
        table_file.write(table)
    print(table)
    tables.append(table)
    print('-----------------------------------------------------------------')
    print()
