import os

import numpy as np
import pandas as pd
import seaborn
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.dataset import CERDataset
from src.mlp import MLPLucas
from src.saets import AutoEncoder, FinalModule

seaborn.set()
torch.manual_seed(32)
np.random.seed(32)

HORIZONS = 12
WINDOW = 3
FORWARD_EXPANSION = 1
N_LAYERS = 1
DROPOUT = 0.0
DECOMP_METHOD = 'fft'
DECOMP = True
DEVICE = torch.device('cuda')

YELLOW = '\033[93m'
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
ENDC = '\033[0m'

PARQUES_DICT = {
    'Amontada': ['BC', 'IG', 'RIB'],
    'Caldeirao': ['Santa_Angelina', 'Santa_Barbara',
                  'Santa_Edwiges', 'Santa_Fatima',
                  'Santa_Regina', 'Santo_Adriano',
                  'Santo_Albano'],
    'Icarai': ['Icarai_I', 'Icarai_II'],
    'Riachao': ['Riachao_I', 'Riachao_II', 'Riachao_IV',
                'Riachao_VI', 'Riachao_VII'],
    'Taiba': ['Aguia', 'Andorinha', 'Colonia']
}

MESES = ['Sep', 'Oct', 'Nov', 'Dec']
TIME_STEPS = ['30_min', '1_day']


def save_pred(pred, horizon, dir_name):
    y = [None]*horizon + pred.tolist()
    df = pd.DataFrame(y, columns=[f'horizonte_{horizon}'])
    file_name = dir_name.split('/')[-1] + f'_{horizon+1}' + '.csv'
    df.to_csv(f'{dir_name}/{file_name}')


def main():
    for complexo, centrais in PARQUES_DICT.items():
        print(f'{RED}Rodando Complexo de {complexo}:{ENDC}')
        for time_step in TIME_STEPS:
            print(f'{YELLOW}Rodando Time step {time_step}:{ENDC}')
            for mes in MESES:
                print(f'{BLUE}Rodando Mes {mes}:{ENDC}')
                for central in centrais:
                    print(f'{GREEN}Rodando Central {central}:{ENDC}')
                    if not os.path.exists(f'data/out/{central}_{mes}_{time_step}_SAETS'):
                        os.system(
                            f'mkdir data/out/{central}_{mes}_{time_step}_SAETS')
                    if not os.path.exists(f'data/out/{central}_{mes}_{time_step}_Cabral'):
                        os.system(
                            f'mkdir data/out/{central}_{mes}_{time_step}_Cabral')
                    dir_name = f'data/out/{central}_{mes}_{time_step}'
                    dataset = CERDataset(window=WINDOW,
                                         horizons=HORIZONS,
                                         complexo_eolico=complexo,
                                         central_eolica=central,
                                         time_step=time_step,
                                         mes=mes,
                                         decomp=DECOMP,
                                         decomp_method=DECOMP_METHOD)

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

                    dataset_lucas = CERDataset(window=WINDOW,
                                               horizons=HORIZONS,
                                               complexo_eolico=complexo,
                                               central_eolica=central,
                                               time_step=time_step,
                                               mes=mes,
                                               decomp=DECOMP,
                                               decomp_method='2fft')

                    train_loader_lucas = DataLoader(dataset_lucas, batch_size=128,
                                                    shuffle=True, num_workers=8)
                    input_example_lucas = next(iter(train_loader_lucas))[0]
                    mlp = MLPLucas(window_size=input_example_lucas.shape[1],
                                   n_comps=input_example_lucas.shape[2],
                                   horizons=HORIZONS)
                    trainer = Trainer(gpus=1, max_epochs=5)
                    trainer.fit(mlp, train_dataloader=train_loader_lucas)

                    dataset.set_type('test')
                    dataset_lucas.set_type('test')
                    mlp = mlp.cpu()
                    final_model = final_model.cpu()
                    X_test = dataset.samples
                    X_test_lucas = dataset_lucas.samples
                    y = dataset.labels
                    y_final = final_model(X_test).detach() / \
                        dataset.test_scaler.scale_
                    y_mlp = mlp(X_test_lucas).detach() / \
                        dataset_lucas.test_scaler.scale_

                    y_final = y_final.numpy()
                    y_mlp = y_mlp.numpy()
                    for i in range(y.shape[1]):
                        save_pred(y_final[:, i], horizon=i,
                                  dir_name=dir_name + '_SAETS')
                        save_pred(y_mlp[:, i], horizon=i,
                                  dir_name=dir_name + '_Cabral')


if __name__ == "__main__":

    main()
