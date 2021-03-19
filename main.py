import matplotlib.pyplot as plt
import numpy as np
import seaborn
import skill_metrics as sm
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.dataset import GoncaDataset
from src.model import MLP, AutoEncoder, FinalModule

seaborn.set()

HORIZONS = 12
WINDOW = 3
FORWARD_EXPANSION = 1
N_LAYERS = 1
DECOMP_METHOD = 'fft'
DEVICE = torch.device('cuda')


def plot_taylor(refs: dict, predictions_dict: dict):

    taylor_stats = []
    for key, pred in predictions_dict.items():
        model = key.split('_')[0]
        taylor_stats.append(sm.taylor_statistics(pred, refs[model][key], 'data'))

    sdev = np.array([taylor_stats[0]['sdev'][0]]+[x['sdev'][1]
                                                  for x in taylor_stats])
    crmsd = np.array([taylor_stats[0]['crmsd'][0]]+[x['crmsd'][1]
                                                    for x in taylor_stats])
    ccoef = np.array([taylor_stats[0]['ccoef'][0]]+[x['ccoef'][1]
                                                    for x in taylor_stats])

    # To change other params in the plot, check SkillMetrics documentation in
    # https://github.com/PeterRochford/SkillMetrics/wiki/Target-Diagram-Options
    sm.taylor_diagram(sdev, crmsd, ccoef, styleOBS='-',
                      colOBS='g', markerobs='o',
                      titleOBS='Observation',
                      markerLabel=['placeholder']+[
                          k+1 for k, v in predictions_dict.items()])


def main():

    dataset = GoncaDataset(window=WINDOW,
                           horizons=HORIZONS,
                           data_name='G7',
                           decomp_method=DECOMP_METHOD,
                           decomp=False)

    train_loader = DataLoader(dataset, batch_size=128,
                              shuffle=True, num_workers=8)

    input_example = next(iter(train_loader))[0]
    input_size = input_example.shape[1]*input_example.shape[2]

    auto_encoder = AutoEncoder(input_size=input_size,
                               horizons=HORIZONS, device=DEVICE,
                               forward_expansion=FORWARD_EXPANSION,
                               num_layers=N_LAYERS)

    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(auto_encoder, train_dataloader=train_loader)

    final_model = FinalModule(input_size=input_size,
                              horizons=HORIZONS, device=DEVICE,
                              forward_expansion=FORWARD_EXPANSION,
                              num_layers=N_LAYERS)
    final_model.load_encoder(auto_encoder.encoder)
    final_trainer = Trainer(gpus=1, max_epochs=5)
    final_trainer.fit(final_model, train_dataloader=train_loader)
    final_model.cuda()

    mlp = MLP(input_size=input_size, horizons=HORIZONS)
    mlp_trainer = Trainer(gpus=1, max_epochs=5)
    mlp_trainer.fit(mlp, train_dataloader=train_loader)
    mlp.cuda()

    dataset.set_type('test')
    X_test = dataset.samples
    y = dataset.labels
    y_final = final_model(X_test.to(DEVICE)).detach().cpu().numpy()
    y_mlp = mlp(X_test.to(DEVICE)).detach().cpu().numpy()
    preds = {f'T_{i}': y_final[:, i] for i in range(HORIZONS)}
    for i in range(HORIZONS):
        preds[f'M_{i}'] = y_mlp[:, i]
    refs = {key: {i: y[:, i].numpy() for i in range(HORIZONS)} for key in ['T', 'M']}
    plot_taylor(refs, preds)

    plt.show()


if __name__ == '__main__':
    main()
